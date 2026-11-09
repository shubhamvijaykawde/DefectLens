"""The Docker contract of Phase 6b, checked without Docker (the real run: scripts/docker_smoke.sh).

These guard the decisions that broke or could break the image: no torch, the CUDA libraries that
the ONNX Runtime ``[cuda]`` extra does not bring (cuBLAS, cuDNN; found only by running the first
image), weights mounted and never copied, a non-root user.
"""

from __future__ import annotations

import importlib
import re
import subprocess
import tomllib
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = (ROOT / "Dockerfile").read_text()
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
GROUPS = PYPROJECT["dependency-groups"]


def names(group: str) -> list[str]:
    return [re.split(r"[\[<>=! ]", spec, maxsplit=1)[0].lower() for spec in GROUPS[group]]


def dockerfile_instructions() -> list[str]:
    """Instructions without comments, continuation lines joined."""
    text = re.sub(r"\\\n", " ", DOCKERFILE)
    lines = [ln.strip() for ln in text.splitlines()]
    return [ln for ln in lines if ln and not ln.startswith("#")]


# ---- dependency groups -----------------------------------------------------------------------


@pytest.mark.parametrize("group", ["serve", "serve-gpu"])
def test_image_groups_have_no_torch_or_tensorrt(group: str) -> None:
    heavy = ("torch", "tensorrt", "transformers", "anomalib")
    bad = [n for n in names(group) if n.startswith(heavy)]
    assert not bad, bad


def test_serve_gpu_has_what_the_cuda_extra_lacks_and_what_the_code_imports() -> None:
    have = set(names("serve-gpu"))
    # `onnxruntime-gpu[cuda]` brings cudart, cuFFT, cuRAND and NVRTC, not these two:
    assert {"onnxruntime-gpu", "nvidia-cublas", "nvidia-cudnn-cu13", "opencv-python-headless",
            "numpy"} <= have
    assert set(names("serve")) >= {"fastapi", "gradio", "uvicorn", "pyyaml", "python-multipart"}


def test_serve_gpu_pins_the_same_onnxruntime_as_the_local_cuda_group() -> None:
    pins = lambda g: [s for s in GROUPS[g] if s.startswith("onnxruntime-gpu")]  # noqa: E731
    assert pins("serve-gpu") == pins("cu130") and len(pins("serve-gpu")) == 1  # noqa: PT018


def test_serve_gpu_can_not_be_installed_with_the_cpu_group() -> None:
    conflicts = PYPROJECT["tool"]["uv"]["conflicts"]
    pairs = {frozenset(c["group"] for c in conflict) for conflict in conflicts}
    assert frozenset({"serve-gpu", "cpu"}) in pairs


# ---- Dockerfile ------------------------------------------------------------------------------


def test_dockerfile_is_multi_stage_and_installs_only_the_serve_groups() -> None:
    ins = dockerfile_instructions()
    assert sum(i.startswith("FROM ") for i in ins) >= 2
    sync = next(i for i in ins if "uv sync" in i)
    assert "--locked" in sync and "--only-group serve-gpu" in sync
    assert re.search(r"--only-group serve(\s|$)", sync)
    assert "--group" not in sync.replace("--only-group", "")
    assert not re.search(r"torch|tensorrt|pip install", "\n".join(ins), re.I)


def test_dockerfile_does_not_bake_in_weights_and_runs_as_non_root() -> None:
    ins = dockerfile_instructions()
    copies = [i for i in ins if i.startswith("COPY") and "--from" not in i]
    assert copies and not any(re.search(r"\b(models|data|mlruns)\b", c) for c in copies)
    assert not any(re.search(r"\.onnx|\.pt\b|\.pth", i) for i in copies)
    user = [i for i in ins if i.startswith("USER ")]
    assert user and user[-1] not in ("USER root", "USER 0")
    assert any(i.startswith("HEALTHCHECK") and "/health" in i for i in ins)
    assert any("/app/models" in i for i in ins)  # the mount point exists and is documented


def test_dockerfile_copies_only_paths_that_exist_and_the_dockerignore_allows_them() -> None:
    lines = (ROOT / ".dockerignore").read_text().splitlines()
    allowed = [ln[1:] for ln in lines if ln.startswith("!")]
    ignore_lines = [ln for ln in lines if ln and ln[0] != "#"]
    assert ignore_lines[0] == "*"  # deny by default
    for ins in dockerfile_instructions():
        if not ins.startswith("COPY") or "--from" in ins:
            continue
        args = [a for a in ins.split()[1:] if not a.startswith("--")]
        for src in args[:-1]:
            path = src.removeprefix("./")
            assert (ROOT / path).exists(), path
            ok = any(path == a or path.startswith(a + "/") for a in allowed)
            assert ok, f"{path} is ignored"
    for must_not in ("models", "data", "mlruns", "outputs", ".git"):
        assert not any(a == must_not or a.startswith(must_not + "/") for a in allowed)


def test_the_container_command_is_an_importable_factory() -> None:
    cmd = next(i for i in dockerfile_instructions() if i.startswith("CMD"))
    target = re.search(r'"(defectlens\.[\w.]+:\w+)"', cmd)
    assert target and "--factory" in cmd
    module, attr = target.group(1).split(":")
    assert callable(getattr(importlib.import_module(module), attr))


def test_every_onnx_file_in_the_serve_config_is_under_the_mounted_directory() -> None:
    cfg = yaml.safe_load((ROOT / "configs/serve/default.yaml").read_text())
    paths = [p["onnx"] for p in cfg["products"].values()]
    assert paths and all(p.startswith("models/") for p in paths)
    assert (ROOT / cfg["frozen_thresholds"]).is_file()  # the one non-weight file the image needs


# ---- compose and smoke script ----------------------------------------------------------------


def test_compose_mounts_weights_read_only_and_asks_for_the_gpu() -> None:
    svc = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["defectlens"]
    assert svc["image"] == "defectlens:local" and svc["build"] == "."
    assert "./models:/app/models:ro" in svc["volumes"]
    dev = svc["deploy"]["resources"]["reservations"]["devices"][0]
    assert dev["driver"] == "nvidia" and "gpu" in dev["capabilities"]
    assert "8000:8000" in svc["ports"]


def test_smoke_script_is_valid_bash_and_covers_what_the_phase_requires() -> None:
    path = ROOT / "scripts/docker_smoke.sh"
    done = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    text = path.read_text()
    needles = ("/health", "/inspect?product=", "steel", "visa_pcb1", "CUDAExecutionProvider",
               "torch is not installed", "no TensorRT", "--gpus=all", "reports/docker_smoke.json")
    for needle in needles:
        assert needle in text, needle


# ---- Phase 6c: the Docker job of CI and the static checks it runs -------------------------------


CI = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())


def ci_steps(job: str) -> list[dict[str, str]]:
    return CI["jobs"][job]["steps"]


def test_ci_has_a_docker_job_that_builds_without_pushing() -> None:
    assert {"test", "docker"} <= set(CI["jobs"])
    runs = " ".join(s.get("run", "") for s in ci_steps("docker"))
    uses = " ".join(s.get("uses", "") for s in ci_steps("docker"))
    assert "docker build" in runs
    assert "push" not in runs and "login" not in runs and "--push" not in runs
    assert "docker/login-action" not in uses and "build-push-action" not in uses


def test_ci_docker_job_checks_disk_before_it_builds_and_runs_the_static_checks() -> None:
    names = [s.get("name", "") for s in ci_steps("docker")]
    disk = next(i for i, n in enumerate(names) if "disk" in n.lower())
    build = next(i for i, n in enumerate(names) if n.startswith("Build"))
    assert disk < build
    text = " ".join(s.get("run", "") for s in ci_steps("docker"))
    assert "df -h" in text and "scripts/docker_static_checks.sh" in text
    assert CI["jobs"]["docker"]["runs-on"] == CI["jobs"]["test"]["runs-on"]  # pinned image


def test_static_checks_script_is_valid_bash_and_needs_no_gpu_or_weights() -> None:
    script = ROOT / "scripts" / "docker_static_checks.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    body = script.read_text()
    code = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))
    assert "--gpus" not in code and "/app/models" not in code and " -v \"" not in code
