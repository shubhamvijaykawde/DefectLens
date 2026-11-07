"""PatchCore / AnomalyDINO wrappers and the runner, on tiny CPU fixtures (no weight downloads)."""

from pathlib import Path

import pytest
import torch

import defectlens.anomaly.runner as runner
from defectlens.anomaly.anomaly_dino import AnomalyDino
from defectlens.anomaly.datamodule import VisaImages, load_split_paths, make_loader
from defectlens.anomaly.patchcore import PatchCore, gather_rows, greedy_coreset
from defectlens.config import load_config


def test_greedy_coreset_covers_clusters() -> None:
    z = torch.cat([torch.zeros(50, 2), torch.full((50, 2), 10.0), torch.full((50, 2), -10.0)])
    z = z + 0.01 * torch.randn(150, 2, generator=torch.Generator().manual_seed(0))
    idx = greedy_coreset(z, 3, seed=0)
    assert len(idx) == 3
    assert sorted(int(i) // 50 for i in idx) == [0, 1, 2]  # one point from each cluster
    assert torch.equal(idx, greedy_coreset(z, 3, seed=0))  # deterministic
    assert torch.equal(greedy_coreset(z, 500, seed=0), torch.arange(150))


def test_gather_rows_matches_concatenation() -> None:
    chunks = [torch.randn(4, 3), torch.randn(1, 3), torch.randn(7, 3)]
    index = torch.tensor([0, 3, 4, 5, 11])
    assert torch.equal(gather_rows(chunks, index), torch.cat(chunks)[index])


def _images(n: int, anomalous: bool, size: int = 64) -> torch.Tensor:
    g = torch.Generator().manual_seed(1 if anomalous else 0)
    x = 0.05 * torch.randn(n, 3, size, size, generator=g)
    if anomalous:
        x[:, :, 20:44, 20:44] += 3.0
    return x


def test_patchcore_scores_anomaly_higher_and_shapes() -> None:
    model = PatchCore(
        backbone="resnet18", layers=("layer2", "layer3"), pre_trained=False, device="cpu",
        coreset_ratio=0.1, num_neighbors=1, projection_dim=16,
    )
    model.fit([_images(4, False), _images(4, False)])
    # 8 images x (64/8)^2 = 64 patches at layer2 stride 8
    assert model.model.memory_bank.shape[0] == round(0.1 * 8 * 64)
    s_norm, m_norm = model.predict(_images(2, False))
    s_anom, m_anom = model.predict(_images(2, True))
    assert s_norm.shape == (2,) and m_norm.shape == (2, 64, 64)
    assert bool((s_anom > s_norm.max()).all())
    assert m_anom[:, 20:44, 20:44].mean() > m_anom[:, :10, :10].mean()


def test_patchcore_fit_needs_data() -> None:
    model = PatchCore(backbone="resnet18", pre_trained=False, device="cpu")
    with pytest.raises(ValueError, match="no training"):
        model.fit([])


def test_anomaly_dino_wrapper_with_real_encoder() -> None:
    import huggingface_hub

    cache = Path(huggingface_hub.constants.HF_HUB_CACHE)
    if not (cache / "models--timm--vit_small_patch14_dinov2.lvd142m").exists():
        pytest.skip("DINOv2-small weights are not cached; not downloading in tests")
    model = AnomalyDino(device="cpu")
    train = _images(4, False, size=56)
    model.fit([train])
    assert model.model.memory_bank.shape == (4 * 16, 384)  # 4x4 patch grid, 384-d
    s_train, _ = model.predict(train[:1])
    s_anom, maps = model.predict(_images(1, True, size=56))
    assert maps.shape == (1, 56, 56)
    assert float(s_train[0]) < 1e-3 < float(s_anom[0])  # a training image is its own neighbour


def _tiny_cfg(tiny: dict[str, Path], model: str = "patchcore") -> object:
    return load_config(
        [
            "data=visa",
            f"model={model}",
            f"data.raw_dir={tiny['root']}",
            f"data.csv={tiny['csv']}",
            f"data.splits_dir={tiny['splits']}",
            "model.backbone=resnet18",
            "model.pre_trained=false",
            "model.image_size=[64,64]",
            "model.coreset_ratio=0.1",
            "model.projection_dim=16",
            "trainer.device=cpu",
            "trainer.num_workers=0",
            "trainer.fast_dev_run=true",
        ]
    )


def test_run_anomaly_smoke_reads_only_train_and_calibration(
    tiny_visa: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested: list[str] = []
    real = runner.load_split_paths

    def spy(splits_dir: Path, category: str, split: str) -> list[str]:
        requested.append(split)
        return real(splits_dir, category, split)

    monkeypatch.setattr(runner, "load_split_paths", spy)
    monkeypatch.chdir(tmp_path)
    out = runner.run_anomaly(_tiny_cfg(tiny_visa), "cat")  # type: ignore[arg-type]
    assert set(requested) == {"train", "threshold_calibration"}
    assert 0.0 <= out["image_auroc"] <= 1.0 and out["n_train"] == 8 and out["n_calibration"] == 8
    assert (tmp_path / "outputs/anomaly/patchcore_cat/heatmaps.png").is_file()
    assert (tmp_path / "outputs/anomaly/patchcore_cat/calibration_scores.json").is_file()


def test_run_anomaly_logs_to_mlflow(
    tiny_visa: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    monkeypatch.chdir(tmp_path)
    cfg = _tiny_cfg(tiny_visa)
    cfg.trainer.fast_dev_run = False  # type: ignore[attr-defined]
    cfg.mlflow.tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"  # type: ignore[attr-defined]
    cfg.mlflow.artifact_root = str(tmp_path / "art")  # type: ignore[attr-defined]
    cfg.data.dvc_file = str(tmp_path / "missing.dvc")  # type: ignore[attr-defined]
    runner.run_anomaly(cfg, "cat")  # type: ignore[arg-type]
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)  # type: ignore[attr-defined]
    exp = mlflow.get_experiment_by_name("defectlens")
    assert exp is not None
    run = mlflow.search_runs([exp.experiment_id], output_format="list")[0]
    assert run.info.run_name == "patchcore_cat"
    assert {"calib/image_auroc", "calib/pixel_auroc", "calib/aupro", "fit_seconds"} <= set(
        run.data.metrics
    )
    assert run.data.tags["dataset_version"] == "unversioned" and run.data.tags["git_commit"]
    assert (tmp_path / "models/anomaly/patchcore_cat/memory_bank.pt").is_file()


def test_build_model_rejects_unknown_name(tiny_visa: dict[str, Path]) -> None:
    cfg = _tiny_cfg(tiny_visa)
    cfg.model.name = "nope"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="unknown anomaly model"):
        runner.build_model(cfg, "cpu", 0)  # type: ignore[arg-type]


def test_loader_is_deterministic(tiny_visa: dict[str, Path]) -> None:
    paths = load_split_paths(tiny_visa["splits"], "cat", "threshold_calibration")
    ds = VisaImages(tiny_visa["root"], tiny_visa["csv"], "cat", paths, (32, 40))
    a = [b["path"] for b in make_loader(ds, 4, 0)]
    b = [b["path"] for b in make_loader(ds, 4, 0)]
    assert a == b


def test_child_commands_one_process_per_combination() -> None:
    jobs = runner.child_commands(["patchcore", "efficientad"], ["a", "b"], ["model.shots=5"])
    assert jobs == [
        ["model=patchcore", "category=a", "model.shots=5"],
        ["model=patchcore", "category=b", "model.shots=5"],
        ["model=efficientad", "category=a", "model.shots=5"],
        ["model=efficientad", "category=b", "model.shots=5"],
    ]
