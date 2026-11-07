"""EfficientAD wrapper and runner on CPU with tiny fixtures (no teacher or Imagenette download).

The network needs 256 px inputs, so a step costs about 1.5 s on CPU: every test keeps to a few
steps and few images, and the module fixture fits once for all the assertions that can share it.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import torch

import defectlens.anomaly.efficientad as efficientad
import defectlens.anomaly.runner as runner
from defectlens.anomaly.datamodule import IMAGENET_MEAN, IMAGENET_STD
from defectlens.anomaly.efficientad import (
    EfficientAD,
    endless,
    ensure_imagenette,
    penalty_loader,
)
from defectlens.config import load_config
from defectlens.seed import seed_everything

SIZE = 256


def _normalised(n: int, seed: int = 0) -> torch.Tensor:
    """``n`` smooth random images, ImageNet-normalised like the datamodule's."""
    g = torch.Generator().manual_seed(seed)
    unit = torch.rand(n, 3, 16, 16, generator=g)
    unit = torch.nn.functional.interpolate(unit, size=(SIZE, SIZE), mode="bilinear")
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (unit - mean) / std


def _model(steps: int = 3, **kwargs: Any) -> EfficientAD:
    return EfficientAD(
        pretrained_teacher=False,
        penalty_images=[torch.rand(1, 3, SIZE, SIZE)],
        max_steps=steps,
        log_every=2,
        device="cpu",
        **kwargs,
    )


@pytest.fixture(scope="module")
def fitted() -> EfficientAD:
    seed_everything(0)
    model = _model(steps=3)
    model.fit([_normalised(3), _normalised(2, seed=1)])
    return model


def test_fit_sets_normalisation_and_records_history(fitted: EfficientAD) -> None:
    net = fitted.model
    assert net.is_set(net.mean_std) and net.is_set(net.quantiles)
    assert float(net.mean_std["std"].detach().min()) > 0
    q = {k: float(v.detach()) for k, v in net.quantiles.items()}
    assert q["qb_st"] > q["qa_st"] and q["qb_ae"] > q["qa_ae"]  # 99.5% quantile above the 90% one
    # 3 steps with log_every=2 -> a full window at step 2 and a partial one at step 3
    assert [row["step"] for row in fitted.history] == [2.0, 3.0]
    assert all(row["loss_st"] > 0 for row in fitted.history)
    # the teacher is frozen: it is not in the optimiser and receives no gradient
    assert all(p.grad is None for p in net.teacher.parameters())


def test_fit_info_counts_the_holdout(fitted: EfficientAD) -> None:
    info = fitted.fit_info()
    assert info["n_holdout"] == 1 and info["n_trained"] == 4  # 5 images, 10% -> 1 held out
    assert info["train_steps"] == 3 and info["steps_per_second"] > 0


def test_predict_shapes_and_normalisation(fitted: EfficientAD) -> None:
    x = _normalised(2, seed=5)
    scores, maps = fitted.predict(x)
    assert scores.shape == (2,) and maps.shape == (2, SIZE, SIZE)
    assert torch.allclose(scores, maps.amax(dim=(-2, -1)))  # image score = max of the map
    assert torch.isfinite(maps).all()


def test_state_dict_round_trip_gives_same_scores(fitted: EfficientAD) -> None:
    other = _model()
    other.model.load_state_dict(fitted.state_dict())
    x = _normalised(1, seed=7)
    assert torch.allclose(fitted.predict(x)[0], other.predict(x)[0])


@pytest.mark.slow  # ~6 s: two CPU trainings
def test_same_seed_same_training() -> None:
    runs = []
    for _ in range(2):
        seed_everything(3)
        model = _model(steps=2)
        model.fit([_normalised(3)])
        runs.append(model.history)
    assert runs[0] == runs[1]


def test_batch_size_other_than_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="batch size 1"):
        _model(train_batch_size=4)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"image_size": (224, 224)}, "too small"),
        ({"max_steps": 0}, "max_steps"),
        ({"holdout_fraction": 1.0}, "holdout_fraction"),
    ],
)
def test_bad_arguments_are_rejected(kwargs: dict[str, Any], match: str) -> None:
    args: dict[str, Any] = {
        "pretrained_teacher": False, "penalty_images": [torch.rand(1, 3, SIZE, SIZE)],
        "device": "cpu",
    }
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        EfficientAD(**args)


def test_needs_a_penalty_source() -> None:
    with pytest.raises(ValueError, match="penalty"):
        EfficientAD(pretrained_teacher=False, device="cpu")


def test_fit_needs_data() -> None:
    with pytest.raises(ValueError, match="no training"):
        _model().fit([])
    with pytest.raises(ValueError, match="at least 2"):
        _model().fit([_normalised(1)])


def _imagenette(root: Path, n: int = 3) -> Path:
    folder = root / "imagenette2" / "train" / "n01"
    folder.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        cv2.imwrite(str(folder / f"{i}.jpg"), rng.integers(0, 255, (90, 120, 3), dtype=np.uint8))
    return root


def test_penalty_loader_yields_unit_range_images(tmp_path: Path) -> None:
    root = _imagenette(tmp_path / "imagenette")
    stream = endless(penalty_loader(root, (SIZE, SIZE), num_workers=0))
    first = [next(stream) for _ in range(5)]  # 5 draws from 3 images: the stream restarts
    assert all(x.shape == (1, 3, SIZE, SIZE) for x in first)
    assert all(float(x.min()) >= 0.0 and float(x.max()) <= 1.0 for x in first)


def test_ensure_imagenette_is_a_noop_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _imagenette(tmp_path / "imagenette")

    def boom(*args: object) -> None:
        msg = "must not download"
        raise AssertionError(msg)

    monkeypatch.setattr(efficientad, "download_and_extract", boom)
    ensure_imagenette(root)


def test_ensure_imagenette_downloads_to_partial_then_renames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "imagenette"
    stale = tmp_path / "imagenette.partial"
    stale.mkdir()
    (stale / "imagenette2.tgz").write_bytes(b"truncated")  # an interrupted earlier download
    seen: list[Path] = []

    def fake(target: Path, info: object) -> None:
        seen.append(target)
        assert not any(target.iterdir()) if target.exists() else True  # restarted from scratch
        _imagenette(target)

    monkeypatch.setattr(efficientad, "download_and_extract", fake)
    ensure_imagenette(root)
    assert seen == [stale] and (root / "imagenette2").is_dir() and not stale.exists()


def _cfg(tiny: dict[str, Path], imagenet: Path, *extra: str) -> Any:
    return load_config(
        [
            "data=visa",
            "model=efficientad",
            f"data.raw_dir={tiny['root']}",
            f"data.csv={tiny['csv']}",
            f"data.splits_dir={tiny['splits']}",
            f"model.imagenet_dir={imagenet}",
            "model.pretrained_teacher=false",
            "model.max_steps=2",
            "model.log_every=1",
            "trainer.device=cpu",
            "trainer.num_workers=0",
            *extra,
        ]
    )


def test_build_model_caps_steps_under_fast_dev_run(
    tiny_visa: dict[str, Path], tmp_path: Path
) -> None:
    cfg = _cfg(tiny_visa, tmp_path, "trainer.fast_dev_run=true", "model.max_steps=70000")
    model = runner.build_model(cfg, "cpu", 0)
    assert isinstance(model, EfficientAD) and model.max_steps == runner.FAST_STEPS


@pytest.mark.slow  # ~8 s: full runner round trip with MLflow
def test_run_anomaly_efficientad_logs_loss_curve_and_checkpoint(
    tiny_visa: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow

    monkeypatch.chdir(tmp_path)
    imagenet = _imagenette(tmp_path / "imagenette")
    cfg = _cfg(
        tiny_visa, imagenet, "+run_suffix=_smoke", f"mlflow.tracking_uri=sqlite:///{tmp_path}/m.db",
        f"mlflow.artifact_root={tmp_path / 'art'}", f"data.dvc_file={tmp_path / 'missing.dvc'}",
    )
    out = runner.run_anomaly(cfg, "cat")
    assert 0.0 <= out["image_auroc"] <= 1.0
    assert out["n_trained"] == 11 and out["n_holdout"] == 1  # 12 normal train images
    assert out["train_steps"] == 2 and "memory_bank_rows" not in out

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    exp = mlflow.get_experiment_by_name("defectlens")
    assert exp is not None
    run = mlflow.search_runs([exp.experiment_id], output_format="list")[0]
    assert run.info.run_name == "efficientad_cat_smoke"
    assert run.data.tags["phase"] == "3b" and run.data.tags["model"] == "efficientad"
    assert {"calib/image_auroc", "calib/aupro", "loss_st", "loss_ae", "loss_stae"} <= set(
        run.data.metrics
    )
    history = mlflow.MlflowClient().get_metric_history(run.info.run_id, "loss_st")
    assert [m.step for m in history] == [1, 2]  # log_every=1, 2 steps
    assert (tmp_path / "models/anomaly/efficientad_cat_smoke/model.pt").is_file()
    assert (tmp_path / "outputs/anomaly/efficientad_cat_smoke/heatmaps.png").is_file()
