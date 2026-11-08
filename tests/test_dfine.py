from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from defectlens.config import load_config
from defectlens.detection.common import make_scheduler
from defectlens.detection.datamodule import NeuDetDataset, make_loader
from defectlens.detection.dfine import (
    build_model,
    make_optimizer,
    predict,
    to_hf_labels,
    train_one_epoch,
)
from defectlens.detection.trainer import run_training
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor

CPU = torch.device("cpu")
SIZE = 64


def _pre() -> Preprocessor:
    return Preprocessor(PreprocessConfig(size=SIZE))


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(pretrained=False, lr=1e-3, backbone_lr_mult=0.5, weight_decay=1e-4)


def test_labels_are_normalised_cxcywh_and_zero_based() -> None:
    t = {"boxes": torch.tensor([[10.0, 20.0, 30.0, 60.0], [0.0, 0.0, 0.0, 5.0]]),
         "labels": torch.tensor([3, 1])}
    (out,) = to_hf_labels([t], (80, 40))  # H=80, W=40
    assert out["class_labels"].tolist() == [2]  # the zero-width box is dropped
    torch.testing.assert_close(out["boxes"], torch.tensor([[0.5, 0.5, 0.5, 0.5]]))


def test_labels_clamp_boxes_outside_the_image() -> None:
    t = {"boxes": torch.tensor([[-10.0, 0.0, 20.0, 40.0]]), "labels": torch.tensor([1])}
    (out,) = to_hf_labels([t], (40, 40))
    torch.testing.assert_close(out["boxes"], torch.tensor([[0.25, 0.5, 0.5, 1.0]]))


def test_optimizer_groups_backbone_lr_and_no_decay_on_norms() -> None:
    model = build_model(_cfg(), num_classes=2)
    opt = make_optimizer(model, _cfg())
    lrs = {g["lr"] for g in opt.param_groups}
    assert lrs == {1e-3, 5e-4}
    assert {g["weight_decay"] for g in opt.param_groups} == {0.0, 1e-4}
    n_grouped = sum(len(g["params"]) for g in opt.param_groups)
    assert n_grouped == sum(p.requires_grad for p in model.parameters())


def test_accumulation_steps_scheduler_per_optimizer_step(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    torch.manual_seed(0)
    model = build_model(_cfg(), num_classes=2)
    ds = NeuDetDataset(processed / "train.json", raw / "IMAGES", _pre())
    loader = make_loader(ds, 2, shuffle=False, num_workers=0, seed=0)  # 4 batches
    opt = make_optimizer(model, _cfg())
    sched = make_scheduler(opt, 10, 1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    before = [p.detach().clone() for p in model.parameters()]
    loss = train_one_epoch(model, loader, opt, sched, scaler, CPU, amp=False, grad_clip=0.1,
                           max_batches=3, accum_steps=2)  # 3 batches -> 2 optimizer steps
    assert np.isfinite(loss) and loss > 0
    assert sched.last_epoch == 2
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_predict_returns_original_frame_boxes_and_one_based_labels(
    tiny_data: tuple[Path, Path],
) -> None:
    raw, processed = tiny_data
    model = build_model(_cfg(), num_classes=2)
    ds = NeuDetDataset(processed / "train.json", raw / "IMAGES", _pre())
    loader = make_loader(ds, 2, shuffle=False, num_workers=0, seed=0)
    dets = predict(model, loader, CPU, amp=False, max_batches=1, score_thresh=0.0,
                   detections_per_img=15)
    assert len(dets) == 2
    for d in dets:
        assert d.boxes.shape == (15, 4)
        assert set(d.labels.tolist()) <= {1, 2}
        assert (d.boxes[:, [0, 2]] <= 100).all() and (d.boxes[:, [1, 3]] <= 80).all()
        assert (d.boxes >= 0).all() and (d.scores[:-1] >= d.scores[1:]).all()


def _run_args(raw: Path, processed: Path) -> list[str]:
    return [
        "model=dfine_s", "trainer.device=cpu", "trainer.num_workers=0", "model.pretrained=false",
        f"model.input_size={SIZE}", "model.batch_size=2", "model.accum_steps=2",
        f"data.raw_dir={raw}", f"data.processed_dir={processed}",
    ]


def test_fast_dev_run_returns_a_valid_eval_dict(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    res = run_training(load_config([*_run_args(raw, processed), "trainer.fast_dev_run=true"]))
    for key in ("map50", "map50_95", "recall", "precision", "fp_per_image", "recall_max"):
        assert key in res["val"]
    assert "test" not in res


def test_early_stopping_ends_a_flat_run(
    tiny_data: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw, processed = tiny_data
    # data/splits/*.json is DVC-tracked and absent from a clean checkout (CI): use a stand-in
    splits = tmp_path / "splits.json"
    splits.write_text('{"meta": {"split_hash": "test"}}')
    args = [
        *_run_args(raw, processed), "model.epochs=6", "model.patience=1", "trainer.run_test=false",
        f"data.splits_file={splits}",
        "model.warmup_iters=1", f"model.checkpoint_dir={tmp_path / 'ckpt'}",
        f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}",
        f"mlflow.artifact_root={tmp_path / 'artifacts'}", "experiment=test_early_stop",
    ]
    run_training(load_config(args))
    out = capsys.readouterr().out
    assert "early stopping after epoch" in out
    assert out.count("\nepoch ") + out.startswith("epoch ") < 6
    assert (tmp_path / "ckpt" / "best.pt").exists()
