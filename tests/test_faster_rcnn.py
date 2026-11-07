from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from defectlens.config import load_config
from defectlens.detection.datamodule import NeuDetDataset, build_augment, collate, make_loader
from defectlens.detection.faster_rcnn import (
    build_model,
    make_optimizer,
    make_scheduler,
    predict,
    train_one_epoch,
)
from defectlens.detection.trainer import run_training
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor

SIZE = 64


def _pre() -> Preprocessor:
    return Preprocessor(PreprocessConfig(size=SIZE))


def _model_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        pretrained=False, trainable_backbone_layers=3, input_size=SIZE, score_thresh=0.001,
        detections_per_img=20, lr=0.005, momentum=0.9, weight_decay=1e-4,
    )


def _bright_box(img: torch.Tensor) -> list[int]:
    ys, xs = torch.where(img[0] > 0.7)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def test_dataset_item_boxes_follow_the_object(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    ds = NeuDetDataset(processed / "train.json", raw / "IMAGES", _pre())
    img, target, meta = ds[0]
    assert img.shape == (3, SIZE, SIZE) and img.min() >= 0.0 and img.max() <= 1.0
    assert target["labels"].tolist() == [1] and target["boxes"].dtype == torch.float32
    got = target["boxes"][0].tolist()
    for a, b in zip(got, _bright_box(img), strict=True):
        assert abs(a - b) <= 1.5  # within interpolation error at the box edge
    back = meta["affine"].invert(target["boxes"].numpy().astype(np.float64))
    np.testing.assert_allclose(back, [[30, 20, 70, 50]], atol=1e-4)
    assert meta["size"] == (80, 100)


def test_flip_augmentation_keeps_boxes_on_object(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    aug = build_augment(SimpleNamespace(hflip=1.0, vflip=1.0, brightness_contrast=0.0), seed=0)
    img, target, _ = NeuDetDataset(processed / "train.json", raw / "IMAGES", _pre(), aug)[0]
    for a, b in zip(target["boxes"][0].tolist(), _bright_box(img), strict=True):
        assert abs(a - b) <= 1.5


def test_loader_is_seeded_and_collates(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    ds = NeuDetDataset(processed / "train.json", raw / "IMAGES", _pre())
    first = [m["image_id"] for _, _, ms in make_loader(ds, 4, shuffle=True, num_workers=0, seed=1)
             for m in ms]
    again = [m["image_id"] for _, _, ms in make_loader(ds, 4, shuffle=True, num_workers=0, seed=1)
             for m in ms]
    assert first == again and sorted(first) == [1, 2, 3, 4, 5, 6, 7, 8]
    images, targets, metas = collate([ds[0], ds[1]])
    assert len(images) == len(targets) == len(metas) == 2


def test_scheduler_warms_up_then_decays() -> None:
    p = torch.nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=1.0)
    sched = make_scheduler(opt, total_iters=10, warmup_iters=2)  # type: ignore[arg-type]
    lrs = []
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.5) and lrs[1] == pytest.approx(1.0)
    assert lrs[-1] < 0.1 and all(a >= b for a, b in zip(lrs[1:], lrs[2:], strict=False))


def test_train_step_and_predict_in_original_frame(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    torch.manual_seed(0)
    cfg = _model_cfg()
    model = build_model(cfg, num_classes=2)
    assert model.roi_heads.box_predictor.cls_score.out_features == 3  # 2 classes + background
    ds = NeuDetDataset(processed / "train.json", raw / "IMAGES", _pre())
    loader = make_loader(ds, 2, shuffle=False, num_workers=0, seed=0)
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, 2, 1)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loss = train_one_epoch(model, loader, opt, sched, scaler, torch.device("cpu"),
                           amp=False, grad_clip=10.0, max_batches=2)
    assert np.isfinite(loss) and loss > 0
    dets = predict(model, loader, torch.device("cpu"), amp=False, max_batches=1)
    assert len(dets) == 2
    for d in dets:
        assert d.boxes.shape[1] == 4 and len(d.boxes) <= 20
        assert (d.boxes[:, [0, 2]] <= 100).all() and (d.boxes[:, [1, 3]] <= 80).all()
        assert (d.boxes >= 0).all()


@pytest.mark.slow  # ~24 s: builds the model and runs the trainer end to end on CPU
def test_fast_dev_run_returns_a_valid_eval_dict(tiny_data: tuple[Path, Path]) -> None:
    raw, processed = tiny_data
    cfg = load_config([
        "model=faster_rcnn", "trainer.fast_dev_run=true", "trainer.device=cpu",
        "trainer.num_workers=0", "model.pretrained=false", f"model.input_size={SIZE}",
        "model.batch_size=2", f"data.raw_dir={raw}", f"data.processed_dir={processed}",
    ])
    res = run_training(cfg)
    val = res["val"]
    for key in ("map50", "map50_95", "recall", "precision", "fp_per_image", "recall_max"):
        assert key in val
    assert 0.0 <= val["map50"] <= 1.0 and 0.0 <= val["recall_max"] <= 1.0
    assert "test" not in res  # the smoke run never touches the test split


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported model"):
        run_training(load_config(["model=classical"]))
