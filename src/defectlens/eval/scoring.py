"""Score a split with a trained model reloaded from its checkpoint (Phase 4a).

Every entry point takes ``purpose`` ("selection" or "evaluation") and the split to read, and
``check_split`` refuses a split that does not belong to that purpose, so a selection step cannot
load ``threshold_evaluation`` / ``test`` and an evaluation step cannot re-read the selection split.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.anomaly.base import Predictions, collect_predictions
from defectlens.anomaly.datamodule import VisaImages, load_split_paths, make_loader
from defectlens.anomaly.efficientad import EfficientAD
from defectlens.anomaly.runner import build_model
from defectlens.config import load_config
from defectlens.detection.datamodule import NeuDetDataset
from defectlens.detection.trainer import BACKENDS, evaluate_split
from defectlens.eval.detection import Detections, GroundTruth
from defectlens.eval.threshold_runs import (
    DetectorOutput,
    DetectorProvider,
    VisaProvider,
    VisaScores,
    assert_reproduces,
)
from defectlens.eval.thresholds import ProtocolError, check_split
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor

ANOMALY_CHECKPOINTS = Path("models/anomaly")
DETECTOR_CHECKPOINTS = {
    "faster_rcnn": Path("models/faster_rcnn/best.pt"),
    "faster_rcnn_aug": Path("models/faster_rcnn_aug/best.pt"),
    "dfine_s": Path("models/dfine_s/best.pt"),
}


def restore_checkpoint(model: Any, ckpt_dir: Path) -> None:
    """Load what ``run_anomaly`` saved: ``model.pt`` (EfficientAD) or ``memory_bank.pt``."""
    if isinstance(model, EfficientAD):
        state = torch.load(ckpt_dir / "model.pt", map_location=model.device, weights_only=True)
        model.model.load_state_dict(state)
        model.model.eval()
        return
    bank = torch.load(ckpt_dir / "memory_bank.pt", map_location=model.device, weights_only=True)
    model.model.memory_bank = bank.to(model.device)  # a registered buffer in both models


def load_anomaly_model(model_name: str, category: str, device: str) -> tuple[Any, DictConfig]:
    """The fitted model of ``(model_name, category)`` from ``models/anomaly/<model>_<category>``."""
    overrides = ["data=visa", f"model={model_name}"]
    if model_name == "efficientad":
        overrides.append("model.pretrained_teacher=false")  # the state dict carries the teacher
    cfg = load_config(overrides)
    model = build_model(cfg, device, int(cfg.seed))
    restore_checkpoint(model, ANOMALY_CHECKPOINTS / f"{model_name}_{category}")
    return model, cfg


def score_visa(
    model_name: str, category: str, split: str, purpose: str, device: str = "cuda"
) -> Predictions:
    """Image scores, maps and labels of one VisA split (``check_split`` guards the split)."""
    check_split(purpose, "visa", split)
    model, cfg = load_anomaly_model(model_name, category, device)
    data = cfg.data
    size = (int(cfg.model.image_size[0]), int(cfg.model.image_size[1]))
    paths = load_split_paths(Path(data.splits_dir), category, split)
    ds = VisaImages(Path(data.raw_dir), Path(data.csv), category, paths, size)
    loader = make_loader(ds, int(cfg.model.eval_batch_size), int(cfg.trainer.num_workers))
    return collect_predictions(model, loader, device)


def detect_neu(
    run_name: str, split: str, purpose: str, device: torch.device
) -> tuple[list[Detections], list[GroundTruth], list[str], dict[str, float], dict[str, Any]]:
    """Detections on one NEU-DET split in the original frame (``check_split`` guards the split).

    Returns detections, ground truth, image file names, the metrics computed on this split with
    the shared evaluation code, and the checkpoint (to compare with its stored val metrics). The
    model is rebuilt exactly as trained (``pretrained`` weights come from the local cache) and then
    overwritten by the checkpoint's state.
    """
    check_split(purpose, "neu_det", split)
    ckpt = torch.load(DETECTOR_CHECKPOINTS[run_name], map_location="cpu", weights_only=False)
    cfg = OmegaConf.create(ckpt["config"])
    pre_cfg: dict[str, Any] = {
        **OmegaConf.to_container(cfg.preprocess),  # type: ignore[dict-item]
        "size": int(cfg.model.input_size),
    }
    pre = Preprocessor(PreprocessConfig.from_dict(pre_cfg))
    processed, images_dir = Path(cfg.data.processed_dir), Path(cfg.data.raw_dir) / "IMAGES"
    ds = NeuDetDataset(processed / f"{split}.json", images_dir, pre)
    model = BACKENDS[str(cfg.model.name)].build(cfg.model, len(ds.class_names)).to(device)
    model.load_state_dict(ckpt["model"])
    metrics, dets, gts = evaluate_split(model, ds, cfg, device)
    return dets, gts, [it.file_name for it in ds.items], metrics, ckpt


def providers(
    device: str, *, verify: bool
) -> tuple[VisaProvider, DetectorProvider]:
    """Score providers for ``threshold_runs``. With ``verify`` (the selection step) every reloaded
    model must first reproduce what training saved: the per-image calibration scores for VisA
    models, the stored best-epoch val metrics for detectors."""

    def visa(model: str, category: str, split: str, purpose: str) -> VisaScores:
        p = score_visa(model, category, split, purpose, device)
        if verify:
            saved = json.loads(
                (Path("outputs/anomaly") / f"{model}_{category}" / "calibration_scores.json")
                .read_text()
            )
            if saved["paths"] != p.paths:
                msg = f"{model}/{category}: image order differs from the saved calibration scores"
                raise ProtocolError(msg)
            diff = assert_reproduces(f"{model}/{category}", p.scores, saved["scores"])
            print(f"  {model}/{category}: reload reproduces saved scores (max |diff| {diff:.2g})")
        return VisaScores(p.paths, p.scores, p.labels)

    def detect(run: str, split: str, purpose: str) -> DetectorOutput:
        dets, gts, names, metrics, ckpt = detect_neu(run, split, purpose, torch.device(device))
        if verify:
            keys = ("map50", "map50_95", "recall")
            diff = assert_reproduces(
                f"{run} val metrics", [metrics[k] for k in keys], [ckpt["val"][k] for k in keys],
                rtol=1e-2, atol=2e-3,
            )
            print(f"  {run}: reload reproduces stored val metrics (max |diff| {diff:.2g})")
        return DetectorOutput(dets, gts, names, metrics)

    return visa, detect
