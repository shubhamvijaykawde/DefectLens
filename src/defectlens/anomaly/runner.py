"""Fit one anomaly model on one VisA category and score ``threshold_calibration``.

Only ``train`` (normal images) and ``threshold_calibration`` are read here. ``threshold_evaluation``
is reserved for one final frozen-threshold evaluation in Phase 4 and is never loaded in Phase 3.
"""

from __future__ import annotations

import json
import resource
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.anomaly.anomaly_dino import AnomalyDino
from defectlens.anomaly.base import AnomalyModel, collect_predictions
from defectlens.anomaly.datamodule import (
    VisaImages,
    few_shot_subset,
    load_split_paths,
    make_loader,
)
from defectlens.anomaly.efficientad import EfficientAD
from defectlens.anomaly.patchcore import PatchCore
from defectlens.eval.anomaly import evaluate_anomaly, save_heatmap_grid
from defectlens.seed import seed_everything
from defectlens.tracking import tracked_run

MODELS = ("patchcore", "anomaly_dino", "efficientad")
FAST_STEPS = 3  # training steps of a gradient-trained model under trainer.fast_dev_run
PHASES = {"patchcore": "3a", "anomaly_dino": "3a", "efficientad": "3b"}


def build_model(cfg: DictConfig, device: str, seed: int) -> AnomalyModel:
    """Model from ``cfg.model`` (a Hydra model group)."""
    m = cfg.model
    if m.name == "patchcore":
        return PatchCore(
            backbone=m.backbone,
            layers=list(m.layers),
            coreset_ratio=float(m.coreset_ratio),
            num_neighbors=int(m.num_neighbors),
            projection_dim=int(m.projection_dim),
            pre_trained=bool(m.pre_trained),
            device=device,
            seed=seed,
        )
    if m.name == "anomaly_dino":
        return AnomalyDino(
            encoder_name=m.encoder_name,
            num_neighbours=int(m.num_neighbours),
            masking=bool(m.masking),
            coreset_subsampling=bool(m.coreset_subsampling),
            device=device,
        )
    if m.name == "efficientad":
        return EfficientAD(
            model_size=str(m.model_size),
            teacher_out_channels=int(m.teacher_out_channels),
            pretrained_teacher=bool(m.pretrained_teacher),
            imagenet_dir=Path(m.imagenet_dir),
            image_size=(int(m.image_size[0]), int(m.image_size[1])),
            max_steps=FAST_STEPS if cfg.trainer.fast_dev_run else int(m.max_steps),
            lr=float(m.lr),
            weight_decay=float(m.weight_decay),
            train_batch_size=int(m.train_batch_size),
            holdout_fraction=float(m.holdout_fraction),
            log_every=int(m.log_every),
            num_workers=int(cfg.trainer.num_workers),
            device=device,
            seed=seed,
        )
    msg = f"unknown anomaly model {m.name!r}, expected one of {MODELS}"
    raise ValueError(msg)


def _calibration_subset(labels: list[int], paths: list[str], n: int) -> list[str]:
    """First ``n // 2`` anomalous and ``n // 2`` normal paths (fast_dev_run only)."""
    anomalous = [p for p, lab in zip(paths, labels, strict=True) if lab][: n // 2]
    normal = [p for p, lab in zip(paths, labels, strict=True) if not lab][: n // 2]
    return anomalous + normal


def run_anomaly(cfg: DictConfig, category: str) -> dict[str, float]:
    """Train + score one (model, category). Returns the metrics dict (also logged to MLflow)."""
    seed = int(cfg.seed)
    seed_everything(seed)
    data = cfg.data
    dev = str(cfg.trainer.device)
    fast = bool(cfg.trainer.fast_dev_run)
    root, csv_path = Path(data.raw_dir), Path(data.csv)
    size = (int(cfg.model.image_size[0]), int(cfg.model.image_size[1]))
    workers = int(cfg.trainer.num_workers)

    train_paths = few_shot_subset(
        load_split_paths(Path(data.splits_dir), category, "train"), cfg.model.shots, seed
    )
    calib_paths = load_split_paths(Path(data.splits_dir), category, "threshold_calibration")
    if fast:
        train_paths = train_paths[:8]
    train_ds = VisaImages(root, csv_path, category, train_paths, size)
    calib_ds = VisaImages(root, csv_path, category, calib_paths, size)
    if fast:
        calib_ds = VisaImages(
            root, csv_path, category, _calibration_subset(calib_ds.labels, calib_ds.paths, 8), size
        )
    bs = int(cfg.model.batch_size)
    train_loader = make_loader(train_ds, bs, workers)
    calib_loader = make_loader(calib_ds, int(cfg.model.eval_batch_size), workers)

    name = (
        f"{cfg.model.name}_{category}"
        + (f"_{cfg.model.shots}shot" if cfg.model.shots else "")
        + str(cfg.get("run_suffix", ""))  # e.g. +run_suffix=_smoke keeps a smoke run apart
    )
    out_dir = Path("outputs/anomaly") / name
    model = build_model(cfg, dev, seed)

    if dev.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model.fit(b["image"] for b in train_loader)
    fit_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    pred = collect_predictions(model, calib_loader, dev)
    score_s = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 2**20 if dev.startswith("cuda") else 0.0
    # Process-wide running maximum (MiB on Linux): per run only when the run has its own process,
    # which train_anomaly.py arranges (see child_commands).
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    metrics = evaluate_anomaly(pred.scores, pred.labels, pred.maps, pred.masks)
    extra = {
        "n_train": float(len(train_ds)),
        "n_calibration": float(len(calib_ds)),
        "fit_seconds": fit_s,
        "score_seconds": score_s,
        "peak_vram_alloc_mib": peak_vram,
        "peak_ram_mib": peak_ram,
    }
    bank = getattr(model, "memory_bank", None)  # memory-bank models only
    fit_info = getattr(model, "fit_info", None)  # gradient-trained models only
    if bank is not None:
        extra["memory_bank_rows"] = float(bank.shape[0])
    if callable(fit_info):
        extra.update(fit_info())

    heatmap = out_dir / "heatmaps.png"
    save_heatmap_grid(
        pred.images, pred.maps, pred.masks, pred.scores, pred.labels, heatmap,
        f"{name}: threshold_calibration",
    )
    scores_json = out_dir / "calibration_scores.json"
    scores_json.write_text(
        json.dumps(
            {"paths": pred.paths, "scores": pred.scores.tolist(), "labels": pred.labels.tolist()}
        )
    )

    if not fast:
        config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
        config["category"] = category
        dvc = Path(data.get("dvc_file", cfg.dvc_file))
        with tracked_run(
            cfg.experiment, config, seed=seed, dvc_file=dvc, run_name=name,
            tracking_uri=cfg.mlflow.tracking_uri, artifact_root=cfg.mlflow.artifact_root,
        ):
            mlflow.set_tags({
                "phase": PHASES[str(cfg.model.name)], "category": category,
                "model": str(cfg.model.name), "eval_split": "threshold_calibration",
            })
            mlflow.log_metrics({f"calib/{k}": v for k, v in metrics.items()})
            mlflow.log_metrics(extra)
            mlflow.log_artifact(str(heatmap))
            mlflow.log_artifact(str(scores_json))
            for row in getattr(model, "history", []):  # training loss curve, one point per window
                mlflow.log_metrics(
                    {k: v for k, v in row.items() if k != "step"}, step=int(row["step"])
                )
        ckpt = Path("models/anomaly") / name  # kept for Phase 4 / 5 (git-ignored)
        if bank is not None:
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(bank.cpu(), ckpt / "memory_bank.pt")
        elif isinstance(model, EfficientAD):
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt / "model.pt")

    print(
        f"{name}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        + f" | fit {fit_s:.0f}s score {score_s:.0f}s peak VRAM {peak_vram:.0f} MiB "
        f"peak RAM {peak_ram:.0f} MiB"
        + (f" bank {bank.shape[0]} rows" if bank is not None else "")
    )
    return {**metrics, **extra}


def child_commands(
    models: Sequence[str], categories: Sequence[str], rest: Sequence[str]
) -> list[list[str]]:
    """Argument lists (after ``train_anomaly.py``) that run each (model, category) alone.

    ``train_anomaly.py`` starts one process per combination because ``peak_ram_mib`` is the
    process-wide ``ru_maxrss``: in a shared process every run after the largest one would report
    that run's peak, and GPU memory would not be freed between runs.
    """
    return [[f"model={m}", f"category={c}", *rest] for m in models for c in categories]


def parse_list(value: str) -> list[str]:
    """``"a,b"`` -> ``["a", "b"]``."""
    return [v for v in value.split(",") if v]

