"""Fit one anomaly model on one VisA category and score ``threshold_calibration``.

Only ``train`` (normal images) and ``threshold_calibration`` are read here. ``threshold_evaluation``
is reserved for one final frozen-threshold evaluation in Phase 4 and is never loaded in Phase 3.
"""

from __future__ import annotations

import json
import resource
import time
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
from defectlens.anomaly.patchcore import PatchCore
from defectlens.eval.anomaly import evaluate_anomaly, save_heatmap_grid
from defectlens.seed import seed_everything
from defectlens.tracking import tracked_run

MODELS = ("patchcore", "anomaly_dino")


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

    name = f"{cfg.model.name}_{category}" + (f"_{cfg.model.shots}shot" if cfg.model.shots else "")
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
    peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MiB on Linux

    metrics = evaluate_anomaly(pred.scores, pred.labels, pred.maps, pred.masks)
    extra = {
        "n_train": float(len(train_ds)),
        "n_calibration": float(len(calib_ds)),
        "fit_seconds": fit_s,
        "score_seconds": score_s,
        "peak_vram_alloc_mib": peak_vram,
        "peak_ram_mib": peak_ram,
    }
    bank = getattr(model, "model", None)
    bank_rows = int(bank.memory_bank.shape[0]) if bank is not None else 0
    extra["memory_bank_rows"] = float(bank_rows)

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
            mlflow.set_tags({"phase": "3a", "category": category, "model": str(cfg.model.name),
                             "eval_split": "threshold_calibration"})
            mlflow.log_metrics({f"calib/{k}": v for k, v in metrics.items()})
            mlflow.log_metrics(extra)
            mlflow.log_artifact(str(heatmap))
            mlflow.log_artifact(str(scores_json))
        if bank is not None:  # keep the memory bank for Phase 4 (git-ignored)
            ckpt = Path("models/anomaly") / name
            ckpt.mkdir(parents=True, exist_ok=True)
            torch.save(bank.memory_bank.cpu(), ckpt / "memory_bank.pt")

    print(
        f"{name}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        + f" | fit {fit_s:.0f}s score {score_s:.0f}s peak VRAM {peak_vram:.0f} MiB "
        f"peak RAM {peak_ram:.0f} MiB bank {bank_rows} rows"
    )
    return {**metrics, **extra}


def parse_list(value: str) -> list[str]:
    """``"a,b"`` -> ``["a", "b"]``."""
    return [v for v in value.split(",") if v]

