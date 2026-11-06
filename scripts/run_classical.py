"""Tune the classical baseline on val, then score it on the configured splits (MLflow-logged).

Usage: uv run python scripts/run_classical.py [hydra overrides, e.g. model.splits=[val]]
Parameters are tuned on val only; test is scored once with the frozen parameters.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import mlflow

from defectlens.config import load_config
from defectlens.data.coco import read_coco
from defectlens.detection.classical import (
    Gray,
    detect,
    params_dict,
    score,
    tune,
)
from defectlens.eval.detection import GroundTruth
from defectlens.eval.report import log_metrics, metrics_table
from defectlens.seed import seed_everything
from defectlens.tracking import tracked_run


def load_split(coco_path: Path, images_dir: Path) -> tuple[list[Gray], list[GroundTruth]]:
    items, _ = read_coco(coco_path)
    imgs: list[Gray] = []
    for it in items:
        g = cv2.imread(str(images_dir / it.file_name), cv2.IMREAD_GRAYSCALE)
        if g is None:
            msg = f"cannot read {images_dir / it.file_name}"
            raise FileNotFoundError(msg)
        imgs.append(g)
    return imgs, [GroundTruth.of(it.boxes, it.labels) for it in items]


def main() -> int:
    cfg = load_config(["model=classical", *sys.argv[1:]])
    seed_everything(cfg.seed)
    processed, images_dir = Path(cfg.data.processed_dir), Path(cfg.data.raw_dir) / "IMAGES"
    val_imgs, val_gts = load_split(processed / "val.json", images_dir)

    with tracked_run(
        cfg.experiment,
        {**dict(cfg), "model": dict(cfg.model)},  # type: ignore[arg-type]
        seed=cfg.seed,
        dvc_file=Path(cfg.dvc_file),
        run_name="classical",
        tracking_uri=cfg.mlflow.tracking_uri,
        artifact_root=cfg.mlflow.artifact_root,
    ):
        mlflow.set_tag("model", "classical")
        params, val_tuned, tried = tune(val_imgs, val_gts, dict(cfg.model.grid))  # type: ignore[arg-type]
        print(f"tuned on val over {tried} configs: {params}")
        mlflow.log_params({f"best_{k}": v for k, v in params_dict(params).items()})
        mlflow.log_metric("configs_tried", tried)

        results = {"val": val_tuned}  # already computed with the frozen parameters
        for split in cfg.model.splits:
            if split == "val":
                continue
            imgs, gts = load_split(processed / f"{split}.json", images_dir)
            results[split] = score([detect(im, params) for im in imgs], gts)
        for split, m in results.items():
            log_metrics(split, m)
        table = metrics_table(results)
        print(table)
        mlflow.log_text(table + "\n", "metrics.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
