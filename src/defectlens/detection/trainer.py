"""Train and evaluate a detector from a composed Hydra config (Faster R-CNN for now).

Flow: fixed seed -> train/val loaders -> epochs (train, score val, keep best-val checkpoint) ->
best checkpoint scored once on test. ``trainer.fast_dev_run`` shrinks everything to a few
batches and skips MLflow and checkpoints: it is only a check that the whole path runs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.detection.datamodule import NeuDetDataset, build_augment, make_loader
from defectlens.detection.faster_rcnn import (
    build_model,
    make_optimizer,
    make_scheduler,
    predict,
)
from defectlens.detection.faster_rcnn import train_one_epoch as frcnn_epoch
from defectlens.eval.confusion import confusion_matrix
from defectlens.eval.detection import evaluate_detections
from defectlens.eval.report import confusion_table, log_metrics, metrics_table, per_class_table
from defectlens.preprocess.pipeline import PreprocessConfig, Preprocessor
from defectlens.seed import seed_everything
from defectlens.tracking import tracked_run

HEADLINE_KEYS = ("map50", "map50_95", "recall", "precision", "fp_per_image", "recall_max")


def _device(cfg: DictConfig) -> torch.device:
    want = str(cfg.trainer.device)
    if want == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(want)


def _split_hash(cfg: DictConfig) -> str:
    meta = json.loads(Path(cfg.data.splits_file).read_text())["meta"]
    return str(meta["split_hash"])


def evaluate_split(
    model: Any,
    ds: NeuDetDataset,
    cfg: DictConfig,
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[dict[str, float], list[Any], list[Any]]:
    loader = make_loader(
        ds, int(cfg.model.batch_size), shuffle=False, num_workers=int(cfg.trainer.num_workers),
        seed=int(cfg.seed),
    )
    dets = predict(model, loader, device, amp=bool(cfg.trainer.amp), max_batches=max_batches)
    gts = ds.ground_truth(len(dets))
    metrics = evaluate_detections(
        dets, gts, ds.class_names, recall_score_thr=float(cfg.model.recall_score_thr)
    )
    return metrics, dets, gts


def run_training(cfg: DictConfig) -> dict[str, dict[str, float]]:
    """Returns ``{"val": ..., "test": ...}`` metric dicts (``test`` only if it was scored)."""
    if cfg.model is None or cfg.model.name != "faster_rcnn":
        msg = f"unsupported model {None if cfg.model is None else cfg.model.name!r}"
        raise ValueError(msg)
    seed_everything(int(cfg.seed))
    device = _device(cfg)
    amp = bool(cfg.trainer.amp) and device.type == "cuda"
    smoke = bool(cfg.trainer.fast_dev_run)
    mcfg = cfg.model

    pre_cfg: dict[str, Any] = {**cfg.preprocess, "size": int(mcfg.input_size)}
    pre = Preprocessor(PreprocessConfig.from_dict(pre_cfg))
    processed, images_dir = Path(cfg.data.processed_dir), Path(cfg.data.raw_dir) / "IMAGES"
    train_ds = NeuDetDataset(
        processed / "train.json", images_dir, pre, build_augment(mcfg.augment, int(cfg.seed))
    )
    val_ds = NeuDetDataset(processed / "val.json", images_dir, pre)
    train_loader = make_loader(
        train_ds, int(mcfg.batch_size), shuffle=True, num_workers=int(cfg.trainer.num_workers),
        seed=int(cfg.seed),
    )

    limit_train = 2 if smoke else cfg.trainer.limit_train_batches
    limit_val = 2 if smoke else cfg.trainer.limit_val_batches
    epochs = 1 if smoke else int(mcfg.epochs)
    iters_per_epoch = min(len(train_loader), int(limit_train)) if limit_train else len(train_loader)

    model = build_model(mcfg, len(train_ds.class_names)).to(device)
    opt = make_optimizer(model, mcfg)
    sched = make_scheduler(opt, epochs * iters_per_epoch, int(mcfg.warmup_iters))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    print(f"device={device.type} amp={amp} train={len(train_ds)} val={len(val_ds)} "
          f"iters/epoch={iters_per_epoch} epochs={epochs}")

    def loop(log: bool) -> dict[str, dict[str, float]]:
        best, best_epoch, results = -1.0, -1, {}
        ckpt_dir = Path(mcfg.checkpoint_dir)
        if log:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        t_start = time.time()
        for epoch in range(epochs):
            t0 = time.time()
            loss = frcnn_epoch(
                model, train_loader, opt, sched, scaler, device, amp=amp,
                grad_clip=float(mcfg.grad_clip), max_batches=limit_train,
            )
            t_train = time.time() - t0
            val, _, _ = evaluate_split(model, val_ds, cfg, device, max_batches=limit_val)
            results["val"] = val
            sel = val[str(mcfg.select_metric)]
            print(f"epoch {epoch + 1}/{epochs} loss={loss:.4f} val map50={val['map50']:.4f} "
                  f"map50_95={val['map50_95']:.4f} recall={val['recall']:.4f} "
                  f"train {t_train:.0f}s total {time.time() - t0:.0f}s")
            if not log:
                continue
            mlflow.log_metric("train/loss", loss, step=epoch)
            mlflow.log_metric("lr", float(sched.get_last_lr()[0]), step=epoch)
            mlflow.log_metric("epoch_seconds", time.time() - t0, step=epoch)
            log_metrics("val", val, step=epoch)
            if sel > best:
                best, best_epoch = sel, epoch
                torch.save(
                    {"model": model.state_dict(), "epoch": epoch, "val": val,
                     "config": OmegaConf.to_container(cfg, resolve=True)},
                    ckpt_dir / "best.pt",
                )
        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**20
            reserved = torch.cuda.max_memory_reserved() / 2**20
            print(f"peak VRAM: {peak:.0f} MiB allocated, {reserved:.0f} MiB reserved")
            if log:
                mlflow.log_metrics(
                    {"peak_vram_alloc_mib": peak, "peak_vram_reserved_mib": reserved}
                )
        print(f"training wall time: {time.time() - t_start:.0f}s")
        if log:
            mlflow.log_metrics({"train_seconds": time.time() - t_start, "best_epoch": best_epoch})
            if cfg.trainer.run_test:
                ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"])
                test_ds = NeuDetDataset(processed / "test.json", images_dir, pre)
                test, dets, gts = evaluate_split(model, test_ds, cfg, device)  # the one test pass
                results["test"] = test
                results["val"] = ckpt["val"]
                log_metrics("test", test)
                names = [test_ds.class_names[i] for i in sorted(test_ds.class_names)]
                cm = confusion_matrix(
                    dets, gts, len(names), score_thr=float(mcfg.recall_score_thr)
                )
                mlflow.log_text(confusion_table(cm, names) + "\n", "test_confusion.md")
                mlflow.log_text(per_class_table(test, names) + "\n", "test_per_class.md")
                mlflow.log_text(metrics_table(results) + "\n", "metrics.md")
                print(metrics_table(results))
                print(per_class_table(test, names))
        return results

    if smoke:
        results = loop(log=False)
        missing = [k for k in HEADLINE_KEYS if k not in results["val"]]
        assert not missing, f"eval dict is missing {missing}"
        print("eval dict:", json.dumps(results["val"], indent=1, default=float))
        return results

    with tracked_run(
        str(cfg.experiment),
        OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        seed=int(cfg.seed),
        dvc_file=Path(cfg.dvc_file),
        run_name=str(mcfg.name),
        tracking_uri=str(cfg.mlflow.tracking_uri),
        artifact_root=str(cfg.mlflow.artifact_root),
    ):
        mlflow.set_tags({"model": str(mcfg.name), "split_hash": _split_hash(cfg)})
        mlflow.log_params({"epochs": epochs, "batch_size": int(mcfg.batch_size),
                           "lr": float(mcfg.lr), "input_size": int(mcfg.input_size),
                           "amp": amp, "pretrained": bool(mcfg.pretrained)})
        return loop(log=True)
