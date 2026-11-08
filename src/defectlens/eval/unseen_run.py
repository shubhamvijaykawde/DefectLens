"""Run one leave-one-defect-type-out experiment end to end (Phase 4b).

One call = one held-out type: build the detector's training set, score the anomaly models,
train the detector, score every test image with it, fix the operating points on the calibration
normals, count recall on the evaluation half, write one JSON file and log one MLflow run. The
anomaly scorer is injected, so tests need no checkpoints. The protocol itself is in
``defectlens.eval.unseen``.
"""

from __future__ import annotations

import gc
import json
import math
import resource
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.anomaly.datamodule import few_shot_subset, load_split_paths
from defectlens.data.coco import write_coco
from defectlens.data.visa_to_coco import BoxConfig, build_visa_coco, read_type_labels
from defectlens.detection.visa_detector import image_scores, make_dataset, train_detector
from defectlens.eval.threshold_runs import clean
from defectlens.eval.thresholds import ProtocolError
from defectlens.eval.unseen import (
    CALIBRATION,
    DETECTOR,
    EVALUATION,
    NO_HOLDOUT,
    choose_thresholds,
    evaluate_thresholds,
    make_type_split,
    type_counts,
)
from defectlens.seed import seed_everything
from defectlens.tracking import dataset_version, git_commit_hash, tracked_run

# (model, category, split, purpose) -> {image path: image score}
AnomalyScorer = Callable[[str, str, str, str], Mapping[str, float]]
SMOKE_BATCHES = 3


def result_path(cfg: DictConfig) -> Path:
    u = cfg.unseen
    return Path(u.out_dir) / f"{u.category}_{u.holdout}.json"


def summary(rows: list[dict[str, Any]], *, evaluation: bool = True) -> str:
    """Plain-text table of the rows (printed at the end of a run, so it can be pasted back).

    ``evaluation=False`` leaves out everything computed on ``threshold_evaluation``: a smoke run
    must not show results of the real experiment before its recipe is final.
    """
    head = f"{'model':13s} {'fpr<=':>6s} {'threshold':>10s} {'cal FP':>7s}"
    if evaluation:
        head += f" {'unseen':>8s} {'seen':>8s} {'eval FP':>8s} {'AUROC unseen/seen':>18s}"
    out = [head]
    for r in rows:
        line = (f"{r['model']:13s} {r['max_fpr']:6.2f} {r['threshold']:10.4g} "
                f"{r['cal_fp']:>3d}/{r['n_cal_normals']:<3d}")
        if evaluation:
            line += (f" {r['unseen_tp']:>4d}/{r['n_unseen']:<3d}"
                     f" {r['seen_tp']:>4d}/{r['n_seen']:<3d}"
                     f" {r['eval_fp']:>4d}/{r['n_eval_normals']:<3d}"
                     f" {r['auroc_unseen']:8.3f} / {r['auroc_seen']:.3f}")
        out.append(line)
    return "\n".join(out)


def run_unseen(
    cfg: DictConfig, anomaly_scorer: AnomalyScorer, *, device: torch.device, track: bool = True
) -> dict[str, Any]:
    """Run the experiment described by ``cfg.unseen`` and return the result dict.

    With ``unseen.smoke`` the detector trains for one epoch of ``SMOKE_BATCHES`` batches and
    nothing is written or logged: it only shows that the path runs and what it costs.
    """
    u = cfg.unseen
    smoke = bool(u.smoke)
    category, holdout_name = str(u.category), str(u.holdout)
    holdout = None if holdout_name == NO_HOLDOUT else holdout_name
    seed = int(cfg.seed)
    out_file = result_path(cfg)
    if out_file.exists() and not smoke and not bool(u.force):
        msg = (f"{out_file} exists. Re-running a held-out type with the same recipe is how a "
               "favourable seed gets picked: delete the file yourself, or pass unseen.force=true "
               "and say so in the report")
        raise ProtocolError(msg)
    seed_everything(seed)

    root, csv_path = Path(cfg.data.raw_dir), Path(cfg.data.csv)
    splits_dir = Path(cfg.data.splits_dir)
    types = read_type_labels(root, csv_path, category)
    cal = load_split_paths(splits_dir, category, CALIBRATION)
    ev = load_split_paths(splits_dir, category, EVALUATION)
    train_pool = load_split_paths(splits_dir, category, "train")
    split = make_type_split(category, cal, ev, types, holdout)

    normals = few_shot_subset(train_pool, int(u.n_normals), seed)  # same sample in every run
    train_paths = [*split.train_anomalies, *normals]
    box_cfg = BoxConfig(**OmegaConf.to_container(u.box))  # type: ignore[arg-type]
    work = Path(u.work_dir) / f"{category}_{holdout_name}"
    train_json, test_json = work / "train.json", work / "test.json"
    write_coco(build_visa_coco(root, csv_path, category, train_paths, split.classes, types,
                               box_cfg), train_json)
    write_coco(build_visa_coco(root, csv_path, category, [*cal, *ev], split.classes, None,
                               box_cfg), test_json)
    print(f"holdout={holdout_name} classes={list(split.classes)} "
          f"train anomalies={len(split.train_anomalies)} normals={len(normals)} | "
          f"eval: unseen={len(split.unseen_eval)} seen={len(split.seen_eval)} "
          f"normals={len(split.eval_normals)}")

    scores: dict[str, dict[str, float]] = {}
    for model in u.anomaly_models:
        sc = dict(anomaly_scorer(str(model), category, CALIBRATION, "selection"))
        sc.update(anomaly_scorer(str(model), category, EVALUATION, "evaluation"))
        scores[str(model)] = sc
        print(f"scored {model}: {len(sc)} images")
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    epochs = 1 if smoke else int(u.epochs)
    max_batches = SMOKE_BATCHES if smoke else None
    amp = bool(cfg.trainer.amp)
    workers = int(cfg.trainer.num_workers)
    config: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]

    with ExitStack() as stack:
        if track and not smoke:
            stack.enter_context(tracked_run(
                str(cfg.experiment), config, seed=seed, dvc_file=Path(cfg.data.dvc_file),
                run_name=f"unseen_{category}_{holdout_name}",
                tracking_uri=str(cfg.mlflow.tracking_uri),
                artifact_root=str(cfg.mlflow.artifact_root),
            ))
            mlflow.set_tags({"phase": "4b", "category": category, "holdout": holdout_name,
                             "model": DETECTOR})
            mlflow.log_params({"epochs": epochs, "n_normals": len(normals),
                               "n_train_anomalies": len(split.train_anomalies),
                               "classes": ",".join(split.classes)})

        def log_epoch(epoch: int, loss: float) -> None:
            print(f"epoch {epoch + 1}/{epochs} loss={loss:.4f}")
            if track and not smoke:
                mlflow.log_metric("train/loss", loss, step=epoch)

        train_ds = make_dataset(train_json, root, cfg.model, cfg.preprocess, augment_seed=seed)
        model_, stats = train_detector(
            cfg.model, train_ds, device=device, amp=amp, seed=seed, num_workers=workers,
            epochs=epochs, max_batches=max_batches, on_epoch=log_epoch,
        )
        test_ds = make_dataset(test_json, root, cfg.model, cfg.preprocess, augment_seed=None)
        scores[DETECTOR] = image_scores(
            model_, test_ds, device=device, amp=amp, batch_size=int(cfg.model.batch_size),
            num_workers=workers, seed=seed,
        )
        missing = (set(cal) | set(ev)) - set(scores[DETECTOR])
        if missing:
            msg = f"the detector did not score {len(missing)} test images"
            raise RuntimeError(msg)

        # Thresholds first, from calibration normals only; then the evaluation half is read.
        thresholds = choose_thresholds(split, scores, [float(a) for a in u.max_fprs])
        rows = evaluate_thresholds(split, scores, thresholds)
        peak_ram = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # MiB, this process

        result: dict[str, Any] = {
            "category": category, "holdout": holdout_name, "seed": seed,
            "git_commit": git_commit_hash(),
            "dataset_version": dataset_version(Path(cfg.data.dvc_file)),
            "mlflow_run_id": run.info.run_id if (run := mlflow.active_run()) else None,
            "config": config,
            "type_counts": {
                "calibration": type_counts(types, cal), "evaluation": type_counts(types, ev),
                "both": type_counts(types, [*cal, *ev]),
            },
            "n_cal_normals": len(split.cal_normals), "n_eval_normals": len(split.eval_normals),
            "n_train_normals_available": len(train_pool),
            "detector": {
                "classes": list(split.classes), "n_train_anomalies": len(split.train_anomalies),
                "n_train_normals": len(normals), "epochs": stats.epochs,
                "iters_per_epoch": stats.iters_per_epoch, "seconds": stats.seconds,
                "peak_vram_alloc_mib": stats.peak_vram_alloc_mib,
                "peak_vram_reserved_mib": stats.peak_vram_reserved_mib, "loss": stats.loss,
                "peak_ram_mib": peak_ram,
            },
            "split": {
                "train_anomalies": list(split.train_anomalies), "train_normals": list(normals),
                "cal_normals": list(split.cal_normals), "eval_normals": list(split.eval_normals),
                "unseen_eval": list(split.unseen_eval), "seen_eval": list(split.seen_eval),
                "unseen_all": list(split.unseen_all),
            },
            "thresholds": {m: {str(a): t for a, t in d.items()} for m, d in thresholds.items()},
            "rows": rows,
            "scores": {m: {p: sc[p] for p in sorted(set(cal) | set(ev))}
                       for m, sc in scores.items()},
        }
        print("\n" + summary(rows, evaluation=not smoke))
        print(f"\ndetector: {stats.seconds:.0f}s train, peak VRAM "
              f"{stats.peak_vram_alloc_mib:.0f} MiB allocated / "
              f"{stats.peak_vram_reserved_mib:.0f} reserved, process peak RAM {peak_ram:.0f} MiB")
        if smoke:
            print("smoke run: nothing written, nothing logged")
            return result

        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(json.dumps(clean(result), indent=1) + "\n")
        print(f"wrote {out_file}")
        if track:
            mlflow.log_metrics({
                "train_seconds": stats.seconds, "peak_vram_alloc_mib": stats.peak_vram_alloc_mib,
                "peak_vram_reserved_mib": stats.peak_vram_reserved_mib, "peak_ram_mib": peak_ram,
            })
            for r in rows:
                key = f"{r['model']}/fpr{r['max_fpr']:g}"
                for k in ("unseen_tp", "seen_tp", "eval_fp", "cal_fp", "threshold"):
                    mlflow.log_metric(f"{key}/{k}", float(r[k]))
                if math.isfinite(r["auroc_unseen"]):  # NaN for the no-holdout control
                    mlflow.log_metric(f"{r['model']}/auroc_unseen", float(r["auroc_unseen"]))
            mlflow.log_artifact(str(out_file))
            mlflow.log_artifact(str(train_json))
    return result
