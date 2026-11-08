"""Detector on VisA images (colour, negatives) and the 4b run end to end with fake scores."""

import csv
import json
import zlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import mlflow
import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from defectlens.anomaly.datamodule import load_split_paths
from defectlens.config import load_config
from defectlens.data.coco import write_coco
from defectlens.data.visa_to_coco import BoxConfig, build_visa_coco, read_type_labels
from defectlens.detection.datamodule import NeuDetDataset
from defectlens.detection.visa_detector import image_scores, make_dataset, train_detector
from defectlens.eval.thresholds import ProtocolError
from defectlens.eval.unseen import CALIBRATION, DETECTOR, EVALUATION
from defectlens.eval.unseen_run import result_path, run_unseen, summary

EXPERIMENT = Path(__file__).resolve().parents[1] / "configs" / "unseen" / "pcb1.yaml"
BOX = BoxConfig(merge_px=9, min_area=4, pad_px=2)
SIZE = 64


def _model_cfg(**kw: Any) -> SimpleNamespace:
    base = {
        "pretrained": False, "trainable_backbone_layers": 3, "input_size": SIZE,
        "score_thresh": 0.001, "detections_per_img": 20, "lr": 0.005, "momentum": 0.9,
        "weight_decay": 1e-4, "batch_size": 2, "warmup_iters": 2, "grad_clip": 10.0,
        "augment": SimpleNamespace(hflip=0.0, vflip=0.0, brightness_contrast=0.0,
                                   scale_translate=0.0),
    }
    return SimpleNamespace(**{**base, **kw})


def _coco(fix: dict[str, Path], tmp_path: Path, n_normals: int = 3) -> tuple[Path, list[str]]:
    types = read_type_labels(fix["root"], fix["csv"], "cat")
    pool = sorted(p for p, t in types.items() if len(t) == 1)[:4]  # a few single-type anomalies
    pool += sorted(p for p, t in types.items() if not t)[:n_normals]
    classes = sorted({t[0] for t in types.values() if len(t) == 1})
    path = tmp_path / "train.json"
    write_coco(build_visa_coco(fix["root"], fix["csv"], "cat", pool, classes, types, BOX), path)
    return path, pool


def test_dataset_reads_colour_and_gives_empty_targets_for_normals(
    tiny_typed_visa: dict[str, Path], tmp_path: Path
) -> None:
    coco, pool = _coco(tiny_typed_visa, tmp_path)
    ds = make_dataset(coco, tiny_typed_visa["root"], _model_cfg(), {"size": 640}, augment_seed=None)
    assert len(ds) == len(pool)
    for i, item in enumerate(ds.items):
        img, target, meta = ds[i]
        assert img.shape == (3, SIZE, SIZE) and 0.0 <= float(img.min()) <= float(img.max()) <= 1.0
        assert meta["size"] == (96, 128)
        if "Normal" in item.file_name:
            assert target["boxes"].shape == (0, 4) and target["labels"].shape == (0,)
        else:
            assert target["boxes"].shape[1] == 4 and len(target["boxes"]) >= 1
    # colour is kept: the bright defect is bright in all three (RGB) channels, background is not
    anomaly = next(i for i, it in enumerate(ds.items) if "Anomaly" in it.file_name)
    img, target, _ = ds[anomaly]
    x0, y0, x1, y1 = (int(v) for v in target["boxes"][0].tolist())
    assert float(img[:, y0 + 3 : y1 - 3, x0 + 3 : x1 - 3].mean()) > 0.8
    assert float(img[:, :4, :4].mean()) < 0.6  # the grey padding / dark background


def test_negatives_are_augmented_too_but_neu_det_is_unchanged(
    tiny_typed_visa: dict[str, Path], tmp_path: Path
) -> None:
    assert NeuDetDataset.augment_empty is False  # steel path keeps its exact behaviour
    coco, _ = _coco(tiny_typed_visa, tmp_path)
    cfg = _model_cfg(augment=SimpleNamespace(hflip=0.0, vflip=0.0, brightness_contrast=0.0,
                                             scale_translate=1.0))
    plain = make_dataset(coco, tiny_typed_visa["root"], cfg, {"size": 640}, augment_seed=None)
    aug = make_dataset(coco, tiny_typed_visa["root"], cfg, {"size": 640}, augment_seed=3)
    normal = next(i for i, it in enumerate(plain.items) if "Normal" in it.file_name)
    a, b = plain[normal][0], aug[normal][0]
    assert not torch.equal(a, b)  # the shift / scale reached the negative image
    assert aug[normal][1]["boxes"].shape == (0, 4)


@pytest.mark.slow
def test_train_detector_and_image_scores_cover_every_image(
    tiny_typed_visa: dict[str, Path], tmp_path: Path
) -> None:
    coco, pool = _coco(tiny_typed_visa, tmp_path)
    cfg = _model_cfg()
    train_ds = make_dataset(coco, tiny_typed_visa["root"], cfg, {"size": 640}, augment_seed=0)
    seen: list[tuple[int, float]] = []
    model, stats = train_detector(
        cfg, train_ds, device=torch.device("cpu"), amp=True, seed=0, num_workers=0, epochs=2,
        max_batches=2, on_epoch=lambda e, loss: seen.append((e, loss)),
    )
    assert stats.epochs == 2 and stats.iters_per_epoch == 2 and len(stats.loss) == 2
    assert [e for e, _ in seen] == [0, 1] and all(np.isfinite(v) for _, v in seen)
    assert stats.seconds > 0 and stats.peak_vram_alloc_mib == 0.0  # CPU
    assert len(model.roi_heads.box_predictor.cls_score.weight) == 3 + 1  # classes + background
    scores = image_scores(
        model, make_dataset(coco, tiny_typed_visa["root"], cfg, {"size": 640}, augment_seed=None),
        device=torch.device("cpu"), amp=False, batch_size=2, num_workers=0, seed=0,
    )
    assert sorted(scores) == sorted(pool)
    assert all(0.0 <= v <= 1.0 for v in scores.values())


# --------------------------------------------------------------------------------------------
# End to end, with fake anomaly scores


def _cfg(fix: dict[str, Path], tmp_path: Path, **unseen: Any) -> DictConfig:
    cfg = load_config(["data=visa", "model=faster_rcnn"])
    OmegaConf.set_struct(cfg, False)
    over = {
        "data": {"raw_dir": str(fix["root"]), "csv": str(fix["csv"]),
                 "splits_dir": str(fix["splits"])},
        "unseen": {"category": "cat", "holdout": "melt", "n_normals": 2, "epochs": 1,
                   "anomaly_models": ["a", "b"], "out_dir": str(tmp_path / "out"),
                   "work_dir": str(tmp_path / "work"), "max_fprs": [0.0, 0.5],
                   "box": {"merge_px": 9, "min_area": 4, "pad_px": 2}, **unseen},
        "model": {"pretrained": False, "input_size": SIZE, "batch_size": 2, "warmup_iters": 2},
        "trainer": {"device": "cpu", "num_workers": 0, "amp": False},
        "mlflow": {"tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
                   "artifact_root": str(tmp_path / "artifacts")},
    }
    merged = OmegaConf.merge(cfg, OmegaConf.load(EXPERIMENT), OmegaConf.create(over))
    assert isinstance(merged, DictConfig)
    return merged


def _scorer(fix: dict[str, Path]) -> Any:
    labels = {r["image"]: r["label"] for r in csv.DictReader(fix["csv"].open())}
    calls: list[tuple[str, str]] = []

    def scorer(model: str, category: str, split: str, purpose: str) -> dict[str, float]:
        calls.append((split, purpose))
        assert (split, purpose) in {(CALIBRATION, "selection"), (EVALUATION, "evaluation")}
        rng = np.random.default_rng(zlib.crc32(f"{model}{split}".encode()))
        return {p: float(rng.random() * 0.5 + (labels[p] == "anomaly"))
                for p in load_split_paths(fix["splits"], category, split)}

    scorer.calls = calls  # type: ignore[attr-defined]
    return scorer


@pytest.mark.slow
def test_run_unseen_end_to_end(tiny_typed_visa: dict[str, Path], tmp_path: Path) -> None:
    cfg = _cfg(tiny_typed_visa, tmp_path)
    scorer = _scorer(tiny_typed_visa)
    result = run_unseen(cfg, scorer, device=torch.device("cpu"), track=True)

    saved = json.loads(result_path(cfg).read_text())
    assert saved["holdout"] == "melt" and saved["category"] == "cat"
    assert saved["detector"]["classes"] == ["bent", "missing"]  # no class for the held-out type
    split = saved["split"]
    types = read_type_labels(tiny_typed_visa["root"], tiny_typed_visa["csv"], "cat")
    assert split["train_anomalies"] and all(types[p] and "melt" not in types[p]
                                            and len(types[p]) == 1
                                            for p in split["train_anomalies"])
    cal = set(load_split_paths(tiny_typed_visa["splits"], "cat", CALIBRATION))
    assert set(split["train_anomalies"]) <= cal  # only the calibration half trains the detector
    assert len(split["train_normals"]) == 2 and len(split["unseen_eval"]) == 2  # melt x2
    rows = saved["rows"]
    assert sorted({r["model"] for r in rows}) == sorted(["a", "b", DETECTOR])
    assert len(rows) == 3 * 2
    a = next(r for r in rows if r["model"] == "a" and r["max_fpr"] == 0.0)
    assert a["unseen_tp"] == a["n_unseen"] and a["cal_fp"] == 0  # fake scores separate perfectly
    assert saved["detector"]["epochs"] == 1 and saved["detector"]["peak_ram_mib"] > 0
    assert saved["type_counts"]["both"]["melt"] == 6  # calibration 3+1 (melt+missing), evaluation 2
    assert set(saved["scores"][DETECTOR]) == set(saved["scores"]["a"])
    assert scorer.calls.count((CALIBRATION, "selection")) == 2  # models a and b
    assert scorer.calls.count((EVALUATION, "evaluation")) == 2
    assert len(result["rows"]) == len(rows)

    client = mlflow.MlflowClient(tracking_uri=str(cfg.mlflow.tracking_uri))
    exp = client.get_experiment_by_name(str(cfg.experiment))
    assert exp is not None
    (run,) = client.search_runs([exp.experiment_id])
    assert run.info.run_name == "unseen_cat_melt" and saved["mlflow_run_id"] == run.info.run_id
    assert run.data.tags["phase"] == "4b" and run.data.tags["holdout"] == "melt"
    assert run.data.tags["git_commit"] and run.data.tags["dataset_version"]
    assert run.data.params["seed"] == "42" and run.data.params["classes"] == "bent,missing"
    assert {"faster_rcnn/fpr0.5/unseen_tp", "a/fpr0/cal_fp", "a/auroc_unseen", "train_seconds",
            "peak_ram_mib"} <= set(run.data.metrics)
    assert len(client.get_metric_history(run.info.run_id, "train/loss")) == 1
    assert {a.path for a in client.list_artifacts(run.info.run_id)} >= {"cat_melt.json",
                                                                          "train.json"}

    with pytest.raises(ProtocolError, match="exists"):  # no quiet re-roll of the same holdout
        run_unseen(cfg, scorer, device=torch.device("cpu"), track=False)
    forced = _cfg(tiny_typed_visa, tmp_path, force=True)
    run_unseen(forced, scorer, device=torch.device("cpu"), track=False)  # allowed on request


@pytest.mark.slow
def test_smoke_run_writes_nothing_and_control_run_works(
    tiny_typed_visa: dict[str, Path], tmp_path: Path
) -> None:
    smoke = _cfg(tiny_typed_visa, tmp_path, smoke=True, epochs=5)
    result = run_unseen(smoke, _scorer(tiny_typed_visa), device=torch.device("cpu"), track=False)
    assert not result_path(smoke).exists() and not (tmp_path / "out").exists()
    assert result["detector"]["epochs"] == 1  # smoke overrides the epoch count

    control = _cfg(tiny_typed_visa, tmp_path, holdout="none", anomaly_models=[])
    res = run_unseen(control, _scorer(tiny_typed_visa), device=torch.device("cpu"), track=False)
    assert res["detector"]["classes"] == ["bent", "melt", "missing"]
    assert {r["model"] for r in res["rows"]} == {DETECTOR}
    assert all(r["n_unseen"] == 0 for r in res["rows"])
    assert result_path(control).name == "cat_none.json"


def test_smoke_summary_shows_no_evaluation_numbers() -> None:
    row = {"model": "m", "max_fpr": 0.05, "threshold": 0.5, "cal_fp": 2, "n_cal_normals": 50,
           "unseen_tp": 7, "n_unseen": 9, "seen_tp": 3, "n_seen": 4, "eval_fp": 1,
           "n_eval_normals": 50, "auroc_unseen": 0.81, "auroc_seen": 0.92}
    full, smoke = summary([row]), summary([row], evaluation=False)
    assert "7/9" in full and "1/50" in full and "0.810" in full
    assert "2/50" in smoke and "7/9" not in smoke and "unseen" not in smoke and "0.810" not in smoke
