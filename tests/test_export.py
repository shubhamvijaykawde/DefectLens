"""ONNX export helpers on a tiny CPU model: export, failure records, ORT agreement, comparisons."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from defectlens.detection import dfine
from defectlens.eval.detection import Detections, GroundTruth
from defectlens.export.config import load_export_config
from defectlens.export.onnx_export import AnomalyWrapper, _reason, export_module
from defectlens.export.verify import (
    RawOutputs,
    anomaly_agreement,
    bootstrap_differences,
    compare_arrays,
    make_session,
    rank_agreement,
    raw_agreement,
    topk_overlap,
)


class Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.conv(x).relu()
        return y.mean(dim=(1, 2, 3)), y[:, 0]


class Boom(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        msg = "boom: unsupported thing"
        raise RuntimeError(msg)


@pytest.mark.parametrize("dynamo", [True, False])
def test_export_roundtrip_matches_pytorch(tmp_path: Path, dynamo: bool) -> None:
    torch.manual_seed(0)
    net = Tiny().eval()
    x = torch.randn(1, 3, 16, 16)
    out = tmp_path / "tiny.onnx"
    rec = export_module(
        net, (x,), out, input_names=["image"], output_names=["score", "map"], dynamo=dynamo
    )
    assert rec.status == "ok", rec.reason
    assert rec.opset and rec.size_mib and out.is_file()
    assert json.loads(out.with_suffix(".json").read_text())["input_shape"] == [1, 3, 16, 16]
    sess = make_session(str(out), tf32=False, device="cpu")
    got_score, got_map = sess.run(None, {"image": x.numpy()})
    ref_score, ref_map = net(x)
    assert compare_arrays(ref_score.detach().numpy(), got_score)["max_abs"] < 1e-5
    assert compare_arrays(ref_map.detach().numpy(), got_map)["max_abs"] < 1e-5


def test_failed_export_is_recorded_not_raised(tmp_path: Path) -> None:
    out = tmp_path / "boom.onnx"
    rec = export_module(Boom(), (torch.randn(1, 3, 8, 8),), out, input_names=["x"],
                        output_names=["y"])
    assert rec.status == "failed" and "boom" in rec.reason
    assert not out.exists() and rec.path is None


def test_reason_walks_the_cause_chain_and_strips_colours() -> None:
    try:
        try:
            raise ValueError("\x1b[96mroot cause\x1b[0m\nsecond line")
        except ValueError as inner:
            msg = "outer"
            raise RuntimeError(msg) from inner
    except RuntimeError as exc:
        text = _reason(exc)
    assert text == "RuntimeError: outer <- ValueError: root cause"


def test_anomaly_wrapper_undoes_the_imagenet_normalisation_for_unit_input() -> None:
    class Inner(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> SimpleNamespace:
            return SimpleNamespace(pred_score=x.mean(dim=(1, 2, 3)), anomaly_map=x[:, :1])

    x = torch.randn(1, 3, 4, 4)
    plain = AnomalyWrapper(Inner())(x)[0]
    unit = AnomalyWrapper(Inner(), unit_input=True)(x)[0]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert torch.allclose(plain, x.mean(dim=(1, 2, 3)))
    assert torch.allclose(unit, (x * std + mean).mean(dim=(1, 2, 3)), atol=1e-6)


def test_config_overrides_make_lists_where_the_file_has_lists() -> None:
    cfg = load_export_config(["models=dfine_s", "verify.dfine_splits=val", "benchmark.runs=7",
                              "providers=cuda,cpu", "benchmark.tf32=true"])
    assert list(cfg.export.models) == ["dfine_s"]
    assert list(cfg.verify.dfine_splits) == ["val"]
    assert cfg.benchmark.runs == 7 and cfg.benchmark.providers == ["cuda", "cpu"]
    assert list(cfg.benchmark.tf32) == [True]
    with pytest.raises(ValueError, match="key=value"):
        load_export_config(["oops"])


# ---- comparison helpers ----------------------------------------------------------------------


def test_compare_arrays_and_shape_check() -> None:
    d = compare_arrays([1.0, 2.0, 4.0], [1.0, 2.5, 4.0])
    assert d["max_abs"] == pytest.approx(0.5) and d["mean_abs"] == pytest.approx(0.5 / 3)
    assert d["max_rel"] == pytest.approx(0.25)
    with pytest.raises(ValueError, match="shape"):
        compare_arrays(np.zeros(3), np.zeros(4))


def test_topk_overlap_is_one_for_identical_and_drops_when_candidates_change() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, 20, 5)).astype(np.float32)
    assert topk_overlap(a, a, k=10) == 1.0
    b = a.copy()
    b[:, :, :] = rng.normal(size=a.shape).astype(np.float32)
    assert topk_overlap(a, b, k=10) < 0.6


def test_rank_agreement() -> None:
    assert rank_agreement([1, 2, 3], [1, 2, 3]) == 1.0
    assert rank_agreement([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert rank_agreement([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_anomaly_agreement_reports_auroc_of_both() -> None:
    labels = [0, 0, 1, 1]
    out = anomaly_agreement([0.1, 0.2, 0.8, 0.9], [0.1, 0.25, 0.7, 0.9], labels, 0.01)
    assert out["auroc_ref"] == out["auroc_got"] == 1.0 and out["max_map_abs"] == 0.01
    assert out["max_abs"] == pytest.approx(0.1)


def _dets(n_imgs: int, score: float) -> list[Detections]:
    return [Detections.of([[10, 10, 50, 50]], [score], [1]) for _ in range(n_imgs)]


def test_bootstrap_detects_a_real_difference_and_no_difference() -> None:
    gts = [GroundTruth.of([[10, 10, 50, 50]], [1]) for _ in range(20)]
    good = _dets(20, 0.9)
    miss = [Detections.of([[100, 100, 150, 150]], [0.9], [1]) for _ in range(20)]  # wrong place
    pairs = {"same": ("good", "good2"), "worse": ("good", "miss")}
    out = bootstrap_differences(
        {"good": good, "good2": list(good), "miss": miss}, gts, {1: "a"}, pairs, ["map50"],
        n_boot=20, seed=0,
    )
    assert out["same"]["map50"] == (0.0, 0.0, 0.0)
    diff, lo, hi = out["worse"]["map50"]
    assert diff == pytest.approx(-1.0) and hi < 0


def test_dfine_decode_takes_topk_pairs_and_applies_the_threshold() -> None:
    logits = torch.full((1, 3, 2), -10.0)
    logits[0, 1, 1] = 3.0  # query 1, class index 1 -> label 2
    logits[0, 2, 0] = 0.0  # query 2, class index 0 -> label 1, score 0.5
    boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.25, 0.5, 0.5, 1.0], [0.5, 0.5, 1.0, 1.0]]])
    (xyxy, scores, labels), = dfine.decode(
        logits, boxes, (100, 200), score_thresh=0.1, detections_per_img=5
    )
    assert labels.tolist() == [2, 1]
    assert scores[0] > scores[1] > 0.1
    assert xyxy[0].tolist() == pytest.approx([0.0, 0.0, 100.0, 100.0])  # w=200, h=100


def test_raw_agreement_survives_a_slot_permutation_but_not_a_changed_score() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(2, 30, 4)).astype(np.float32)
    boxes = rng.random(size=(2, 30, 4)).astype(np.float32)
    ref = RawOutputs(logits, boxes)
    perm = rng.permutation(30)
    swapped = RawOutputs(logits[:, perm], boxes[:, perm])  # same detections in other slots
    a = raw_agreement(ref, swapped, atol=1e-3, ks=(5,))
    assert a["logits_max_abs"] > 0.1 and a["images_within_atol"] == 0
    assert a["sorted_prob_max_abs"] == pytest.approx(0.0, abs=1e-6)
    assert a["top5_overlap"] < 1.0  # slot-aware: a detection that moved to another slot is a miss
    same = raw_agreement(ref, ref, atol=1e-3, ks=(5,))
    assert same["images_within_atol"] == same["n_images"] == 2
    changed = RawOutputs(logits + 1.0, boxes)
    assert raw_agreement(ref, changed, atol=1e-3)["sorted_prob_max_abs"] > 0.05
