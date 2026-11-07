"""Checkpoint restore for Phase 4a: a reloaded model must score like the one that was saved."""

from pathlib import Path

import pytest
import torch

from defectlens.anomaly.efficientad import EfficientAD
from defectlens.anomaly.patchcore import PatchCore
from defectlens.eval.scoring import restore_checkpoint
from defectlens.eval.thresholds import ProtocolError, check_split


def _patchcore() -> PatchCore:
    return PatchCore(
        backbone="resnet18", layers=("layer2", "layer3"), pre_trained=False, device="cpu",
        coreset_ratio=0.1, num_neighbors=1, projection_dim=16,
    )


def test_patchcore_memory_bank_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    fitted = _patchcore()
    x = 0.1 * torch.randn(6, 3, 64, 64)
    fitted.fit([x])
    torch.save(fitted.memory_bank.cpu(), tmp_path / "memory_bank.pt")  # what run_anomaly saves

    other = _patchcore()
    weights = {k: v for k, v in fitted.model.state_dict().items() if k != "memory_bank"}
    other.model.load_state_dict(weights, strict=False)  # same backbone weights, no bank
    assert other.memory_bank.numel() == 0
    restore_checkpoint(other, tmp_path)
    assert torch.equal(other.memory_bank, fitted.memory_bank)
    probe = 0.1 * torch.randn(2, 3, 64, 64)
    s1, m1 = fitted.predict(probe)
    s2, m2 = other.predict(probe)
    assert torch.allclose(s1, s2) and torch.allclose(m1, m2)


def test_efficientad_state_dict_round_trip(tmp_path: Path) -> None:
    def build(seed: int) -> EfficientAD:
        torch.manual_seed(seed)
        return EfficientAD(
            pretrained_teacher=False, penalty_images=[torch.rand(1, 3, 256, 256)], device="cpu"
        )

    saved = build(0)
    torch.save(saved.state_dict(), tmp_path / "model.pt")  # what run_anomaly saves
    other = build(1)
    assert not torch.equal(
        next(iter(other.model.student.parameters())), next(iter(saved.model.student.parameters()))
    )
    restore_checkpoint(other, tmp_path)
    for (name, a), (_, b) in zip(
        saved.model.state_dict().items(), other.model.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    assert not other.model.training


def test_scoring_entry_points_check_the_split() -> None:
    from defectlens.eval.scoring import detect_neu, score_visa

    with pytest.raises(ProtocolError):  # raised before any model or data is touched
        score_visa("patchcore", "cashew", "threshold_evaluation", "selection")
    with pytest.raises(ProtocolError):
        detect_neu("dfine_s", "test", "selection", torch.device("cpu"))
    with pytest.raises(ProtocolError):
        score_visa("patchcore", "cashew", "threshold_calibration", "evaluation")
    check_split("evaluation", "visa", "threshold_evaluation")
