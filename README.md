# DefectLens

Industrial visual inspection: a supervised detector for known defects (NEU-DET) plus an anomaly
detector for unseen defects (VisA), exported to ONNX and served as an API. Work in progress; see
[PLAN.md](PLAN.md) for the phase status.

## Status

Phases 0a-0c (environment, project skeleton, risk spikes) are done. No model has been trained or
evaluated yet, so this README reports **no accuracy numbers**. TODO: results tables arrive with Phases 2-4.

## Setup (WSL2 Ubuntu, NVIDIA GPU)

```bash
uv sync                              # locked env from uv.lock, CUDA 13 torch build
uv run python scripts/check_env.py   # prints torch/CUDA/ONNX Runtime provider status
uv run pytest && uv run ruff check . && uv run mypy src
```

CI uses the `cpu` dependency group (`uv sync --group cpu --no-group cu130`), not yet exercised.

## Spike findings so far

Measured in Phase 0c, details in [reports/spike_notes.md](reports/spike_notes.md):

- D-FINE-S exports to ONNX and runs on ONNX Runtime CUDA with outputs matching PyTorch (max abs diff
  about 7e-5), but only with TF32 disabled in ORT. With ORT's default TF32 the outputs differ completely;
  the effect on detection accuracy is not yet measured.
- AnomalyDINO uses the encoder `vit_small_patch14_dinov2` (confirmed from installed anomalib 2.7.0 source
  and a forward pass).

## Data

VisA and NEU-DET are not redistributed; they are tracked with DVC (a local remote is planned, none is configured yet). Usage terms are
not fully resolved and will be documented in `docs/DATA_TERMS.md` (Phase 1a). Trained-weight terms are
unresolved and are not covered by this repository's code license.
