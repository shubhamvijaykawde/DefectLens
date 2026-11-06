# DefectLens

Industrial visual inspection: a supervised detector for known defects (NEU-DET) plus an anomaly
detector for unseen defects (VisA), exported to ONNX and served as an API. Work in progress; see
[PLAN.md](PLAN.md) for the phase status.

## Status

Phases 0a-0c (environment, project skeleton, risk spikes), 1a-1b (data, splits, preprocessing) and 2a.1
(evaluation code, classical baseline) are done; Faster R-CNN (2a.2) has its code and smoke test, and its
full training run has not been done. The only accuracy numbers so far are the classical baseline's
(below). TODO: deep-model results arrive with Phases 2b-4.

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

VisA and NEU-DET are not redistributed; they are tracked with DVC (a local remote is planned, none is
configured yet). Official sources, citations and the exact licence wording are in
[docs/DATA_TERMS.md](docs/DATA_TERMS.md); NEU-DET states no licence and the VisA sources conflict, so
those questions are open. Trained-weight terms are unresolved and are not covered by this repository's
code license.

```bash
uv run python scripts/download_neu_det.py   # gdown from the official link, sha256 checked
uv run python scripts/download_visa.py
uv run python scripts/prepare_data.py       # COCO files + fixed splits, deterministic
```

- **NEU-DET:** 1,800 images, 6 classes, stratified 70/15/15 split (1,260 / 270 / 270), seed 42. The raw
  annotations had 3 exact duplicate boxes, which are dropped (4,189 -> 4,186 boxes). It has no defect-free
  images, so image-level false alarms on good steel cannot be measured.
- **VisA (4 categories so far):** each official `1cls` test set is split once, 50/50 and stratified by label,
  into `threshold_calibration` (choose thresholds and models) and `threshold_evaluation` (one final
  evaluation with frozen thresholds). Splits live in `data/splits/`.

## Preprocessing

`defectlens.preprocess` is the single path for training and serving: ROI crop, optional CLAHE, letterbox
resize, grayscale to 3 channels. Box coordinates are tracked through one affine, which also maps
predictions back to the original image. `uv run python scripts/preview_preprocess.py` writes
`reports/figures/preprocess_grid.png`. Whether CLAHE helps detection is not yet measured (TODO).

## Detection evaluation and the classical baseline

All detectors are scored by `defectlens.eval.detection`: COCO-style AP (101-point, greedy matching,
checked against `pycocotools` in the tests), recall and precision at a score threshold, a confusion matrix
with a background row, all in the original 200x200 frame. Thresholds are tuned on val only; test is scored
once with frozen parameters.

The classical baseline (`uv run python scripts/run_classical.py`, MLflow run `eae981c9...`) subtracts a
Gaussian background, thresholds a robust z-score of the local residual energy and boxes the connected
components. It has no class output, so it is scored class-agnostically:

| split | AP@50 | AP@50:95 | recall | precision | FP / image |
|---|---|---|---|---|---|
| val (tuned on) | 0.0941 | 0.0408 | 0.2073 | 0.1004 | 4.55 |
| test | 0.1072 | 0.0490 | 0.1990 | 0.0909 | 4.63 |

Its best `bg_sigma` (64) is at the edge of the search grid and was not extended after test was scored.

Faster R-CNN: `uv run python scripts/train_detector.py model=faster_rcnn trainer.fast_dev_run=true` is the
smoke test. The full run is TODO (no Faster R-CNN results yet).
