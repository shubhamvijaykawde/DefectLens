# DefectLens

Industrial visual inspection: a supervised detector for known defects (NEU-DET) plus an anomaly
detector for unseen defects (VisA), exported to ONNX and served as an API. Work in progress; see
[PLAN.md](PLAN.md) for the phase status.

## Status

Phases 0a-0c (environment, project skeleton, risk spikes), 1a-1b (data, splits, preprocessing),
2a.1 (evaluation code, classical baseline), 2a.2 (Faster R-CNN), 2b.1 (D-FINE-S), 2b.2 (comparison) and
the matched Faster R-CNN rerun are done. Measured so far: the classical baseline, Faster R-CNN (with and
without scale/shift augmentation) and D-FINE-S (below). Phase 3a (PatchCore, AnomalyDINO) has code and tests;
the full anomaly runs are pending (TODO), so there are no anomaly results yet.

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

## Faster R-CNN (ResNet50-FPN v2)

`uv run python scripts/train_detector.py model=faster_rcnn` (AMP, batch 4, 480 px, 30 epochs, 41 min, 1.75 GiB
peak VRAM; MLflow run `377b83c4...`). The best-val checkpoint (epoch 28, chosen by val AP@50:95) is scored
once on test. Recall and precision are at score threshold 0.5.

| metric | val | test |
|---|---|---|
| AP@50 | 0.7228 | 0.7613 |
| AP@50:95 | 0.3780 | 0.4224 |
| recall | 0.7398 | 0.7771 |
| precision | 0.6868 | 0.6778 |
| FP / image | 0.8259 | 0.8593 |
| recall_max (any score) | 0.9047 | 0.9315 |

| class (test) | AP@50 | recall |
|---|---|---|
| crazing | 0.4113 | 0.5253 |
| inclusion | 0.8108 | 0.8212 |
| patches | 0.9128 | 0.8603 |
| pitted_surface | 0.8540 | 0.8281 |
| rolled-in_scale | 0.6173 | 0.6765 |
| scratches | 0.9615 | 0.9605 |

Crazing and rolled-in scale are the weak classes. The model overfits: train loss fell every epoch
(0.650 to 0.078) but val AP@50 peaked at 0.7379 in epoch 10 and then stayed at 0.70-0.73, while val recall
dropped from 0.8215 (epoch 8) to about 0.74. Val AP@50:95 still rose slowly, so the AP@50:95 criterion kept
a late checkpoint. The test numbers exceed val; with 270 images per split and a single seed, differences of
this size are not yet shown to be meaningful. D-FINE therefore uses early stopping on val plus
scale/shift augmentation (see [PLAN.md](PLAN.md), Phase 2b.1).

## D-FINE-S

`uv run python scripts/train_detector.py model=dfine_s` (AMP, batch 8 x 2, 640 px, 40 epochs, 37 min, 3.4 GiB
peak VRAM; MLflow run `2f90dfc4...`). The best-val checkpoint (epoch 40) is scored once on test. Early stopping
(patience 10) was configured but never fired: val AP@50:95 was still creeping up at epoch 40.

| metric | val | test |
|---|---|---|
| AP@50 | 0.7105 | 0.7304 |
| AP@50:95 | 0.4346 | 0.4442 |
| recall | 0.6641 | 0.6927 |
| precision | 0.7491 | 0.7411 |
| FP / image | 0.5444 | 0.5630 |
| recall_max (any score) | 0.9849 | 0.9936 |

## Detector comparison

Full tables, protocol, per-class results and confusion matrices: [reports/detection_comparison.md](reports/detection_comparison.md),
generated from MLflow by `uv run python scripts/compare_detectors.py`.

| detector (test) | AP@50 | AP@50:95 | recall@0.5 | precision@0.5 | FP/img@0.5 | recall_max |
|---|---|---|---|---|---|---|
| classical (class-agnostic) | 0.1072 | 0.0490 | 0.1990 | 0.0909 | 4.6296 | 0.1990 |
| Faster R-CNN | 0.7613 | 0.4224 | 0.7771 | 0.6778 | 0.8593 | 0.9315 |
| Faster R-CNN + scale/shift (`faster_rcnn_aug`) | 0.7921 | 0.4319 | 0.8535 | 0.5648 | 1.5296 | 0.9841 |
| D-FINE-S | 0.7304 | 0.4442 | 0.6927 | 0.7411 | 0.5630 | 0.9936 |

**Not fully like-for-like.** `faster_rcnn_aug` gives Faster R-CNN the same scale/shift augmentation as D-FINE,
but input size (480 vs 640 px), optimizer (SGD vs AdamW), batch and epoch budget (30 vs 40) still differ. One
seed each, 270 test images.

What the matched run shows: augmentation helped Faster R-CNN (test AP@50 +0.031, AP@50:95 +0.010, recall
0.777 to 0.854) at the cost of precision (0.678 to 0.565; 230 to 411 false alarms on background). D-FINE's
AP@50:95 lead shrinks from 0.022 to 0.012 on test (0.057 to 0.035 on val) but remains; Faster R-CNN's AP@50 lead
grows from 0.031 to 0.062. So neither architecture is shown to be better: D-FINE leads on the stricter
AP@50:95 and precision, Faster R-CNN on AP@50 and recall at 0.5, and a gap of 0.012 AP@50:95 on one seed is not
shown to be meaningful. D-FINE's best epoch was its last (40), so it may be under-trained. Crazing is the
weakest class for all (test recall 0.20 for D-FINE, 0.53 Faster R-CNN, 0.70 with augmentation), and no
model confuses classes much: the errors are missed defects and false alarms (see the confusion matrices in the report).

## Anomaly detection (Phase 3a, in progress)

`uv run python scripts/train_anomaly.py model=patchcore category=cashew` (also `model=anomaly_dino`). Models fit
on the normal VisA training images and are scored on the `threshold_calibration` half of the official test set
(`threshold_evaluation` is not read until Phase 4). Metrics are threshold-free: image AUROC, pixel AUROC, AUPRO.
Cashew only so far, single seed, 75 calibration images, untuned defaults (MLflow runs `patchcore_cashew`,
`anomaly_dino_cashew`); the other three categories are TODO:

| model (cashew, calibration) | image AUROC | pixel AUROC | AUPRO | fit | peak VRAM |
|---|---|---|---|---|---|
| PatchCore (WRN50, 224 px, coreset 1%) | 0.9464 | 0.9911 | 0.9195 | 19 s | 809 MiB |
| AnomalyDINO (DINOv2-S, 252 px) | 0.9624 | 0.9952 | 0.9100 | 3 s | 1,390 MiB |

With 75 images one image is about 1.3 AUROC points, so the models are not shown to differ.
