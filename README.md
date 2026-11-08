# DefectLens

Industrial visual inspection: a supervised detector for known defects (NEU-DET) plus an anomaly
detector for unseen defects (VisA), exported to ONNX and served as an API. Work in progress; see
[PLAN.md](PLAN.md) for the phase status.

## Status

Phases 0a-0c (environment, project skeleton, risk spikes), 1a-1b (data, splits, preprocessing),
2a.1 (evaluation code, classical baseline), 2a.2 (Faster R-CNN), 2b.1 (D-FINE-S), 2b.2 (comparison), the
matched Faster R-CNN rerun, 3a-3c (PatchCore, AnomalyDINO, EfficientAD, few-shot curve) and 4a (cost-based
thresholds, one frozen evaluation, error gallery) are done. Phase 4b (leave-one-defect-type-out on VisA pcb1) is run: it weakly supports the claim that the anomaly path catches
defect types the detector never saw (clearly for `missing`, undecided for the rest, small sample), see its section. Phase 5a (ONNX export, verification against PyTorch, ORT CUDA benchmark) and Phase 6a (the FastAPI service) are done. Not started: TensorRT (5b, optional), the Gradio demo and the Docker image (6b), the final CI run and README tables (6c). Every number below
comes from an MLflow run or a script output; details per phase are in [PLAN.md](PLAN.md).

## Setup (WSL2 Ubuntu, NVIDIA GPU)

```bash
uv sync                              # locked env from uv.lock, CUDA 13 torch build
uv run python scripts/check_env.py   # prints torch/CUDA/ONNX Runtime provider status
uv run pytest && uv run ruff check . && uv run mypy src
uv run pytest -m slow                # the 6 tests above 5 s, skipped by default
```

The default run skips tests marked `slow` (above 5 s each) to stay under 60 s: 297 tests in 36 s here, plus 6 slow
tests in 105 s. `.github/workflows/ci.yml` installs the `cpu` group from the lockfile and runs ruff, mypy, `pytest` and
`pytest -m slow`; it has not run yet (it needs a push).

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

## Anomaly detection: PatchCore and AnomalyDINO (Phase 3a)

`uv run python scripts/train_anomaly.py model=patchcore,anomaly_dino category=cashew,candle,pcb1,macaroni2`.
Models fit on the normal VisA training images and are scored on the `threshold_calibration` half of the official
test set (`threshold_evaluation` is not read until Phase 4). Metrics are threshold-free: image AUROC, pixel AUROC,
AUPRO. Single seed, 75 (cashew) or 100 calibration images, untuned defaults, one MLflow run per model and category. (Peak RAM
is not tabulated: it was the process-wide maximum in this run, so the AnomalyDINO values are not its own; see PLAN.md.)

| category | model | image AUROC | pixel AUROC | AUPRO | fit | peak VRAM |
|---|---|---|---|---|---|---|
| cashew | PatchCore (WRN50, 224 px, coreset 1%) | 0.9464 | 0.9911 | 0.9195 | 19 s | 809 MiB |
| cashew | AnomalyDINO (DINOv2-S, 252 px) | 0.9624 | 0.9952 | 0.9100 | 2 s | 1,390 MiB |
| candle | PatchCore | 0.9736 | 0.9910 | 0.9623 | 63 s | 1,513 MiB |
| candle | AnomalyDINO | 0.9572 | 0.9841 | 0.9503 | 4 s | 2,686 MiB |
| pcb1 | PatchCore | 0.9188 | 0.9967 | 0.9119 | 65 s | 1,536 MiB |
| pcb1 | AnomalyDINO | 0.9304 | 0.9953 | 0.8323 | 16 s | 2,696 MiB |
| macaroni2 | PatchCore | 0.7208 | 0.9412 | 0.8379 | 63 s | 1,513 MiB |
| macaroni2 | AnomalyDINO | 0.7336 | 0.9354 | 0.7556 | 16 s | 2,686 MiB |
| **mean of 4** | PatchCore | 0.8899 | 0.9800 | 0.9079 | | |
| **mean of 4** | AnomalyDINO | 0.8959 | 0.9775 | 0.8620 | | |

What the numbers show (and do not): macaroni2 is the hard category for both models (image AUROC 0.72-0.73 against
0.92-0.97 elsewhere). Image AUROC and pixel AUROC are close between the models, and with 75-100 images and one
seed the differences of 0.01-0.02 are not shown to be meaningful. AUPRO is higher for PatchCore in all four
categories (mean 0.908 vs 0.862, largest gaps on pcb1 and macaroni2), so the one consistent difference is how
well the models cover whole defect regions, not how well they rank images. AnomalyDINO fits 4-16x faster here
(it builds no coreset; the cause was not profiled). Per-category heatmaps are in `outputs/anomaly/*/heatmaps.png` (git-ignored).

## Anomaly detection: EfficientAD (Phase 3b)

`uv run python scripts/train_anomaly.py model=efficientad category=cashew,candle,pcb1,macaroni2`. EfficientAD-S with
our own training loop on anomalib's network: pretrained teacher, student and autoencoder, batch size 1, Imagenette
penalty images, 70,000 steps, Adam 1e-4. The map-normalisation quantiles come from a seeded 10% hold-out of the
training normals (so it trains on 90% of the images and never sees calibration data). Same protocol as above: single
seed, untuned, scored on `threshold_calibration`.

| category | image AUROC | pixel AUROC | AUPRO | training | peak VRAM | peak RAM |
|---|---|---|---|---|---|---|
| cashew | 0.9304 | 0.9625 | 0.7065 | 4,207 s | 1,202 MiB | 2,368 MiB |
| candle | 0.9108 | 0.9161 | 0.8714 | 4,190 s | 1,202 MiB | 2,707 MiB |
| pcb1 | 0.9716 | 0.9919 | 0.9506 | 4,188 s | 1,202 MiB | 2,711 MiB |
| macaroni2 | 0.8268 | 0.9905 | 0.9191 | 4,186 s | 1,202 MiB | 2,707 MiB |
| **mean of 4** | 0.9099 | 0.9652 | 0.8619 | | | |

Each category ran in its own process, so the RAM column is per run. A 500-step smoke run on cashew
(`efficientad_cashew_smoke500`, image AUROC 0.8840) only checked the pipeline and is not a result.

## Three-model comparison (Phase 3c)

Generated from MLflow by `uv run python scripts/compare_anomaly.py` into
[reports/anomaly_comparison.md](reports/anomaly_comparison.md) (all three metrics per category, training cost, run
ids). Means over the four categories, `threshold_calibration`, one seed:

| model | image AUROC | pixel AUROC | AUPRO | training time per category | peak VRAM |
|---|---|---|---|---|---|
| PatchCore | 0.8899 | 0.9800 | 0.9079 | 19-65 s | 1,536 MiB |
| AnomalyDINO | 0.8959 | 0.9775 | 0.8620 | 2-16 s | 2,696 MiB |
| EfficientAD | 0.9099 | 0.9652 | 0.8619 | about 70 min | 1,202 MiB |

No model wins everywhere. EfficientAD has the highest image AUROC on pcb1 (0.9716) and on macaroni2 (0.8268 against
0.7208-0.7336 for the others) and the lowest on cashew and candle; its pixel AUROC is the lowest on cashew and
candle, and its AUPRO is the highest on pcb1 and macaroni2 (pixel AUROC too on macaroni2). PatchCore has the best pixel AUROC and AUPRO on average. With
75-100 calibration images, one seed and no confidence intervals, only the large gaps (macaroni2, EfficientAD's
cashew AUPRO of 0.7065) are worth attention; the means differ by 0.01-0.02 in image AUROC, which is not shown to be
meaningful. The "training time" column compares different things (a coreset, an embedding pass, 70,000 gradient
steps) and is not inference cost; latency is measured in Phase 5.

## Few-shot AnomalyDINO (Phase 3c)

`uv run python scripts/few_shot.py shots=1,5,10 draws=5` fits AnomalyDINO on 1, 5 and 10 normal images (5 saved,
nested draws per category: the 1-shot image is part of the 5-shot set, which is part of the 10-shot set), 60 runs in
total, scored on `threshold_calibration`. Image AUROC, mean ± sample standard deviation over the 5 draws
(not a confidence interval); the mean row averages the four categories within each draw first:

| category | 1-shot | 5-shot | 10-shot | full data |
|---|---|---|---|---|
| cashew | 0.8962 ± 0.0274 | 0.9034 ± 0.0219 | 0.9112 ± 0.0161 | 0.9624 |
| candle | 0.8560 ± 0.0294 | 0.8699 ± 0.0186 | 0.8962 ± 0.0214 | 0.9572 |
| pcb1 | 0.7039 ± 0.1747 | 0.7714 ± 0.0251 | 0.7902 ± 0.0333 | 0.9304 |
| macaroni2 | 0.6241 ± 0.0920 | 0.6934 ± 0.0532 | 0.6768 ± 0.0484 | 0.7336 |
| **mean** | 0.7700 ± 0.0588 | 0.8095 ± 0.0157 | 0.8186 ± 0.0138 | 0.8959 |

![few-shot curve](reports/figures/few_shot_curve.png)

Pixel AUROC and AUPRO tables, run ids and notes: [reports/anomaly_comparison.md](reports/anomaly_comparison.md).

- Going from 1 to 5 shots helps (mean +0.040); from 5 to 10 it does not clearly (+0.009, smaller than the
  0.014-0.016 spread over draws).
- Ten shots stay below full data in every category (gaps of 0.051 to 0.140) and, in the mean, below full-data
  PatchCore (0.8899) and EfficientAD (0.9099); those two use all 450-900 normal images and only AnomalyDINO was run
  few-shot.
- The choice of the single 1-shot image matters: two of five pcb1 draws score about chance (0.5080, 0.5200), and so
  do two macaroni2 draws (0.5124, 0.5384). More images do not always help within a draw (macaroni2 draw 2: 0.7828
  at 5 shots, 0.6932 at 10). Why those images fail was not examined.
- Cost: at most 1 s to fit and 1-2 s to score per run, peak VRAM 106 / 122 / 155 MiB for 1 / 5 / 10 shots; the whole
  job took about 4 minutes (inferred from file timestamps, not timed).

**EfficientAD training loss.** A rise in `loss_ae` on macaroni2 (from about 0.08 to 0.22) was reported from the log;
the logged curves of all four categories in MLflow show no such rise: `loss_ae` falls in every category (macaroni2:
0.597 over the first 5,000 steps to 0.188 over the last 5,000, no 5,000-step window higher than the one before),
and the quoted values match different categories (about 0.08 is cashew or candle at step 8,000; macaroni2 is 0.43
at that step). macaroni2 and pcb1 plateau at a much higher autoencoder loss than cashew and candle (final 0.17-0.18
against 0.03-0.04) and all four drop at the learning-rate step; there is no sign of a bug, though a falling loss
does not prove the implementation right.

## Cost-based thresholds, one frozen evaluation, error gallery (Phase 4a)

```bash
uv run python scripts/select_thresholds.py cost.fn_fp_ratio=5,10,50   # reads threshold_calibration / val only
uv run python scripts/evaluate_thresholds.py                          # frozen file only; run once
uv run python scripts/error_gallery.py top_k=20
```

Cost = `fn_fp_ratio * FN + FP`, with ratios 5, 10 and 50 that are assumptions, not measured costs. VisA: one
image-level threshold per model and category (`score >= t`). NEU-DET: one score threshold per detector, boxes matched
at IoU 0.5. Thresholds are chosen on `threshold_calibration` (VisA) or `val` (NEU-DET), written to
`reports/frozen_thresholds.json` with a content hash, and evaluated once on `threshold_evaluation` or `test`. The code
refuses to select on evaluation data, to evaluate without the frozen file, or to evaluate changed thresholds a second
time. Every reloaded model first reproduced its saved scores exactly. Full tables with 95% intervals:
[reports/thresholds.md](reports/thresholds.md); figures and notes: [reports/error_gallery.md](reports/error_gallery.md).

Total evaluation cost over the four VisA categories, against the best constant policy (flag nothing or everything):

| model (selection cost in brackets) | ratio 5 | ratio 10 | ratio 50 |
|---|---|---|---|
| PatchCore | 153 (96) | 152 (104) | 274 (107) |
| AnomalyDINO | 105 (94) | 129 (99) | 201 (108) |
| EfficientAD | 136 (96) | 146 (96) | 226 (96) |
| best constant policy | 175 | 175 | 175 |

NEU-DET test cost (the best constant policy, flag nothing, costs 3,140 / 6,280 / 31,400):

| detector | ratio 5 | ratio 10 | ratio 50 |
|---|---|---|---|
| Faster R-CNN | 918 | 1,403 | 4,296 |
| Faster R-CNN + scale/shift | 871 | 1,359 | 3,440 |
| D-FINE-S | 1,005 | 1,578 | 4,109 |

- Thresholds chosen on 75-100 images carried over only partly: of 36 VisA thresholds 25 beat the constant policy on the
  evaluation half, 3 tie it and 8 are worse; the cost per image was higher on evaluation than on selection for 31.
  30 selections missed no anomalous image, and 13 of those missed at least one on evaluation; at ratio 50 each such miss
  costs 50, which is why all three models end above 175.
- On macaroni2 (image AUROC 0.72-0.83) no threshold helps: AnomalyDINO ties the constant policy at every ratio and
  PatchCore is worse than it.
- All nine NEU-DET thresholds beat the constant policy by a wide margin. The matched Faster R-CNN has the lowest test
  cost at every ratio and D-FINE-S at none, although D-FINE-S has the best AP@50:95; one seed, and gaps like 871 against
  918 are not shown to be meaningful.
- Error gallery: most NEU-DET misses are boxes of the right class that miss IoU 0.5 (74 of 84 for Faster R-CNN, 73 of
  79 matched, 75 of 103 D-FINE-S have a same-class kept box overlapping them), concentrated in crazing and rolled-in
  scale; the VisA errors are mostly false alarms. These are observations on a few images; nothing was tuned on them.
- Limits: the ratios are assumed; the VisA halves are 50% (cashew 67%) anomalous, not a production prevalence; NEU-DET
  `val` also chose the checkpoints, so its selection costs are optimistic, and NEU-DET `test` had been scored in
  Phase 2 at a fixed score of 0.5 (no threshold tuned on it); the evaluation split was also read, read-only, by the
  error gallery; one seed throughout.

## Unseen defect types: leave-one-type-out on VisA pcb1 (Phase 4b)

The question behind the whole project: when a defect type never appeared in the supervised detector's training data,
does the anomaly path (trained on normal images only) still catch it, and does the detector miss it? VisA pcb1 has
four defect types (melt, missing, bent, scratch; 100 anomalous images, 8 of them with several types), so the test is
run on it.

```bash
uv run python scripts/run_unseen.py category=pcb1 holdout=bent,melt,missing,scratch,none   # one process per holdout
uv run python scripts/render_unseen.py                                                     # builds the report
```

For each held-out type a Faster R-CNN is trained without it (on `threshold_calibration` anomalies with a single other
type, plus 100 normal images), and compared with the three Phase 3 anomaly models at the same false-alarm budget (5% and
10% on the calibration normals; only normal scores set a threshold). Recall is image-level on `threshold_evaluation`,
for images that contain the held-out type ("unseen") and for those that do not ("seen", which shows whether the detector
is simply weak with 23 to 47 training images). A `none` run holds nothing out. The recipe is fixed in
[configs/unseen/pcb1.yaml](configs/unseen/pcb1.yaml) before any run; the protocol and its limits are in
[reports/unseen_defects.md](reports/unseen_defects.md).

Recall on the evaluation anomalies that contain the held-out type, as images caught / images, at the 5% budget · at the
10% budget (the detector never saw the type; the anomaly models never see any anomaly):

| held-out type | unseen images | Faster R-CNN | PatchCore | AnomalyDINO | EfficientAD |
|---|---|---|---|---|---|
| melt | 28 | 15 · 18 | 17 · 19 | 16 · 19 | 24 · 26 |
| missing | 7 | 0 · 2 | 6 · 6 | 4 · 4 | 5 · 5 |
| bent | 9 | 6 · 8 | 9 · 9 | 9 · 9 | 6 · 8 |
| scratch | 13 | 10 · 12 | 9 · 11 | 8 · 9 | 13 · 13 |

On the types it was trained on, at the 5% budget, the detector catches 44/50 of the control run's anomalies against 37
(PatchCore), 33 (AnomalyDINO) and 41 (EfficientAD), so it is not simply weak; the anomaly path is a complement, not a
better detector. Image AUROC on the held-out type (both halves, 15 to 54 images) is 0.963 / 0.860 / 0.759 / 0.901 for the
detector (bent / melt / missing / scratch), 0.999 / 0.870 / 0.982 / 0.904 for PatchCore, 0.983 / 0.906 / 0.926 / 0.927 for
AnomalyDINO and 0.896 / 0.964 / 0.936 / 0.995 for EfficientAD.

**Verdict: weakly supports the claim, decides it for one type, contradicts it for none, and cannot decide it overall.**
`missing` is clear: the detector flags 0/7 [0.00, 0.35] and PatchCore 6/7 [0.49, 0.97] with the same 2/50 false alarms.
`melt` depends on the model (EfficientAD 24/28 against the detector's 15/28, PatchCore and AnomalyDINO about level), and
`bent` and `scratch` are within noise. No anomaly model wins on every type, and the detector's recall on the held-out type
is below its recall on the other types in all four runs, which fits the claim but does not prove novelty is the cause (the
types differ in difficulty). It is a small sample (7 to 28 unseen images per type, wide intervals, no AUROC intervals), one
seed, one category, a detector trained on 23 to 47 images, and operating points matched on the 50 calibration normals only
(realised false alarms on the evaluation normals: 2 to 6 of 50 at the 5% budget). `threshold_evaluation` is read a second
time here, for a different question than in Phase 4a, with the owner's approval on the condition that these five runs are
the only ones and the recipe is not changed (both held); thresholds come from calibration normals only and no 4a result
changes. Intervals, seen-type tables and notes: [reports/unseen_defects.md](reports/unseen_defects.md).

**Exploratory follow-up (outside the approved protocol).** Afterwards, at the owner's request, the detector's recall on each type was compared between the control run (type trained on) and the held-out run, on the same evaluation images, using only the per-image scores already saved (no new run). Held-out recall is lower for all four types (5% budget: bent 6/9 against 9/9, melt 15/28 against 24/28, missing 0/7 against 5/7, scratch 10/13 against 13/13), and `missing` is not simply undetectable for this detector (the control run catches 5/7). It does not isolate novelty: the held-out detectors also train on fewer images (23 to 42 against 47), each is one seed, the exact p-values (0.25, 0.022, 0.062, 0.25) are uncorrected for eight comparisons, and the Phase 4b verdict stands. Method, tables and caveats: [reports/unseen_followup.md](reports/unseen_followup.md).

## ONNX export and ONNX Runtime CUDA (Phase 5a)

All five trained models export to ONNX (opset 20, batch 1): D-FINE-S, EfficientAD (four VisA categories), PatchCore,
AnomalyDINO and Faster R-CNN. The Dynamo exporter fails on Faster R-CNN (a data-dependent size in torchvision's detection
head); the TorchScript exporter works. Files go to `models/onnx/` (git-ignored); every attempt, including failures, is in
`reports/onnx_export.json`.

```bash
uv run python scripts/export_onnx.py
uv run python scripts/verify_onnx.py models=dfine_s              # accuracy against PyTorch (options in the docstring)
uv run python scripts/benchmark.py providers=cuda models=dfine_s
uv run python scripts/render_benchmark.py                        # builds reports/benchmark.md
```

**Accuracy.** With ORT's TF32 off, D-FINE-S reproduces PyTorch fp32 on the detection metrics (test AP@50 0.7308, AP@50:95
0.4447; val 0.7103 / 0.4345, identical to four decimals), though not slot for slot on every image (233 of 270 val images
are within 1e-3 on the raw outputs). Phase 0c had found that ORT's default TF32 changes D-FINE's raw outputs completely.
Measured now: it does change them (no image within 1e-3, top-100 overlap 0.71), but not the metrics: AP@50:95 differs by
+0.0001 [-0.0008, +0.0011] on test (paired bootstrap over 270 images; a difference of about 0.001 is not excluded), and
PyTorch's own default and the AMP path of Phase 2 differ from strict fp32 just as much with the same AP. The default stays
TF32 off. Faster R-CNN and the three anomaly models also match PyTorch (anomaly scores within 9.3e-5 relative, same AUROC).

**Latency** (batch 1, model only, p50, ORT CUDA with TF32 off / on): D-FINE-S 30.89 / 28.87 ms, EfficientAD 12.95 / 9.03,
PatchCore 10.44 / 9.20, AnomalyDINO 28.71 / 23.21, Faster R-CNN 77.01 / 45.94. ONNX Runtime is **not uniformly faster than
PyTorch eager** here (D-FINE-S in PyTorch strict fp32 takes 27.45 ms), so what the export gives is a portable, verified
artefact, not speed. AnomalyDINO under ORT uses about 4.9 GB of device memory (about 1.9 GB in PyTorch). One machine (RTX 4060
Laptop, WSL2), one run per row, no pre- or post-processing in the timings; protocol, resources, limits and what was not
measured: [reports/benchmark.md](reports/benchmark.md).

## API (Phase 6a)

```bash
uv run python scripts/serve.py                                   # localhost:8000, models load on first use
curl -F image=@tests/fixtures/sample.jpg "localhost:8000/inspect?product=steel"
curl -F image=@some_visa_image.JPG "localhost:8000/inspect?product=visa_pcb1&fn_fp_ratio=10"
curl localhost:8000/health ; curl localhost:8000/products
```

`POST /inspect` takes one image (JPEG, PNG or BMP) and a `product`: `steel` (the augmented Faster R-CNN trained on NEU-DET)
or `visa_cashew`, `visa_candle`, `visa_pcb1`, `visa_macaroni2` (EfficientAD-S, PatchCore for candle). A product runs exactly one
model, so steel and VisA are never fused. The answer holds `flagged`, `defects` (label, score and box in the uploaded
image's pixels), for VisA products the anomaly score and a base64 PNG heatmap, the operating point used (threshold, cost ratio, frozen-file
SHA-256), the execution provider and TF32 setting, and timings. The threshold is the frozen Phase 4a one for `fn_fp_ratio` 5, 10 (default) or 50, which are
assumed cost ratios, not measured ones. `/health` reports the provider that works (`CUDAExecutionProvider`, falling back to CPU with
the reason) and the loaded models; models are loaded lazily under a 5,000 MiB device budget. Bad images answer 400, unknown
products 404, a missing or unloadable model 503. Settings: `configs/serve/default.yaml`.

- **Why these models, and the TF32 settings** (steel off, EfficientAD on, PatchCore off, by a rule fixed before the measurement):
  [reports/serve_models.md](reports/serve_models.md). AnomalyDINO is not served (4.9 GB under ORT, no category where it leads EfficientAD by
  more than 0.05 image AUROC on the selection split).
- **The frozen thresholds were set on PyTorch scores; the API scores with ONNX Runtime.** On the selection splits, 0 of 375 VisA
  images change side of their frozen threshold (TF32 off or on), but the nearest image sits only 0.27-1.61% from its threshold while the two
  paths differ by up to 1.41%, so unseen images within about 1% of a threshold can fall on the other side. For steel, no image changes
  alarm status (none is defect-free), but 13 / 17 / 66 of 270 val images keep a different number of boxes at ratio 5 / 10 / 50, and the val cost at the
  frozen threshold rises by 2.9% / 1.4% / 2.4% (part of it is the fp16 reference path). The thresholds were kept as frozen and the
  shift is disclosed; re-selecting them on ORT scores would need a new evaluation round. Not measured: the CPU fallback path.
- **Limits:** the steel detector was trained on 200 x 200 images (other sizes get a warning); `visa_macaroni2` flags 44 of 50 normal
  calibration images at its frozen threshold, so its flags say little; heatmap regions are derived from the anomaly map and have not been evaluated as
  boxes; a live check with one sequential client gave warm server-side p50 of 75.2 ms (steel), 27.6 ms (pcb1) and 31.0 ms (candle); no load
  test, no authentication.
