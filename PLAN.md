# DefectLens — phased plan

## Status

Plan approved 2026-11-01 (revision 2).

| Phase | Status | Measured results |
|---|---|---|
| 0a WSL2 and tooling | **Done** (confirmed by you, pasted output) | `free -h`: Mem total 11Gi, available 11Gi, swap 8.0Gi. `nproc`: 8. WSL 3.0.3.0 (unchanged), no `.wslconfig` warning. `nvidia-smi` works in Ubuntu and Docker (RTX 4060, ~1.1 GB VRAM in use at idle). Python 3.11.18, uv, gcc, git installed |
| 0b uv project and quality skeleton | **Done** (`dvc init` run by you) | `uv sync` resolved 248 packages; `check_env.py`: torch 2.14.1+cu130, torchvision 0.29.1+cu130, CUDA available (RTX 4060 Laptop), onnxscript 0.7.2, ORT 1.31.1 with CUDA provider, anomalib 2.7.0, transformers 5.22.1. `uv sync --locked` reproduces from `uv.lock`. Dummy MLflow run logged seed, git hash (`no-commit` until your first commit) and dataset version (`unversioned`). First `uv sync` took 9m23s; `.venv` is 7.0 GB. Gates at end of 0b: 7 tests pass, ruff and mypy clean |
| 0c Spike | **Done**, details in `reports/spike_notes.md` | D-FINE-S forward OK, logits (1,300,80), peak 124 MiB. **Dynamo export + ORT CUDA passes** (opset 20, max diff 7e-5 logits / 1e-5 boxes) but only with `TRANSFORMERS_DISABLE_TORCH_CHECK=1`, `optimize=False`, and ORT `use_tf32=0`; with ORT's default TF32 outputs differ completely (mAP effect unmeasured, a Phase 5a task). AnomalyDINO encoder pinned: `vit_small_patch14_dinov2`, features (B,324,384). Cashew 5-shot smoke image AUROC 0.9304, 586 MiB (smoke only). TensorRT EP listed by ORT, untested. VisA: sha256 verified, `dvc add` md5 `e01ee2ec…dir`. Gates: 14 tests in 5 s, ruff and mypy clean |
| 1a Data | **Done** | NEU-DET downloaded by `gdown` from the official Drive link (27,607,059 bytes, sha256 `cf04bb0b…ec835d5`, computed by us). 1,800 images, 4,189 raw boxes; 3 exact duplicate boxes dropped (`crazing_120`, `inclusion_62`, `patches_198`) → 4,186. XML `<filename>` lacks the extension for patches/pitted_surface, so images are found by stem. 123 images have mixed-class boxes, so splits stratify on the file-name class. Split 70/15/15 = 1,260 / 270 / 270, 210/45/45 per class, seed 42, split_hash `e3037af5…956be39`. VisA (cashew, candle, pcb1, macaroni2): train 450/900/904/900 normal; `threshold_calibration` / `threshold_evaluation` = 50A+25N each (cashew) and 50A+50N each (others). Re-running `prepare_data.py` gives byte-identical files. DVC: `data/raw/neu_det` md5 `87d1324f…`, `data/processed` `e21533db…`, `data/splits` tracked (run `dvc add` yourself again if you rerun). `docs/DATA_TERMS.md` written; VisA registry says CC BY 4.0 (conflict with earlier sources not fully resolved), NEU-DET states no license. Gates: 29 tests pass at this point |
| 1b Preprocessing | **Done** | ROI crop → CLAHE → letterbox → 3 channels, one `Preprocessor` with a single composed affine (also maps predictions back). CLAHE is off by default (effect unmeasured). `reports/figures/preprocess_grid.png` shows boxes aligned in all columns for one image per class. Gates: 40 tests in 3.7 s, ruff and mypy clean |
| 2a.1 Eval code + classical baseline | **Done** | `eval/{detection,confusion,report}.py`: COCO-style AP, agrees with `pycocotools` to 1e-9 on random data (test); a hand-computed fixture matches exactly (AP = 253/303) and a perfect prediction scores 1.0. Classical baseline (class-agnostic), tuned on val over 120 configs (best: bg_sigma 64, smooth_sigma 2, z_thresh 1.5, close_kernel 5, min_area 200), MLflow run `eae981c9b1ec4ab8b9219da640f7c77d`, 32 s on CPU. **Val: AP@50 0.0941, AP@50:95 0.0408, recall 0.2073. Test (scored once): AP@50 0.1072, AP@50:95 0.0490, recall 0.1990, precision 0.0909, 4.63 false positives/image.** The best `bg_sigma` sits on the grid edge (64); the grid was not extended after seeing test, to avoid tuning on it. The baseline is weak, as expected |
| 2a.2 Faster R-CNN | **Done** (you ran it; MLflow run `377b83c4a275444cb422dcf88986040c`) | ResNet50-FPN v2, COCO-pretrained, AMP, batch 4, 480 px, 30 epochs, 315 it/epoch. **Val (best-val checkpoint, epoch 28 chosen on val AP@50:95): AP@50 0.7228, AP@50:95 0.3780, recall 0.7398. Test (scored once): AP@50 0.7613, AP@50:95 0.4224, recall 0.7771, precision 0.6778, 0.8593 FP/image, recall_max 0.9315** (recall/precision at score 0.5). Per class on test (AP@50 / recall): crazing 0.4113 / 0.5253, inclusion 0.8108 / 0.8212, patches 0.9128 / 0.8603, pitted_surface 0.8540 / 0.8281, rolled-in_scale 0.6173 / 0.6765, scratches 0.9615 / 0.9605. Peak VRAM 1,752 MiB allocated / 2,262 MiB reserved; about 82 s/epoch, 2,452 s (41 min) total, below the 45-55 min extrapolation. **Overfitting pattern:** train loss fell every epoch (0.650 to 0.078) while val AP@50 peaked at 0.7379 in epoch 10 and then stayed between 0.70 and 0.73, and val recall fell from 0.8215 (epoch 8) to about 0.74 (0.7398 at the end). Val AP@50:95 kept creeping up to about 0.378, which is why the AP@50:95 criterion picked epoch 28. The test numbers sit above val (0.7613 vs 0.7228 AP@50); with 270 images per split that gap is within what split noise could produce, but it is not measured here (no repeat seeds yet) |
| 2b.1 D-FINE-S | **Done** (you ran it; MLflow run `2f90dfc4cce244a187414c018f62208a`) | D-FINE-S from `dfine-small-coco`, AMP, batch 8 x 2, 640 px, 40 epochs, 158 it/epoch. **Val (best-val checkpoint = epoch 40): AP@50 0.7105, AP@50:95 0.4346, recall 0.6641. Test (scored once): AP@50 0.7304, AP@50:95 0.4442, recall 0.6927, precision 0.7411, 0.5630 FP/image, recall_max 0.9936** (recall/precision at score 0.5). Per class on test (AP@50 / recall): crazing 0.3493 / 0.2020, inclusion 0.8474 / 0.8079, patches 0.9065 / 0.8603, pitted_surface 0.8236 / 0.7188, rolled-in_scale 0.4895 / 0.5490, scratches 0.9658 / 0.9737. Peak VRAM 3,394 MiB allocated / 3,620 MiB reserved; about 55 s/epoch, 2,210 s (37 min) total, matching the estimate. **Early stopping never fired:** val AP@50:95 rose slowly to 0.4346 at the last epoch (best epoch = epoch 40), so the 40-epoch budget ended the run, not patience, and more epochs might still help (not tried). The Faster R-CNN pattern (val recall falling after epoch 8) did not appear: D-FINE val recall stayed between 0.64 and 0.66 over the last 20 epochs, but this run cannot say whether the augmentation is the reason |
| 2b.2 Comparison | **Done** | `scripts/compare_detectors.py` builds `reports/detection_comparison.md` and `reports/figures/confusion_{faster_rcnn,dfine_s}.png` from MLflow (val read at each run's best epoch). Numbers equal the ones you pasted. **Not like-for-like** (augmentation, input size, optimizer, batch, epoch budget differ; one seed each); see the report. Gates: 78 tests in 42 s, ruff and mypy clean. The matched Faster R-CNN rerun was then run (row 2b.3) |
| 2b.3 Matched Faster R-CNN rerun | **Done** (you ran it; MLflow run `e73bd71a`, name `faster_rcnn_aug`) | Same recipe as `faster_rcnn` plus only `scale_translate: 0.5` (the D-FINE augmentation); 30 epochs, 2,470 s, peak 1,752 MiB allocated / 2,264 MiB reserved. **Val (best epoch 23): AP@50 0.7513, AP@50:95 0.3996, recall 0.8048. Test (scored once): AP@50 0.7921, AP@50:95 0.4319, recall 0.8535, precision 0.5648, 1.5296 FP/image, recall_max 0.9841.** Per class on test (AP@50 / recall): crazing 0.5329 / 0.6970, inclusion 0.8229 / 0.9139, patches 0.9131 / 0.8971, pitted_surface 0.8940 / 0.9062, rolled-in_scale 0.6082 / 0.7157, scratches 0.9815 / 1.0000. **What it shows (single seed, 270 images/split, remaining recipe differences kept):** (1) Augmentation helped Faster R-CNN: test AP@50 +0.031 and AP@50:95 +0.010 (val +0.029 / +0.022), recall at 0.5 up from 0.777 to 0.854, but precision down from 0.678 to 0.565 and false alarms on background up from 230 to 411 in the test confusion matrix. (2) D-FINE-S's AP@50:95 lead shrinks but does not vanish: test 0.4442 vs 0.4319 (gap 0.012, was 0.022 against the original run); val 0.4346 vs 0.3996 (gap 0.035, was 0.057). (3) Faster R-CNN's AP@50 lead grows: test 0.7921 vs 0.7304 (gap 0.062, was 0.031), val 0.7513 vs 0.7105. (4) So D-FINE is not shown to be the better architecture here: it leads only on the stricter AP@50:95 and by a margin close to what one seed could move; Faster R-CNN leads on AP@50, recall and crazing/rolled-in recall, D-FINE on precision. Caveats: D-FINE's best epoch was its last (40), so it may be under-trained; the Faster R-CNN val AP@50:95 wobbles between 0.38 and 0.40 over epochs 19-30, so the best-epoch pick (23) is noisy; SGD vs AdamW, 480 vs 640 px and batch differ. Seeds would be needed to claim more |
| 3a PatchCore and AnomalyDINO | **Done** (you ran the full command; 8 MLflow runs, phase tag `3a`, git `c63300ec`, dataset `e01ee2ec`) | Code: `anomaly/{base,datamodule,patchcore,anomaly_dino,runner}.py`, `eval/anomaly.py` (image AUROC, pixel AUROC, AUPRO; AUPRO agrees with anomalib's to 1e-3 in a test), `scripts/train_anomaly.py`. **Scored on `threshold_calibration` only (75 images for cashew, 100 for the others), single seed, untuned defaults.** Image AUROC / pixel AUROC / AUPRO, PatchCore (wide_resnet50_2, 224 px, coreset 0.01) then AnomalyDINO (DINOv2-S, 252 px, all patches): **cashew** 0.9464 / 0.9911 / 0.9195 then 0.9624 / 0.9952 / 0.9100; **candle** 0.9736 / 0.9910 / 0.9623 then 0.9572 / 0.9841 / 0.9503; **pcb1** 0.9188 / 0.9967 / 0.9119 then 0.9304 / 0.9953 / 0.8323; **macaroni2** 0.7208 / 0.9412 / 0.8379 then 0.7336 / 0.9354 / 0.7556. Unweighted mean of the 4 categories (computed from these numbers): PatchCore 0.8899 / 0.9800 / 0.9079, AnomalyDINO 0.8959 / 0.9775 / 0.8620. The pasted output equals the MLflow metrics. **Resources (measured):** fit seconds PatchCore 19 / 63 / 65 / 63, AnomalyDINO 2 / 4 / 16 / 16 (score 1 s each for PatchCore, 2 / 5 / 31 / 5 s for AnomalyDINO; the 31 s on pcb1 was not investigated); fit + score summed over the 8 runs 295 s (wall time of the command was not pasted). Peak VRAM (allocated, reset per run): PatchCore 809 MiB (450 images) and about 1.5 GiB (900 images), AnomalyDINO 1,390 MiB and about 2.7 GiB. Memory banks: PatchCore 3,528 to 7,087 rows, AnomalyDINO 145,800 to 292,896 rows. **Peak RAM correction:** `peak_ram_mib` is the process-wide `ru_maxrss`, a running maximum, and the 8 runs shared one process. So **no AnomalyDINO RAM figure was measured separately** (all four logged 6,270 MiB, which is PatchCore's peak carried over; the MLflow values stay as logged and must not be read as AnomalyDINO's). Valid per-run values: PatchCore cashew 3,763 MiB and PatchCore candle 6,228 MiB (each set a new maximum); PatchCore pcb1 6,270 MiB set a new maximum too; PatchCore macaroni2's 6,270 MiB is only an upper bound. The stand-alone AnomalyDINO-cashew smoke run measured 1,871 MiB (450 images; 900-image categories not measured). PatchCore on 900 images therefore needed about 6.2 GiB RAM (the 4.3 GB embeddings are held in CPU RAM), below the 8-10 GB the run summary had assumed, and VRAM was about 1.5 GiB, not the 5-7.5 GB assumed. **Fixed for future runs:** `scripts/train_anomaly.py` now starts one process per (model, category), so `peak_ram_mib` is per run and GPU memory is freed between runs (`runner.child_commands`, tested). **Findings:** (1) macaroni2 is the weak category for both models (image AUROC 0.72-0.73). (2) The models are close on image and pixel AUROC; with 75-100 images and one seed, gaps of 0.01-0.02 are not shown to be meaningful. (3) PatchCore has the higher AUPRO in all four categories (gaps 0.0095 to 0.0823, largest on macaroni2 and pcb1). (4) The cashew metrics of this run are identical to 4 decimals to the earlier smoke run of each model (both are in MLflow), so the pipeline is deterministic at that resolution. (5) A first AnomalyDINO smoke scored with batch 16 hit 8,982 MiB peak VRAM (the kNN similarity matrix grows with batch x patches x bank rows), so scoring uses `eval_batch_size` 2. Gates at the end of 3a: 100 tests, ruff and mypy clean |
| 3b EfficientAD | **Done** (you ran the full command overnight, one process per category; 4 MLflow runs, phase tag `3b`, git `658e52b3`) | Code: `anomaly/efficientad.py` (own training loop on anomalib's `EfficientAdModel`; teacher weights and Imagenette auto-downloaded and hash-checked), `configs/model/efficientad.yaml`, runner support (`+run_suffix`, per-process runs, loss curve and `model.pt` checkpoint), `tests/test_efficientad.py` (16 tests, CPU, no downloads). **Design choices (flag):** map-normalisation quantiles come from a seeded 10% hold-out of the *training* normals (anomalib uses its validation split, which here would be calibration data), so EfficientAD trains on 405 (cashew) or 810 / 814 / 810 images; batch size 1 enforced; inputs under 256 px raise. **Full run, 70,000 steps each, scored on `threshold_calibration` only (75 / 100 images), single seed, untuned defaults.** Image AUROC / pixel AUROC / AUPRO: **cashew** 0.9304 / 0.9625 / 0.7065; **candle** 0.9108 / 0.9161 / 0.8714; **pcb1** 0.9716 / 0.9919 / 0.9506; **macaroni2** 0.8268 / 0.9905 / 0.9191. Unweighted mean of the 4 (from `compare_anomaly.py`): 0.9099 / 0.9652 / 0.8619. The pasted output equals the MLflow metrics. **Resources (measured):** 16.5-16.8 steps/s, training 4,207 / 4,190 / 4,188 / 4,186 s (about 70 min per category, 4.65 h in total; the smoke-based extrapolation was 75 min and 5 h), peak VRAM 1,202 MiB allocated in every run, **peak RAM 2,368 / 2,707 / 2,711 / 2,707 MiB, valid per run because each category had its own process** (expected 3-3.5 GiB for 900 images, so lower), scoring 1 s each. Smoke run `efficientad_cashew_smoke500` (500 steps, image AUROC 0.8840) stays in MLflow under its own name and is not a result. **Findings (single seed, 75-100 calibration images, no confidence intervals):** (1) No model wins everywhere: EfficientAD has the highest image AUROC on pcb1 (0.9716) and macaroni2 (0.8268; the next best is 0.7336; a 0.093 margin between best and second best, the largest in the image-AUROC table) and the lowest on cashew and candle (0.9304 vs 0.9464-0.9624; 0.9108 vs 0.9572-0.9736). (2) Its pixel-level numbers are mixed: worst pixel AUROC on cashew (0.9625) and candle (0.9161) and a low AUPRO on cashew (0.7065 vs 0.9195 / 0.9100), but the best AUPRO on pcb1 and macaroni2. The cashew and candle map quality was not investigated. (3) Mean AUPRO is 0.8619 for EfficientAD and 0.8620 for AnomalyDINO, below PatchCore's 0.9079. (4) The training losses fall smoothly in all four runs (the logged curves are in MLflow); no run diverged. Gates after 3b: 117 tests, ruff and mypy clean |
| 3c Few-shot curve and comparison | **Done** (you ran the 60-run job; MLflow phase tag `3c`, `data/splits` re-added to DVC by you) | Code: `anomaly/few_shot.py` (saved nested draws, resumable runner), `scripts/few_shot.py`, `eval/anomaly_compare.py`, `scripts/compare_anomaly.py`, tests; draws in `data/splits/few_shot_draws.json` (5 per category, seed 42, nested 1 ⊂ 5 ⊂ 10, never re-drawn). **Few-shot AnomalyDINO, image AUROC on `threshold_calibration`, mean ± sample std over 5 draws (mean of the 4 categories): 1-shot 0.7700 ± 0.0588, 5-shot 0.8095 ± 0.0157, 10-shot 0.8186 ± 0.0138, full data 0.8959.** Per category (1 / 5 / 10 shots, full data): cashew 0.8962 / 0.9034 / 0.9112 (0.9624), candle 0.8560 / 0.8699 / 0.8962 (0.9572), pcb1 0.7039 ± 0.1747 / 0.7714 / 0.7902 (0.9304), macaroni2 0.6241 / 0.6934 / 0.6768 (0.7336). Mean pixel AUROC 0.9514 / 0.9661 / 0.9675 (full 0.9775), mean AUPRO 0.7767 / 0.8398 / 0.8412 (full 0.8620). **Reading:** 1 to 5 shots helps (+0.040 mean), 5 to 10 does not clearly (+0.009, below the 0.014-0.016 draw spread); 10 shots stay 0.05-0.14 below full data in every category and below full-data PatchCore (0.8899) and EfficientAD (0.9099) in the mean. One bad 1-shot image can drop a draw to chance (pcb1 draws 0.5080 and 0.5200, macaroni2 0.5124 and 0.5384), and more images can lower a draw (macaroni2 draw 2: 0.7828 at 5 shots, 0.6932 at 10). Full tables, ids and notes: `reports/anomaly_comparison.md` (notes in `reports/anomaly_comparison_notes.md`), figure `reports/figures/few_shot_curve.png`. **Cost:** per run at most 1 s fit and 1-2 s score, peak VRAM 106 / 122 / 155 MiB for 1 / 5 / 10 shots (banks 324 / 1,620 / 3,240 rows); whole job about 4 min wall (inferred from file timestamps 23:53:15 to 23:57:17, not timed); per-run RAM not logged (shared process). All runs scored on `threshold_calibration`. **Three-model comparison:** see rows 3a and 3b; mean image AUROC PatchCore 0.8899, AnomalyDINO 0.8959, EfficientAD 0.9099, mean AUPRO 0.9079 / 0.8620 / 0.8619, mean pixel AUROC 0.9800 / 0.9775 / 0.9652 |
| 3b.1 EfficientAD loss-curve check (no retraining) | **Done** | Read the logged `loss_st`, `loss_ae`, `loss_stae` of the four full runs (700 points each, 100-step means) in MLflow and compared with `~/efficientad_3b.log`. **No rise in `loss_ae` in any category, so there is no sign of a bug:** the macaroni2 curve falls from 0.597 (mean of the first 5,000 steps) to 0.188 (last 5,000); none of the 13 steps between consecutive 5,000-step windows rose for `loss_ae` in any category, and the largest rebound of a 1,000-step mean above its running minimum is 0.006 (macaroni2, step 43,000). The values quoted from the log (about 0.08 near step 8,000, about 0.22 around step 44,000, 0.17 at the end) match different categories: `loss_ae` at step 8,000 is 0.0822 (cashew) and 0.0846 (candle) but 0.2884 (pcb1) and 0.4252 (macaroni2); at step 44,000 it is 0.0472 / 0.0317 / 0.1887 / 0.2374, and the final values are 0.0386 / 0.0254 / 0.1764 / 0.1734. macaroni2 and pcb1 simply plateau at a much higher autoencoder loss than cashew and candle (final 0.17-0.18 against 0.03-0.04), and every category drops after the learning-rate step at 95% of the steps (step 66,500), as expected. Higher reconstruction loss on macaroni2 is consistent with it being the hardest category for the other two models as well, but that is not tested here; a falling loss also does not prove the implementation right (the evidence for that remains the 0.99 pixel AUROC on macaroni2 and pcb1 and the unit tests) |
| CI and slow tests (early part of 6c) | **Done, CI not yet run** | Tests above 5 s are marked `@pytest.mark.slow` (3 tests: `test_fast_dev_run_returns_a_valid_eval_dict` 24.2 s, `test_run_anomaly_efficientad_logs_loss_curve_and_checkpoint` 8.2 s, `test_same_seed_same_training` 6.3 s). `addopts` has `-m 'not slow'`, so `uv run pytest` ran 130 tests in 29 s when the marker was added (the whole suite was 133 tests in 74 s before) and 176 tests in 31 s after the Phase 4a tests were added; `uv run pytest -m slow` runs the 3 in 46 s (47 s later) with `CUDA_VISIBLE_DEVICES=` (CPU only) at the first run, `-m ""` runs everything. `.github/workflows/ci.yml` (new): `uv sync --locked --no-default-groups --group cpu --group serve --group dev` (resolves from `uv.lock`, checked with `--dry-run`), ruff, mypy, `pytest`, `pytest -m slow`. The workflow itself has not run (it needs a push); no Docker step yet (no Dockerfile) |
| 4a Cost-based thresholds, one frozen evaluation, error gallery | **Done** (🟢, run by me; MLflow `thresholds_select` `223169bc`, `thresholds_evaluate` `5ed3e5d5`, git `1b2620b6` + uncommitted changes) | **Code:** `eval/{thresholds,threshold_runs,scoring,gallery,gallery_run}.py`, `configs/cost/default.yaml` (ratios 5, 10, 50; IoU 0.5), `scripts/{select_thresholds,evaluate_thresholds,render_thresholds,error_gallery}.py`, tests `test_{thresholds,scoring,gallery}.py` (+ `detection_fixtures.py`). **Cost** = `fn_fp_ratio * FN + FP`; VisA at image level (one threshold per model and category, `score >= t`), NEU-DET at box level (one threshold per detector, class-aware greedy IoU 0.5; Faster R-CNN, matched Faster R-CNN, D-FINE-S). Candidates are midpoints between distinct scores, ties go to the median tied candidate. **Protocol in code:** `check_split` lets selection read only `threshold_calibration` / `val` and evaluation only `threshold_evaluation` / `test` (tests: selector rejects evaluation data, evaluator rejects selection data); `frozen_thresholds.json` carries a content hash that is verified before any evaluation data is read; the evaluator refuses to run without it or if it was edited, takes no options, and stops if a different frozen file was already evaluated. **Step 1** (71 s): 45 thresholds (36 VisA, 9 NEU-DET), frozen file SHA-256 `fef28e76…1b2620b6`; every reloaded model first reproduced the saved calibration scores (12 anomaly models) or the stored best-epoch val metrics (3 detectors) with max difference 0. Before step 2 I rehearsed the whole evaluator path on the *selection* splits only (providers remapped, nothing written): all 45 frozen counts were reproduced exactly. **Step 2** (67 s, run once): see `reports/thresholds.md`. VisA: of 36 thresholds 25 beat the trivial policy (flag nothing / flag everything), 3 tie, 8 are worse; cost per image was higher on evaluation than on selection for 31; 30 selections had no miss at selection and 13 of them missed at least one on evaluation. Total over four categories at ratio 10, selection to evaluation: PatchCore 104 to 152, EfficientAD 96 to 146, AnomalyDINO 99 to 129 (trivial 175); at ratio 50 all three exceed the trivial 175 (274 / 226 / 201). AnomalyDINO has the lowest total at all three ratios, not shown to be meaningful. macaroni2 is at or below the trivial policy for AnomalyDINO and PatchCore at every ratio. NEU-DET: 9 of 9 beat the trivial policy; the matched Faster R-CNN has the lowest test cost at every ratio (871 / 1,359 / 3,440), D-FINE-S none, single seed. **Step 3** (69 s): `reports/error_gallery.md` and 5 figures. Most NEU-DET misses are boxes of the right class that miss IoU 0.5 (74 of 84, 73 of 79 and 75 of 103 misses have a same-class kept box with IoU >= 0.1), concentrated in crazing and rolled-in scale; VisA errors are mostly false alarms, and on macaroni2 the frozen thresholds flag 45-50 of 50 normal images. The notes are hand-written (`reports/error_gallery_notes.md`) and feed back into nothing. **Flags:** the cost ratios are assumptions; the VisA halves are 67% (cashew) or 50% anomalous, not a production prevalence; NEU-DET `val` also chose each checkpoint, so its selection costs are optimistic, and NEU-DET `test` had been scored in Phase 2 at a fixed 0.5 (no threshold tuned on it); the frozen file records git `+dirty` because the tree was uncommitted; the evaluation split was read once by step 2 and again, read-only, by the gallery runs (5 runs while tuning the figure layout and the notes) with no threshold change; D-FINE-S is rebuilt from the cached HF weights before the checkpoint overwrites them |
| 4b Leave-one-defect-type-out on pcb1 | **Done** (code, tests, 🔴 runs made by you; verdict: weakly supports, undecided overall) | **Code:** `data/visa_to_coco.py` (mask → boxes, defect types from `image_anno.csv`), `detection/visa_detector.py`, `eval/{unseen,unseen_run}.py`, `configs/unseen/pcb1.yaml`, `scripts/{run_unseen,render_unseen}.py`, tests `test_{visa_to_coco,unseen,visa_detector}.py`; `NeuDetDataset` got a `read_image` hook and an `augment_empty` flag (defaults keep the steel path identical). **Data facts (counted from `image_anno.csv` and the saved splits):** pcb1 has 100 anomalous images, 92 with one defect type (melt 46, missing 17, bent 14, scratch 14) and 8 with several; images containing each type in the evaluation half: melt 28, missing 7, bent 9, scratch 13. **Smoke (🟢, nothing logged):** 3 detector iterations + the three checkpoints on 200 images, 47 s wall, reloaded checkpoints reproduce the saved 4a scores exactly. **Runs (🔴, you ran them, one MLflow run each, all at commit `184f6cd`, 5 runs in total, recipe unchanged):** detector train 259-306 s per run, peak VRAM 1,769-1,770 MiB allocated / 2,062-2,066 reserved, process peak RAM 3,053-3,078 MiB; every run reproduced the saved 4a scores (max abs diff 0). **Results (5% budget, recall on the held-out type in the evaluation half, detector / PatchCore / AnomalyDINO / EfficientAD):** bent (9) 6/9, 9/9, 9/9, 6/9; melt (28) 15, 17, 16, 24; missing (7) 0/7, 6/7, 4/7, 5/7; scratch (13) 10/13, 9/13, 8/13, 13/13. Control (`none`, seen types) 44/50, 37, 33, 41. Realised false alarms on 50 evaluation normals at 5%: 2-6. **Verdict:** weakly supports the headline claim; clear for `missing` (0/7 [0.00, 0.35] against 6/7 [0.49, 0.97] PatchCore at 2/50 false alarms each), model-dependent for `melt`, within noise for `bent` and `scratch`; contradicts it for no type; no anomaly model wins everywhere; small sample (7-28 unseen images), one seed, one category, no AUROC intervals. Full tables and notes in `reports/unseen_defects.md`. **Decision:** the second read of `threshold_evaluation` was approved on the condition that these are the only runs and the recipe is not changed after seeing results; both held, and the decision is written in the report. **Gates:** ruff and mypy clean, 205 tests in 31.6 s (176 before), 6 slow tests in 106 s on CPU (3 before) **Exploratory follow-up (outside the approved protocol, added at your request; `src/defectlens/eval/unseen_followup.py`, `scripts/render_unseen_followup.py`, `reports/unseen_followup.md` + notes, `tests/test_unseen_followup.py`; no new run, no data read, only `reports/unseen/*.json`):** on the same evaluation images the detector catches fewer of a type when it was held out than when trained on, in all four types at both budgets as point estimates (5%: bent 6/9 vs 9/9, melt 15/28 vs 24/28, missing 0/7 vs 5/7, scratch 10/13 vs 13/13); the paired outcomes never reverse at 10% and reverse for 2 images in total at 5%. Exact McNemar p at 5%: 0.250, 0.022, 0.062, 0.250 (uncorrected for 8 tests; none passes 0.05/8). The held-out detector also trains on fewer images (23 for melt to 42 for bent, against 47), one seed each, so this is not novelty alone; the Phase 4b verdict is unchanged. The anomaly models' negative control is exact (identical scores, 0 discordant images). |
| 5a ONNX export and ORT CUDA | **Done** (🟢, run by me; no 🔴 run needed; MLflow phase tag `5a`, git `5958a573` + uncommitted changes) | **Code:** `export/{onnx_export,verify,verify_run,benchmark,benchmark_run,config,report}.py`, `configs/export/default.yaml`, `scripts/{export_onnx,verify_onnx,benchmark,render_benchmark}.py`, `tests/test_{export,benchmark}.py`, `reports/{benchmark.md,benchmark_notes.md,onnx_export.json,onnx_verification.json,benchmark.json}`; `detection/dfine.py` refactored (`decode`, `to_detections` shared by PyTorch and ORT paths). **Exported (opset 20, batch 1):** D-FINE-S 51.7 MiB, EfficientAD 31.1 MiB (4 categories), PatchCore 137.8 MiB and AnomalyDINO 515.5 MiB (pcb1), Faster R-CNN 165.3 MiB. The Dynamo exporter fails on Faster R-CNN (`GuardOnDataDependentSymNode` in torchvision's head); the TorchScript exporter works and ORT matches PyTorch's val metrics. **D-FINE, ORT TF32 off vs PyTorch fp32:** AP@50 / AP@50:95 identical to 4 decimals (val 0.7103 / 0.4345, test 0.7308 / 0.4447), but the pre-set slot-wise tolerance 1e-3 holds for 233/270 (val) and 234/270 (test) images only (max abs logits 3.1 in some low-ranked slot; top-10 overlap 1.0000, top-100 0.9998 / 0.9996). **TF32 on vs off:** raw outputs change (0/270 images within 1e-3, top-100 overlap 0.71) but AP@50:95 differs by -0.0000 [-0.0013, +0.0009] (val) and +0.0001 [-0.0008, +0.0011] (test), paired bootstrap, 150 resamples: not detectable, about 0.001 not excluded; PyTorch's own cuDNN default and the Phase 2 AMP path differ from strict fp32 in raw outputs just as much with the same AP. Default for D-FINE stays TF32 off (decided on val and latency; test only reported). **Other models:** Faster R-CNN val AP@50:95 0.3991 equal to PyTorch (TF32 off); anomaly image scores within 9.3e-5 relative (TF32 off), identical AUROC and Spearman 1.0 in all six rows. **Latency p50, batch 1, model only, ORT CUDA TF32 off / on (PyTorch strict fp32):** D-FINE 30.89 / 28.87 ms (27.45), EfficientAD 12.95 / 9.03 (13.77), PatchCore 10.44 / 9.20 (10.88), AnomalyDINO 28.71 / 23.21 (36.45), Faster R-CNN 77.01 / 45.94 (77.56). **ORT CUDA is not uniformly faster than PyTorch eager** (D-FINE is slower). AnomalyDINO needs 4,942-4,960 MiB device memory under ORT (1,850-1,922 under PyTorch). Details, limits and what was not measured: `reports/benchmark.md` and its notes. |
| 5b onward | Not started | TODO |

## Context

DefectLens is a portfolio project for CV/ML engineering roles: a supervised detector for known
defects plus an anomaly detector for unseen defects, exported to ONNX and served as an API. The
project folder held only `CLAUDE.md` when this plan was written, so this is a greenfield build.

**Revision 2 — your corrections applied:**

1. **WSL2:** 12 GB cap kept, 8 processors, `autoMemoryReclaim=gradual` under `[experimental]`, full file
   shown, and `free -h` expectations separate total from available (Phase 0a).
2. **Dependencies:** `onnxscript` is explicit, torchvision is pinned to match torch, and the AnomalyDINO
   encoder name is settled from installed source with a real forward pass (Sections 4 and Phase 0c).
3. **TensorRT:** ONNX Runtime CUDA is the guaranteed deployment target. TensorRT FP16/INT8 is an optional,
   time-boxed experiment that can't block the MVP (Sections 3–4, Phases 0c, 5a, 5b, 6, risks 1–2).
4. **VisA evaluation:** a named `threshold_calibration` subset and an untouched `threshold_evaluation`
   subset, with selection and evaluation as separate steps (Section 3, Phases 1a, 3a, 4a).
5. **Licensing:** uncertain dataset terms are documented, and no license is assumed for trained weights
   (Section 3, Phases 1a, 6c).

**Git boundary:** Claude runs no git command that changes the repo; you run `git init`, `dvc init`, commits
and pushes.

Resource figures below are **estimates**, not measurements. Each is replaced by the measured value
when its phase finishes. No accuracy numbers appear here, on purpose.

---

## 1. Environment findings (checked 2026-11-01)

- **GPU:** RTX 4060 Laptop, 8188 MiB, driver 621.01, CUDA 13.5. 525 MiB in use at idle. Visible in
  both Windows and WSL.
- **OS:** Windows 11 Home build 26200; this session runs in Git Bash on native Windows.
- **WSL2 is already partly set up, contrary to your note:** `Ubuntu` 26.04.1 (default distro, WSL 2,
  user `svkaw`, created about Oct 20) and `docker-desktop` both exist. WSL is version 3.0.3.0. No
  `.wslconfig`, so WSL sees only 7 GB RAM and 2 GB swap.
- **Python:** Windows has 3.15 (default), 3.14, 3.12. WSL Ubuntu has 3.14. **Neither has 3.11**; uv
  will install it.
- **Tools:** git present on both (2.56 Windows, 2.53 WSL). **uv missing on both.** gcc missing on both.
  Docker CLI 29.9.1 on Windows, but the daemon is not running. In Ubuntu, `docker` printed a blank
  version, which suggests Docker Desktop's stub, meaning WSL integration is off.
- **Disk:** 775 GB free on C:, which also backs the WSL disk.
- **Repo:** not a git repository yet.

Not checked: one combined command was blocked by the permission classifier, so I re-ran narrower
read-only versions. That leaves host CPU, total RAM and firmware virtualization unread from Windows.
CPU and RAM below come from `CLAUDE.md`; WSL's 7 GB fits the 50% default on 16 GB. Reading Ubuntu
started the stopped distro; nothing else changed apart from its `.motd_shown` timestamp.

---

## 2. Recommendation: build inside WSL2

| Factor | WSL2 | Native Windows |
|---|---|---|
| TensorRT | Main ORT/TensorRT path, with no DLL or PATH juggling | Wheels exist, but more manual setup |
| Docker | Same Linux userland as the image; Docker Desktop already uses WSL | Containers still run in WSL |
| anomalib | Linux-first; `fork` workers and symlinks work | `spawn` workers re-import torch per worker, costing RAM |
| 16 GB RAM | Needs a cap, costs some RAM for the VM | Slightly lighter |
| CI | Same as `ubuntu-latest` | Differs |
| GPU speed | NVIDIA reports at least 90% of native even in its worst-case test; long jobs near native | Native |

**Decisions:**

- Keep the repo and data on the **ext4 disk (`~/projects/DefectLens`)**, never `/mnt/c`, which is
  several times slower for small files.
- Run Claude Code **inside Ubuntu** so tools and paths match. This Windows session can't run the
  project directly.
- Cap WSL at 12 GB, leaving 4 GB for Windows, and start with 8 processors. Docker Desktop shares this VM
  and cap, so stop containers during training.

---

## 3. Where I disagree with the spec

**Changes that affect the design**

1. **The MVP never shows "known + unseen" on the same data.** The detector sees steel and the anomaly
   models see VisA, so nothing proves the anomaly path catches what the detector misses. That is the
   project's headline claim. **Added Phase 4b:** a leave-one-defect-type-out test on VisA. Cuttable,
   but then the README must not make that claim until Phase 7.
2. **NEU-DET has no defect-free images** (300 defect images per class), so an image-level false alarm
   can't be measured on it. Cost-based thresholds therefore run on **VisA** (image level), and on
   NEU-DET at box level only. The README will state that detector false-alarm rates on good steel are
   unmeasured.
3. **VisA has no validation set with anomalies**; training is normal-only and test is mixed. Thresholds
   need labelled anomalies, so the official `1cls` test set is split once, 50/50, stratified by category
   and label, into two named subsets with different jobs:

   | Subset | Allowed use | Forbidden |
   |---|---|---|
   | `threshold_calibration` | Choosing thresholds, cost operating points, model and hyperparameter choices | Reporting final thresholded results |
   | `threshold_evaluation` (untouched) | One final thresholded evaluation with frozen thresholds | Any selection or tuning, including after seeing its errors |

   - The official full `1cls` test is reserved for **threshold-free** AUROC/AUPRO, for comparison with
     literature. It overlaps calibration, so it never produces a thresholded number and never drives a choice.
   - All selection happens first, and thresholds are frozen to a file before evaluation starts.
4. **INT8 calibration in TensorRT is deprecated** (since 10.1) in favour of explicit quantization. The
   optional INT8 experiment uses **QDQ ONNX**, not `trt_int8_enable`. Accuracy loss on D-FINE's transformer
   decoder is a real unknown, and a documented loss is still a valid result.
5. **Not every model belongs in the deployment table.** PatchCore and AnomalyDINO do a kNN lookup against
   a memory bank, which suits GPU compilers poorly, and Faster R-CNN has dynamic control flow.
   **Required ORT CUDA targets: D-FINE and EfficientAD.** The others get an ONNX + ORT CUDA attempt, with
   failures recorded. Optional TensorRT runs cover D-FINE and EfficientAD only.
6. **Test ONNX export in Phase 0, not Phase 5.** ONNX on ORT CUDA is the guaranteed deployment target, so
   a failure there could change the choice of detector.

**Scope and method**

7. **The D-FINE vs RT-DETRv2 choice:** D-FINE-S is primary. RT-DETRv2 runs only if D-FINE fails the
   export spike. Doing both adds a session without adding insight.
8. **The classical baseline has no class output.** It gets class-agnostic AP and recall, compared
   against class-agnostic scores from the deep models, not per-class mAP.
9. **"PRO" means AUPRO**, integrated to FPR 0.3 (anomalib's convention), stated in the report.
10. **The cost ratio is an assumption** with no real factory number behind it. I'll report 5:1, 10:1
    and 50:1 sweeps, and label them as such.
11. **Few-shot needs repeats:** one reference image is highly sample-dependent, so 5 fixed draws per
    shot count, reported as mean ± std.
12. **The "backbone" ablation** is PatchCore/WRN50 vs AnomalyDINO/DINOv2-S. Those backbones differ along
    with their methods, so the table will say so rather than imply a clean single-variable test.
13. **Phase 9 has no video source.** Default: a synthetic conveyor video built by scrolling dataset
    images, which also gives ground-truth IDs and so real tracking metrics.
14. **Own images (Phase 7):** 50–100 photos are too few to train and test on separately. They are for
    a same-domain demo and a domain-shift test, with results flagged as small-sample.
15. **Phases 0, 1, 2a, 3, 5, 6, 7 and 8 are split** to fit one session each.

**Proposed `CLAUDE.md` edit (yours to accept):** MLflow moved to SQLite as default in 3.7 and put the
file backend in maintenance mode. Use `sqlite:///mlruns/mlflow.db` with artifacts in `mlruns/artifacts`.
Everything stays local and under the git-ignored `mlruns/`.

**Dataset usage terms are uncertain, and the plan doesn't resolve them:**

- **NEU-DET:** the official page I read asks for citation and states no license terms; a third-party card
  says it has no stated license.
- **VisA:** sources I found conflict (CC BY 4.0 vs CC BY-NC-SA 4.0).
- **MVTec AD and pretrained checkpoints:** terms not checked yet; recorded when first used.
- **Handling:** Phase 1a records each source's exact wording, URL and retrieval date in `docs/DATA_TERMS.md`,
  with unresolved items marked unresolved. Datasets are not redistributed, the DVC remote stays local, and the
  README links to official sources.
- **Trained weights:** no license or usage status is assumed either way. The README says their terms are
  unresolved and that the repo's code license doesn't cover them. Weights stay out of git and the Docker image.
  Whether to publish any is your decision, once the terms are checked.

---

## 4. Version baseline (checked 2026-11-01; freeze in `uv.lock` at Phase 0b)

| Package | Version | Note |
|---|---|---|
| Python | 3.11 | Now the minimum for torch, transformers and ORT, so there's no margin to go lower |
| torch | ==2.14.1+cu130 | PyPI has 2.15.0, but the cu130 index stops at 2.14.1 and that index is what matters |
| torchvision | ==0.29.1+cu130 | Pinned as torch's pair: its metadata requires `torch>=2.14.0`, and the cu130 index has a `cp311` wheel. That bound is only a minimum, so the exact pin is what keeps them matched |
| anomalib | ==2.7.0 | Needs torch ≥2.10. AnomalyDINO arrived in 2.3.0 |
| transformers | 5.22.1 | `DFineForObjectDetection` |
| onnx | latest at lock | Required for export |
| onnxscript | latest at lock | **Explicit direct dependency** for the Dynamo ONNX export path. anomalib lists it only under its `openvino` extra, which this project doesn't install |
| onnxruntime-gpu | 1.31.1 | **Guaranteed deployment target.** CUDA 13 deps via the `cuda` extra; has `cp311` manylinux wheels |
| tensorrt (PyPI) | 11.4.0.106 | Optional experiment only, not installed by default. **sdist only**, no wheels |

**Deployment target:** ONNX Runtime with `CUDAExecutionProvider` is the guaranteed path for the benchmark,
API, Docker image and MVP. TensorRT FP16 and INT8 are optional. No MVP phase depends on them.

**Env layout (uv):** dependency groups `cu130` (default, local) and `cpu` (CI), declared as conflicting,
plus `serve` for a slim image with no torch. The `cpu` group uses the same torch and torchvision versions from
the CPU index. `uv run pytest` must work with no flags.

**Execution key:** 🟢 Claude runs it (under about 5 min). 🔴 You run it and paste the output. Fix seeds in
every run; log config, git hash and dataset version. At each phase end: `uv run pytest`, `uv run ruff
check .`, `uv run mypy src` pass, `PLAN.md` and README updated, changed files plus a commit message
handed to you.

---

## 5. Phases

### Phase 0a — WSL2 and tooling (all 🔴, needs sudo and your Windows settings)
**Goal:** a capped, GPU-enabled WSL2 with the repo on ext4.
**Packages:** apt `build-essential libgl1 libglib2.0-0`; uv; Claude Code in WSL.
**Files:** `C:\Users\svkaw\.wslconfig` (outside the repo). This is the full file:
```ini
[wsl2]
memory=12GB
processors=8
swap=8GB

[experimental]
autoMemoryReclaim=gradual
```
**`autoMemoryReclaim` placement, checked against your setup:** you're right. Your installed WSL is 3.0.3.0, and
Microsoft's `.wslconfig` reference (updated 2026-10-16) lists this key under `[experimental]`, with `disabled`,
`gradual` and `dropCache` as values and `dropCache` as the default. The reference doesn't tie that to a specific
WSL version, and I can't test it until the file exists, so the first step after restart is to watch for any
`.wslconfig` warning from WSL. `memory`, `processors` and `swap` are confirmed under `[wsl2]`.

**Commands** — in PowerShell, after saving the file:
```powershell
wsl --update
```
```powershell
wsl --version
```
```powershell
wsl --shutdown
```
Check the WSL version again after the update. If it changed from 3.0.3.0, recheck the key against the current
docs. Then in Ubuntu:
```bash
free -h
```
```bash
nproc
```
**Expected from `free -h`:**
- `Mem:` **total** just under 12 Gi, because the kernel reserves a little.
- `Mem:` **available** is lower than total and varies with load. It's the number that shows usable memory; at
  idle it should sit close to total.
- `Swap:` total about 8 Gi.
- A small `free` column with large `buff/cache` is normal after heavy file reads. `gradual` releases that cache
  slowly once WSL has been idle, so judge memory by `available`, not `free`.

`nproc` should print 8. Then:
```bash
sudo apt update && sudo apt install -y build-essential libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.11
mkdir -p ~/projects && cp -r /mnt/c/Users/svkaw/Documents/Projects/DefectLens ~/projects/
cd ~/projects/DefectLens && git init
```
Install Claude Code in Ubuntu and sign in yourself, then enable Docker Desktop → Settings → Resources →
WSL Integration → Ubuntu.
```bash
nvidia-smi
docker run --rm --gpus=all ubuntu nvidia-smi
```
**Expected:** no `.wslconfig` warning at start; both `nvidia-smi` calls list the RTX 4060.
**Don't:** install NVIDIA Linux drivers or apt's CUDA toolkit in WSL. The Windows driver is the only
one needed, and PyTorch bundles its own runtime.
**Done when:** `free -h` and `nproc` match the above, GPU checks pass, gcc and uv work, `claude` starts in
`~/projects/DefectLens`. After you finish, you paste the `free -h`, `nproc` and `wsl --version` output.
**Resources:** no GPU load; 30–45 min of your time, mostly downloads. Raising processors above 8 later is a
one-line change, if CPU-bound data loading turns out to be the bottleneck.

### Phase 0b — uv project and quality skeleton (🟢)
**Goal:** a locked env with CUDA torch and working lint, type and test gates.
**Packages:** `torch==2.14.1 torchvision==0.29.1 transformers timm anomalib==2.7.0 onnx onnxscript
onnxruntime-gpu[cuda] opencv-python-headless scikit-image albumentations pycocotools hydra-core mlflow dvc
fastapi gradio`; dev `pytest ruff mypy`. No TensorRT package here.
**Files:** `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`, `.dvcignore`,
`src/defectlens/{__init__,seed,tracking}.py`, package `__init__`s per the repo layout, `configs/config.yaml`,
`scripts/check_env.py`, `tests/test_seed.py`, `tests/test_tracking.py`, `PLAN.md`, `README.md` stub.
**Commands:**
```bash
uv sync
uv run python scripts/check_env.py
uv run pytest && uv run ruff check . && uv run mypy src
```
You run `dvc init` (it stages files in git):
```bash
uv run dvc init
```
**Expected:** `check_env` prints torch `2.14.1+cu130`, torchvision `0.29.1+cu130`, CUDA available, the GPU
name, `onnxscript` importable, and the ORT provider list including `CUDAExecutionProvider`. Tests pass in under 60 s.
**Done when:** the env reproduces from `uv.lock`, a dummy MLflow run logs config + git hash + dataset
version.
**Resources:** under 1 GB VRAM, 3 GB RAM, about 10 min; 8–10 GB of disk for the env.

### Phase 0c — Spike: prove the risky paths (🟢)
**Goal:** run each high-risk piece once before building on it.
**Packages:** none new.
**Files:** `scripts/spike_dfine.py`, `scripts/spike_anomalydino.py`, `scripts/spike_export.py`,
`scripts/download_visa.py`, `src/defectlens/data/download.py`, `tests/test_download.py`, `reports/spike_notes.md`.
**Commands:**
```bash
uv run python scripts/download_visa.py
uv run python scripts/spike_dfine.py
uv run python scripts/spike_export.py
uv run python scripts/spike_anomalydino.py category=cashew shots=5
uv run dvc add data/raw/visa
```
**Expected:**
- D-FINE-S: logits and box shapes printed; peak VRAM logged.
- **Export (required):** the Dynamo path, with `onnxscript`, writes an ONNX file. It loads in ORT and outputs
  match PyTorch within tolerance on `CUDAExecutionProvider`.
- **TensorRT probe (informational, can't fail this phase):** lists whether `TensorrtExecutionProvider` is
  present, without installing anything extra.
- **AnomalyDINO encoder, in order, before any data is used:**
  1. Print `anomalib.__version__` (must be 2.7.0) and the installed `anomaly_dino` source path.
  2. Read the `encoder_name` default from that installed `lightning_model.py` and `torch_model.py`, and use
     `inspect.signature` to confirm it. This is the authority, not the docs.
  3. Import the class, instantiate it, and confirm timm resolves the name.
  4. Run a real forward pass on a dummy tensor, and print the feature shape.
- AnomalyDINO end-to-end: image AUROC printed on `cashew` test. This is a smoke value, not a reported result.

**Encoder evidence so far:** anomalib's default-branch source uses `vit_small_patch14_dinov2`, matching the
reference docs. The `v2.7.0` tag URLs returned 404, so the release itself is unconfirmed. Also, the code only
checks that the name contains "dino", so `dinov2_vit_small_14` would pass that check; only steps 3–4 prove a name.

**Done when:** D-FINE, Dynamo export + ORT CUDA, and AnomalyDINO all ran; the encoder name is pinned in config
from installed source. `reports/spike_notes.md` records versions, peak VRAM, opset, the encoder name and the
TensorRT probe result. If D-FINE can't export to ORT CUDA, switch to RT-DETRv2 here.
**Resources:** D-FINE about 1–2 GB VRAM, 1 min. AnomalyDINO about 1–2 GB VRAM, 4–6 GB RAM, 2–4 min. VisA
download 10–30 min, about 2 GB (unverified size, plus a prepared copy).

### Phase 1a — Data: NEU-DET, COCO, fixed splits (🟢, with one possible 🔴 manual download)
**Goal:** both datasets versioned, with saved splits.
**Packages:** `gdown`.
**Files:** `src/defectlens/data/{neu_det,coco,splits,visa_splits}.py`, `scripts/{download_neu_det,prepare_data}.py`,
`configs/data/{neu_det,visa}.yaml`, `data/splits/*.json`, `docs/DATA_TERMS.md`, `tests/test_{neu_det,coco,splits}.py`,
tiny XML fixtures.
**Commands:**
```bash
uv run python scripts/download_neu_det.py
uv run python scripts/prepare_data.py
uv run dvc add data/raw/neu_det data/processed data/splits
```
**Expected:** 1,800 images, 6 classes; stratified 70/15/15 → 1,260 / 270 / 270. Each `data/splits/visa_<category>.json`
holds `train`, `threshold_calibration` and `threshold_evaluation`, the last two from a 50/50 split of `1cls` test,
per category and label. Validator prints no empty, out-of-bounds or duplicate boxes, and no image in two subsets.
Confirm the raw annotation format once downloaded (likely Pascal VOC XML).
**Start with 4 VisA categories:** `cashew`, `candle`, `pcb1`, `macaroni2` (single object, multi-instance,
complex, hard).
**Done when:** splits are saved and hash-tracked; re-running gives identical files; `docs/DATA_TERMS.md` records
each dataset's exact wording, URL, retrieval date and open questions; tests pass.
**Resources:** CPU only, under 2 GB RAM, under 5 min after download.

### Phase 1b — OpenCV preprocessing (🟢)
**Goal:** one preprocessing path, shared by training and serving.
**Packages:** none.
**Files:** `src/defectlens/preprocess/{resize,clahe,roi,pipeline}.py`, `configs/preprocess/default.yaml`,
`scripts/preview_preprocess.py`, `tests/test_preprocess.py`, `reports/figures/preprocess_grid.png`.
**Commands:**
```bash
uv run python scripts/preview_preprocess.py
uv run pytest tests/test_preprocess.py
```
**Expected:** a before/after grid; tests cover box remapping after resize and crop, grayscale→3-channel,
and CLAHE determinism.
**Done when:** boxes stay aligned through every transform; all three gates pass. Calibration stays in Phase 7.
**Resources:** CPU only, 1 GB RAM, under 1 min.

### Phase 2a.1 — Evaluation code and classical baseline (🟢)
**Goal:** the evaluation code every detector uses, plus the first baseline.
**Packages:** none.
**Files:** `src/defectlens/eval/{detection,confusion,report}.py`, `src/defectlens/detection/classical.py`,
`configs/model/classical.yaml`, `scripts/{evaluate,run_classical}.py`, `tests/test_{detection_metrics,confusion,classical}.py`.
**Commands:**
```bash
uv run python scripts/run_classical.py
```
**Expected:** class-agnostic AP and recall on val/test, logged to MLflow. Thresholds tuned on val only.
**Done when:** a hand-computed tiny fixture matches the metric code exactly, and a perfect prediction
scores 1.0. Run first in the suite so later models inherit trusted metrics.
**Resources:** CPU only, 2 GB RAM, under 2 min.

### Phase 2a.2 — Faster R-CNN (🔴 training; 🟢 smoke)
**Goal:** a CNN reference point, using ResNet50-FPN v2.
**Packages:** none.
**Files:** `src/defectlens/detection/{faster_rcnn,datamodule}.py`, `configs/model/faster_rcnn.yaml`,
`scripts/train_detector.py`, `tests/test_faster_rcnn.py`.
**Commands:**
```bash
uv run python scripts/train_detector.py model=faster_rcnn trainer.fast_dev_run=true
uv run python scripts/train_detector.py model=faster_rcnn
```
**Expected:** smoke (🟢, under 2 min) ends with a valid eval dict. Full (🔴) logs loss, mAP@50, mAP@50:95
and per-class recall per epoch to MLflow.
**Done when:** the best-val checkpoint is evaluated once on test, and the run is logged.
**Resources:** AMP, batch 4, input about 480: **3–4 GB VRAM, 6 GB RAM, 20–40 min** for 30 epochs.
**Measured (2026-11-06):** 1.75 GiB peak allocated, 41 min; results in the status table above.

### Phase 2b.1 — D-FINE fine-tune (🔴 training; 🟢 smoke)
**Goal:** train the main detector, D-FINE-S fine-tuned from `ustc-community/dfine-small-coco`.
**Packages:** none.
**Files:** `src/defectlens/detection/{dfine,common}.py` (new), `trainer.py` (backend dispatch, accumulation,
early stopping), `datamodule.py` (optional scale/shift augmentation), `faster_rcnn.py` (scheduler moved to
`common.py`, behaviour unchanged), `configs/model/dfine_s.yaml`, `tests/test_dfine.py`, `tests/conftest.py`
(shared tiny-data fixture; tests use a tiny random-weight D-FINE, no download).
**Commands:**
```bash
uv run python scripts/train_detector.py model=dfine_s trainer.fast_dev_run=true
uv run python scripts/train_detector.py model=dfine_s
```
**Design decisions taken from the Faster R-CNN run** (val AP@50 plateaued near epoch 10 while val recall fell
from about 0.82 to 0.74 and train loss kept dropping, so the model overfits 1,260 images):
- **Early stopping on val** (`patience: 10` epochs without a better val AP@50:95, `epochs: 40` instead of
  the planned 60). The best-val checkpoint is already what gets scored on test, so stopping only saves time;
  it never touches test.
- **Stronger augmentation:** on top of flips and brightness/contrast, random scale 0.75-1.25 and shift up to
  10% (`scale_translate: 0.5`), boxes transformed with the image, borders mid-gray.
- Not done: EMA, mosaic/mixup, a lower LR. Both changes above are applied together, so the report cannot
  attribute any gain to one of them, and D-FINE differs from Faster R-CNN in many other ways anyway.
  Whether either helps is read from the val curves of the full run, not assumed.
- Optimizer settings are from the official D-FINE recipe, scaled, and untuned: AdamW, LR 1e-4 (backbone
  0.5x), weight decay 1e-4 (not on biases/norms), grad clip 0.1, cosine with 100 warm-up optimizer steps,
  effective batch 16 (8 x 2). Input 640 to match the ONNX export spike.
**Done when:** a full run completes without OOM; the best-val checkpoint is saved; peak VRAM is logged.
**Resources (measured in dev runs):** AMP, batch 8, accumulate 2, 640 px: **3.4 GiB peak VRAM allocated**
(3.6 GiB reserved), about 55 s/epoch, so 40 epochs is at most about 37 min (estimate; early stopping may stop
sooner). The OOM fallbacks (batch 4 x 4, input 512, D-FINE-N) are not needed at this VRAM.

### Phase 2b.2 — Comparison and confusion matrix (🟢)
**Goal:** the results table.
**Packages:** `matplotlib` (was only a transitive dependency; now declared).
**Files:** `src/defectlens/eval/compare.py`, `scripts/compare_detectors.py`, `tests/test_compare.py`,
`reports/detection_comparison.md`, `reports/figures/confusion_*.png`; `trainer.py` gained an optional
`model.run_name` so a rerun does not collide with the original run's name.
**Commands:**
```bash
uv run python scripts/compare_detectors.py [runs=classical,faster_rcnn,faster_rcnn_aug,dfine_s]
```
**Done:** the table covers the classical baseline, Faster R-CNN and D-FINE-S, is generated from MLflow, has a
background row in the confusion matrices and per-class AP and recall, and states that the comparison is not
like-for-like. Confusion matrices are row-normalised heat maps with counts, at IoU 0.5 and score 0.5.
**Findings (from the generated report, single seed, 270 test images):**
- AP@50:95: D-FINE ahead (test 0.4442 vs 0.4224, val 0.4346 vs 0.3780). AP@50: Faster R-CNN ahead (test
  0.7613 vs 0.7304, val 0.7228 vs 0.7105). So there is no single winner, and the gaps are small relative to
  what split noise could produce (unmeasured).
- At score 0.5, D-FINE has higher precision (0.7411 vs 0.6778) and fewer false alarms (151 vs 230 on test)
  but lower recall (0.6927 vs 0.7771); its `recall_max` is higher (0.9936 vs 0.9315). A fixed 0.5 is not a
  neutral operating point across the two score types; cost-based thresholds come later.
- Crazing is the weakest class for both, and D-FINE misses most of it at 0.5 (recall 0.2020; 79 of 99
  boxes missed in the matrix). Rolled-in scale is also worse for D-FINE (AP@50 0.4895 vs 0.6173).
- Errors are almost all misses and false alarms, not class swaps: wrong-class matches are 3 (Faster R-CNN)
  and 1 (D-FINE).
**Matched Faster R-CNN rerun (done, row 2b.3 of the status table):** the one recipe difference that can be matched
exactly is the augmentation, so `faster_rcnn_aug` adds only `scale_translate: 0.5` (SGD vs AdamW, 480 vs 640 px
and batch stay each architecture's own). Command used, with a separate checkpoint directory so the original
`models/faster_rcnn/best.pt` is kept:
```bash
uv run python scripts/train_detector.py model=faster_rcnn +model.augment.scale_translate=0.5 +model.run_name=faster_rcnn_aug model.checkpoint_dir=models/faster_rcnn_aug 2>&1 | tee ~/faster_rcnn_aug_run.log
uv run python scripts/compare_detectors.py runs=classical,faster_rcnn,faster_rcnn_aug,dfine_s
```
Result: D-FINE-S's AP@50:95 lead over Faster R-CNN roughly halves on test (0.022 to 0.012) and Faster R-CNN's
AP@50 lead doubles (0.031 to 0.062); see row 2b.3 for the numbers and caveats. Optional, 2x run time:
3 seeds of each model, which is what it would take to say more.
**Resources:** CPU only, under 1 min.

### Phase 3a — PatchCore and AnomalyDINO (🔴 training; 🟢 smoke)
**Goal:** the first two anomaly models, with metrics.
**Packages:** none.
**Files:** `src/defectlens/anomaly/{base,patchcore,anomaly_dino,datamodule}.py`, `src/defectlens/eval/anomaly.py`,
`configs/model/{patchcore,anomaly_dino}.yaml`, `scripts/train_anomaly.py`, `tests/test_anomaly_{metrics,datamodule}.py`.
**Commands:**
```bash
uv run python scripts/train_anomaly.py model=patchcore category=cashew
uv run python scripts/train_anomaly.py -m model=patchcore,anomaly_dino category=cashew,candle,pcb1,macaroni2
```
**Expected:** threshold-free image AUROC, pixel AUROC and AUPRO per category, logged to MLflow, on the full
official test and on `threshold_calibration`. Model and hyperparameter choices use `threshold_calibration` only.
`threshold_evaluation` is not touched in Phase 3.
**Done when:** each category has both models logged; heatmaps saved; anomalib's `val` data is explicitly
`threshold_calibration`, so its adaptive threshold and normalization never see `threshold_evaluation`.
**Scope change (flag):** the original text asked for metrics on "the full official test and on
`threshold_calibration`" while also saying `threshold_evaluation` is not touched. The full official test contains
`threshold_evaluation`, so Phase 3 scores `threshold_calibration` only; the evaluation half is used once in Phase 4.
**Design:** we do not use anomalib's `Engine`/datamodule (their splits and adaptive thresholds are not ours). Our
`VisaImages` reads the fixed split files; the torch models from anomalib run inside thin wrappers. PatchCore's
memory bank is built by our own streaming coreset (embeddings in CPU RAM, greedy k-center on 128-d random
projections), because anomalib's version holds every embedding on the GPU (about 4.3 GB plus a copy for 900 images).
**Resources:** measured in the full run, see the status row of 3a (PatchCore about 1.5 GiB VRAM and 6.2 GiB RAM on 900 images,
AnomalyDINO about 2.7 GiB VRAM; 295 s fit + score in total). Close other GPU apps first.
**Full command (🔴 because it exceeds the "smoke" scope, expected under 10 min):**
```bash
uv run python scripts/train_anomaly.py model=patchcore,anomaly_dino category=cashew,candle,pcb1,macaroni2 2>&1 | tee ~/anomaly_3a.log
```
(the cashew runs were repeated, so MLflow has two `*_cashew` runs per model with identical metrics; `compare` scripts take the latest.)

### Phase 3b — EfficientAD (🔴 overnight; 🟢 smoke)
**Goal:** the speed baseline.
**Packages:** none. Teacher weights (40 MB) and Imagenette (1.56 GB, 13,394 images) download automatically on first use
into `~/.cache/anomalib` and `data/raw/imagenette` (git-ignored); both are already downloaded.
**Files:** `src/defectlens/anomaly/efficientad.py`, `configs/model/efficientad.yaml`, runner/script changes,
`tests/test_efficientad.py`.
**Commands:**
```bash
# smoke (done): 500 steps, about 1 min once the downloads exist, kept apart in MLflow
uv run python scripts/train_anomaly.py model=efficientad category=cashew model.max_steps=500 +run_suffix=_smoke500
# full (overnight, one process per category)
uv run python scripts/train_anomaly.py model=efficientad category=cashew,candle,pcb1,macaroni2 2>&1 | tee ~/efficientad_3b.log
```
(The step count is `model.max_steps`, not `trainer.max_steps` as first written; it is a model setting.)
**Done when:** all 4 categories are logged, batch size 1 as the method requires.
**Resources (measured in the full run):** 1,202 MiB VRAM, 2.4-2.7 GiB RAM per category (separate process each), 16.5-16.8
steps/s, about 70 min per category and 4.65 h for the four. The smoke-based estimate was 75 min per category and 5 h
(RAM guessed at 3-3.5 GiB), so the real run was slightly faster and lighter than extrapolated.

### Phase 3c — Few-shot curve (🔴)
**Goal:** the 1/5/10-shot curve with AnomalyDINO, and the three-model comparison table.
**Packages:** none.
**Files:** `src/defectlens/anomaly/few_shot.py`, `scripts/few_shot.py`, `src/defectlens/eval/anomaly_compare.py`,
`scripts/compare_anomaly.py`, `data/splits/few_shot_draws.json`, `tests/test_few_shot.py`, `tests/test_anomaly_compare.py`,
`reports/{anomaly_comparison.md,figures/few_shot_curve.png}`.
**Commands:**
```bash
# smoke (done): 1 category, 1 draw, separate MLflow store
uv run python scripts/few_shot.py shots=1,5,10 draws=1 category=cashew mlflow.tracking_uri=sqlite:///mlruns/few_shot_smoke.db
# full (🔴): 4 categories x 3 shots x 5 draws = 60 runs, resumable (finished runs are skipped)
uv run python scripts/few_shot.py shots=1,5,10 draws=5 2>&1 | tee ~/few_shot_3c.log
uv run python scripts/compare_anomaly.py        # tables + figure, reads MLflow only
dvc add data/splits                              # the draws file lives in data/splits
```
**Expected:** a curve with error bars (mean ± sample std over 5 draws) per category and averaged, the full-data runs of
the three models as dashed reference lines, and the three-model tables, all generated from MLflow.
**Done when:** draws are saved to disk and reused (done and tested), and the full-data run appears as the reference line
(done; the 🔴 job has run and `reports/figures/few_shot_curve.png` is the real figure).
**Scope notes (flag):** all runs are scored on `threshold_calibration` like the rest of Phase 3. The 5 draws are
nested prefixes, independent between categories; the `mean` row averages categories within a draw index first. Per-run
peak RAM is not logged for the 60 runs (they share one process; the figure would belong to no single run).
**Resources:** smoke-measured, 3 runs in 24.5 s wall including start-up (so at most about 8 s per run, which bounds the
job at about 8 min; the likely figure is 4-6 min, not measured), peak VRAM at most about 0.2 GiB (155 MiB at 10 shots on
cashew), RAM about 2.5 GiB (process max RSS 2.4 GB in the smoke). The first plan said 1-2 GB VRAM, 6 GB RAM, 1-2 h. **Measured in the full job:** at most 155 MiB VRAM, about 4 min wall (inferred from file timestamps), so the smoke-based bound of 8 min held.

### Phase 4a — Cost-based thresholds and error gallery (🟢, needs Phases 2–3 outputs)
**Goal:** operating points chosen on calibration data, one untouched evaluation, then a look at the worst misses.
**Packages:** none.
**Files (as built):** `src/defectlens/eval/{thresholds,threshold_runs,scoring,gallery,gallery_run}.py`, `configs/cost/default.yaml`,
`scripts/{select_thresholds,evaluate_thresholds,render_thresholds,error_gallery}.py`, `reports/{frozen_thresholds.json,threshold_evaluation.json,thresholds.md,thresholds_notes.md,error_gallery.md,error_gallery_notes.md}`,
`reports/figures/error_gallery_*.png`, `tests/test_{thresholds,scoring,gallery}.py`, `tests/detection_fixtures.py`.
**Commands, in this order:**

Step 1 reads only `threshold_calibration` (VisA) or `val` (NEU-DET), and writes the frozen file with its hash:
```bash
uv run python scripts/select_thresholds.py cost.fn_fp_ratio=5,10,50
```
Step 2 reads the frozen file and `threshold_evaluation` (VisA) or `test` (NEU-DET); it has no tuning options:
```bash
uv run python scripts/evaluate_thresholds.py
```
Step 3 runs last:
```bash
uv run python scripts/error_gallery.py top_k=20
```
**Expected:** a threshold per model and ratio, then one thresholded result table with the frozen file's hash logged.
**Done when:**
- Selection and evaluation are separate scripts with separate data access. Tests assert the selector rejects
  evaluation data, and the evaluator refuses to run without a frozen file.
- A second evaluation on changed thresholds is treated as a protocol breach and disclosed in `reports/thresholds.md`.
- The gallery has written notes on failure patterns. Those notes don't feed back into any threshold; a later
  change would need a new untouched subset, or a disclosed result.
**Resources (measured wall time):** 71 s selection, 67 s evaluation, 69 s per gallery run (each mostly model loading). VRAM and RAM were not recorded for these three scripts; the plan's estimate was about 2 GB VRAM, 6 GB RAM, under 5 min.
**Scope notes (flag):** `scripts/render_thresholds.py` (new, reads no data) re-renders `thresholds.md` from the saved evaluation and the hand-written notes. Selection of the NEU-DET detectors needs model inference on `val`, so step 1 also loads checkpoints; step 1 asserts that each reloaded model reproduces what training saved. Per-class and per-pixel thresholds are out of scope (one image-level threshold per VisA model and category, one score threshold per detector).

### Phase 4b — Leave-one-defect-type-out proof (mixed; cuttable, added by me)
**Goal:** directly test the headline claim on one domain.
**Packages:** none.
**Files (as built):** `src/defectlens/data/visa_to_coco.py` (masks → boxes), `src/defectlens/eval/unseen.py` (protocol, numbers,
report), `configs/unseen/pcb1.yaml`, `scripts/run_unseen.py`, `tests/test_visa_to_coco.py`, `reports/unseen_defects.md`.
**Added (flag):** `src/defectlens/detection/visa_detector.py` (the NEU-DET dataset and trainer are tied to steel),
`src/defectlens/eval/unseen_run.py` (orchestration with an injected anomaly scorer, so the protocol is testable),
`scripts/render_unseen.py` (builds the report from the result files, like 4a), `tests/test_{unseen,visa_detector}.py`,
`reports/unseen/pcb1_<holdout>.json` (one per run, written by the runs, committed).
**Deviation:** the config is `configs/unseen/pcb1.yaml`, not `configs/experiment/unseen_pcb1.yaml`. A directory named
`configs/experiment/` makes Hydra read `experiment=<name>` (the MLflow experiment name, used by `tests/test_dfine.py`) as a
config-group override and the test failed with "Could not override 'experiment'". **Phase 8a's `configs/experiment/ablation.yaml`
would break the same way: use another directory name.**
**Protocol (decided before any run, written in `eval/unseen.py` and the report):**
- One run per held-out type T in {bent, melt, missing, scratch}, plus a control (`holdout=none`).
- Detector: Faster R-CNN ResNet50-FPN v2 (COCO-pretrained), class-aware with a class for every type except T, 800 px, batch 2,
  20 fixed epochs, last epoch used (no validation set, no checkpoint selection). Training set: `threshold_calibration`
  anomalies with exactly one defect type and not T (a mask cannot be split by type, so the 8 multi-type images are never
  trained on) plus 100 normal images from `train` (same sample every run). Boxes: mask fragments within 25 px merge, fewer
  than 20 mask pixels are dropped, 6 px padding (chosen by looking at box sizes on the masks, not at any metric).
- Anomaly models: the Phase 3 PatchCore, AnomalyDINO and EfficientAD checkpoints, unchanged, reloaded with the 4a
  reproduce-the-saved-scores check.
- Operating points (same rule for all four models, not the 4a cost thresholds, which stay frozen): lowest threshold whose
  false-alarm rate on the 50 `threshold_calibration` normal images is at most 5% or 10%. Only normal scores are used, so the
  detector having trained on calibration anomalies cannot leak into its threshold (a test makes the selection function fail
  if it reads any other score).
- Reported: recall on `threshold_evaluation` anomalies that contain T (unseen) and that do not (seen, the detector's sanity
  check), realised false-alarm rate on its 50 normals, and threshold-free image AUROC (all images containing T, both halves,
  none of them in any training set, against all 100 test normals). Wilson 95% intervals; **small-sample**: unseen images per
  type in the evaluation half are 28, 7, 9 and 13.
- Image-level scoring: a detector's score is its highest box score; an image counts as caught if flagged anywhere, which is
  generous to both kinds of model. `check_same_recipe` stops the renderer from mixing runs made with different settings, and
  `run_unseen.py` refuses to overwrite a result unless `unseen.force=true`.
**Commands:**
```bash
# 🔴 all five runs, one process each (estimated 20-25 min in total, each run about 4-5 min)
uv run python scripts/run_unseen.py category=pcb1 holdout=bent,melt,missing,scratch,none 2>&1 | tee ~/unseen_4b.log
# 🟢 afterwards, no GPU, a few seconds
uv run python scripts/render_unseen.py
```
A single type: `uv run python scripts/run_unseen.py category=pcb1 holdout=<defect_type>`. Path check without results
(about 1 min, writes nothing, prints no evaluation-half number): `... holdout=melt unseen.smoke=true`.
**Expected:** detector vs anomaly recall on the held-out type, in the same table (`reports/unseen_defects.md`), plus the
seen-type check. Few images per defect type, so flagged as small-sample.
**Done when:** the table exists and the README claim matches what it shows, whichever way it falls. **Met:** the table is
`reports/unseen_defects.md`, the README states the verdict.
**Measured results** (5% budget; full tables with intervals, the 10% budget, seen types, realised false alarms and AUROC are in
`reports/unseen_defects.md`, built from `reports/unseen/pcb1_*.json`):

| held-out (unseen eval images) | Faster R-CNN | PatchCore | AnomalyDINO | EfficientAD |
|---|---|---|---|---|
| bent (9) | 6/9 | 9/9 | 9/9 | 6/9 |
| melt (28) | 15/28 | 17/28 | 16/28 | 24/28 |
| missing (7) | 0/7 | 6/7 | 4/7 | 5/7 |
| scratch (13) | 10/13 | 9/13 | 8/13 | 13/13 |
| control, all seen types (50) | 44/50 | 37/50 | 33/50 | 41/50 |

**Verdict:** weakly supports the claim (anomaly path catches types the detector never saw), decides it for `missing` only,
contradicts it for no type, cannot decide it overall. The detector is not weak on seen types (0.80-0.95 recall at 5%), its
recall on the held-out type is below its seen-type recall in all four runs (disjoint intervals for melt and missing), but
type difficulty is not separated from novelty. The anomaly models are a complement, not better detectors (control: 44/50
against 33-41). **Exploratory follow-up (done, outside the approved protocol):** the detector's per-type recall in the control run (type
trained on) against the held-out run, on the same evaluation images, from the saved per-image scores only (no run, no data
read). Analysis fixed before computing (docstring of `eval/unseen_followup.py`): paired by image, exact McNemar, single-type
images as a secondary check, AUROC on evaluation-half images, negative control on the anomaly models. Result in
`reports/unseen_followup.md`: held-out recall is lower in all four types (5%: 6/9 vs 9/9 bent, 15/28 vs 24/28 melt, 0/7 vs 5/7
missing, 10/13 vs 13/13 scratch); the held-out run never catches an image the control run misses at 10%; exact p 0.250 / 0.022 /
0.062 / 0.250 at 5%, uncorrected. It does not isolate novelty: the held-out detector has fewer training images (23 to 42 against 47)
and one seed. The third read of `threshold_evaluation`; nothing tuned from it.
**Resources, measured in the real runs:** detector training 259-306 s per run (0.21 s per iteration, not the 0.14 s of the smoke estimate), peak VRAM 1,769-1,770 MiB allocated / 2,062-2,066 reserved, process peak RAM 3,053-3,078 MiB. The MLflow start times of consecutive runs are 299-340 s apart, so the five runs took about 27 min (derived from start times, not timed; the estimate below was 20-25 min). **Earlier, in the smoke:** 1,768 MiB
(3 iterations) to 2,110 MiB (25 iterations) allocated, 2,400 MiB reserved; process peak RAM 3.09 GB; 47 s wall for the smoke.
**Estimated, not measured:** 20 epochs of 62-74 iterations (62 melt, 68 missing, 71 bent and scratch, 74 control) at about 0.14 s is
2.9-3.5 min of training, plus about 45 s of loading and scoring: about 4 min per run and 20 min for the five (the 20-25 min above
leaves margin for the rough 0.14 s). The 🔴 label is for the total; peak VRAM should stay below 3 GiB.
**Scope notes (flag):**
- The held-out *type* is a different thing from a held-out *class* in NEU-DET: the detector has no class for T, but at image
  level it can still raise an alarm on T through a known class, and that counts as a catch.
- `threshold_evaluation` was already used once (4a frozen thresholds); here it is read again for a different question with
  thresholds set on calibration normals only. No 4a threshold or result changes. The report says so.
- The detector's training set is small (23 to 47 anomalous images, 31 to 68 boxes), so a weak detector is expected for
  reasons unrelated to type novelty; that is what the seen-type table and the control run are for.
- **Disclosure (kept for the record):** my first two smoke runs (`holdout=melt`, before and after adding the anomaly models) printed evaluation-half
  rows. The detector had 3 training iterations then, so its rows say nothing, and the anomaly rows are the deterministic 4a
  checkpoints. No setting was changed after seeing them. The smoke now prints calibration-side numbers only.
- Only pcb1 was run; `category=` is generic but its masks were not checked for other categories.

### Phase 5a — ONNX export and ORT CUDA (🟢) — **required, the guaranteed deployment target**
**Goal:** verified ONNX models and the benchmark the MVP relies on.
**Packages:** `onnxsim` if useful (`onnxscript` is already in place from Phase 0b).
**Files:** `src/defectlens/export/{onnx_export,verify,benchmark}.py`, `configs/export/default.yaml`,
`scripts/{export_onnx,benchmark}.py`, `tests/test_{export,benchmark}.py` (tiny CPU model), `reports/benchmark.md`.
**Commands:**
```bash
uv run python scripts/export_onnx.py
uv run python scripts/benchmark.py providers=cuda
```
**Expected:** PyTorch-vs-ORT output difference within tolerance per model. Protocol: batch 1, warm-up 50,
500 runs, reporting p50/p95 ms and throughput, on `CUDAExecutionProvider`.
**Done when:** D-FINE and EfficientAD are exported, verified and benchmarked on ORT CUDA, with accuracy compared
to PyTorch. The others are attempted, failures recorded. **This gate alone lets Phase 6 start.** Note the numbers
come from WSL.
**Resources:** 2–3 GB VRAM, 6 GB RAM, 10–15 min split into short runs.
**Files (as built):** `src/defectlens/export/{onnx_export,verify,benchmark}.py` as planned, plus `configs/export/default.yaml`,
`scripts/{export_onnx,benchmark}.py`, `tests/test_{export,benchmark}.py`, `reports/benchmark.md`.
**Added (flag):** `export/{verify_run,benchmark_run,config,report}.py` (orchestration kept out of the scripts, like 4b),
`scripts/verify_onnx.py` (accuracy against PyTorch, incl. the TF32 question and MLflow logging) and `scripts/render_benchmark.py`
(report from the JSON files, like 4a), `reports/{onnx_export,onnx_verification,benchmark}.json` (committed, the report is built
from them), `reports/benchmark_notes.md`. `detection/dfine.py`: the top-k decode and the mapping back to the image frame moved
out of `predict` into `decode` / `to_detections` so the ORT path uses the identical post-processing (tests unchanged and green).
`onnxsim` was not added: ORT 1.31.1's own graph optimiser already broke the HGNetV2 stem in the spike, so no further graph
rewriting was introduced.
**Deviations:** (1) `scripts/benchmark.py` takes `providers=cuda` as planned, plus `models=`; each (model, variant) runs in its
own process. (2) The D-FINE tolerance 1e-3 fixed before the measurement is a slot-wise max abs difference; it is **not** met on
every image even with TF32 off (see the results), so permutation-free statistics were added and reported next to it. (3) The
PyTorch reference is CUDA fp32 with TF32 switched off (`strict_fp32`), because PyTorch's default already uses TF32 convolutions.
**Commands (as run, all 🟢, each under 5 min):**
```bash
uv run python scripts/export_onnx.py                                       # about 50 s, CPU, peak RAM under 3 GB
uv run python scripts/verify_onnx.py models=dfine_s verify.dfine_splits=val   # 246 s, peak RAM 3,081 MiB (test: same, 246 s)
uv run python scripts/verify_onnx.py models=efficientad                       # 33 s;  faster_rcnn 77 s;  patchcore 20 s;  anomaly_dino 63 s
uv run python scripts/benchmark.py providers=cuda models=dfine_s              # 153 s; efficientad 92 s; faster_rcnn 228 s; patchcore 101 s; anomaly_dino 160 s
uv run python scripts/render_benchmark.py
```
VRAM during the verification runs was not measured; the benchmark rows give device memory per variant (424-4,960 MiB, see the
report). **Disclosure:** the first D-FINE `val` verification (6 min 3 s, over the 5-minute guide) reported slot-wise differences
only and a bootstrap over three pairs; it showed that this statistic is dominated by a few images, so the statistics were
extended, the bootstrap reduced to the TF32 pair, and val was rerun together with test. Test was not read by the first run. No
setting that affects an AP number changed between the two.
**Done when:** met. D-FINE and EfficientAD are exported, verified and benchmarked on ORT CUDA; Faster R-CNN, PatchCore and
AnomalyDINO were attempted and also work (Faster R-CNN via TorchScript). **Phase 6 may start.** The numbers come from WSL.

### Phase 5b — TensorRT FP16 and INT8 (🟢 short runs; 🔴 if over 5 min) — **optional experiment**
**Goal:** see how much latency TensorRT gains, without any risk to the MVP.
**Rules:** one session, time-boxed. Runs after the MVP or in parallel with it; Phases 6a–6c don't depend on it.
Any compatibility issue ends with a recorded reason, not a change to the core env. Install in a separate uv env,
so the locked MVP env stays unchanged. The API, Docker image and CI never require TensorRT.
**Packages:** the first working route from risk 1, in the separate env. Possibly `nvidia-modelopt` for QDQ
(confirm first).
**Files:** `src/defectlens/export/{quantize,tensorrt}.py`, `scripts/quantize.py`, `tests/test_quantize.py`
(skipped when TensorRT isn't installed), `reports/benchmark_tensorrt.md`.
**Commands:**
```bash
uv run python scripts/quantize.py
uv run python scripts/benchmark.py providers=tensorrt precision=fp16,int8
```
**Expected:** a separate table per model and precision: ms per image, throughput, accuracy loss against ORT CUDA.
INT8 calibration uses NEU-DET `val` or VisA `threshold_calibration` only.
**Done when:** each row is measured, or marked "not run" with the reason. Stopping with no TensorRT numbers still
counts, and the README then says so plainly.
**Resources:** 2–4 GB VRAM, 8 GB RAM; first TensorRT engine build takes minutes; 15–30 min total.

### Phase 6a — FastAPI (🟢)
**Goal:** image in → defects, boxes, heatmap.
**Packages:** `uvicorn python-multipart httpx` (test).
**Files:** `src/defectlens/serve/{app,schemas,inference,model_registry}.py`, `configs/serve/default.yaml`,
`scripts/serve.py`, `tests/test_api.py`.
**Commands:**
```bash
uv run python scripts/serve.py
curl -F image=@tests/fixtures/sample.jpg "localhost:8000/inspect?product=steel"
```
**Expected:** JSON with defects, boxes and scores, plus a base64 heatmap for anomaly products. `/health` reports
the active provider, which is `CUDAExecutionProvider` by default, falling back to CPU.
**Done when:** tests pass on CPU with a tiny ONNX model; errors are handled for bad images and unknown products.
`product` selects the models, so steel and VisA products aren't fused across domains.
**Resources:** 1–3 GB VRAM, 3 GB RAM, seconds per start.

### Phase 6b — Gradio and Docker (🟢, 🔴 if builds exceed 5 min)
**Goal:** a demo and a slim image.
**Packages:** none new.
**Files:** `src/defectlens/serve/demo.py`, `Dockerfile` (multi-stage, `serve` group, no torch), `.dockerignore`,
`docker-compose.yml`, `scripts/docker_smoke.sh`.
**Commands:**
```bash
docker build -t defectlens:local .
docker run --rm --gpus=all -p 8000:8000 -v "$PWD/models:/app/models:ro" defectlens:local
bash scripts/docker_smoke.sh
```
**Expected:** `/health` responds and an `/inspect` call succeeds, with `CUDAExecutionProvider` active. Weights are
mounted, not baked in. The image contains no TensorRT.
**Done when:** the smoke script passes; image size is recorded.
**Resources:** 2 GB VRAM, 4 GB RAM; build 5–15 min first time.

### Phase 6c — CI and README (🟢; you commit and push) — **end of MVP**
**Goal:** green CI and a README built from measured results.
**Packages:** none new.
**Files:** `.github/workflows/ci.yml` (uv, `cpu` group, ruff, mypy, pytest, Docker build; written early, in the 4a session, without the Docker step), `README.md`,
`scripts/build_report_tables.py`, `reports/figures/*`.
**Commands:**
```bash
uv run python scripts/build_report_tables.py
```
**Expected:** README tables generated from `reports/`; any unmeasured number stays a TODO.
**Done when:** you push and CI is green; a clean clone follows the README to a running API; every number
traces to an MLflow run or pasted output. The README links `docs/DATA_TERMS.md`, cites the datasets, describes
open questions as open, and says trained-weight terms are unresolved and not covered by the code license.
TensorRT results appear only if Phase 5b produced them.
**Resources:** CPU only; CI target under 10 min.

--- **Extensions below** ---

### Phase 7a — Own images and labelling (🔴, your time)
**Goal:** a labelled same-domain dataset.
**Packages:** none in the project env, because Label Studio's pins are heavy. Run it via `uvx` or Docker.
**Files:** `configs/data/own.yaml`, `scripts/import_label_studio.py`, `docs/capture_protocol.md`, `tests/test_import_label_studio.py`.
**Commands:**
```bash
uvx label-studio
uv run python scripts/import_label_studio.py path/to/export.json
uv run dvc add data/raw/own
```
**Expected:** a validated COCO file with normal and defect images, in fixed splits.
**Done when:** 50–100 images labelled, at least some normal-only images exist for anomaly training.
**Resources:** CPU only; 2–3 h of capture and labelling.

### Phase 7b — Calibration and own-data evaluation (🟢 + 🔴)
**Goal:** undistorted input and a same-domain end-to-end test.
**Packages:** none.
**Files:** `src/defectlens/preprocess/calibration.py`, `scripts/{calibrate,evaluate_own}.py`, `tests/test_calibration.py`.
**Commands:**
```bash
uv run python scripts/calibrate.py path/to/checkerboards
uv run python scripts/evaluate_own.py
```
**Expected:** reprojection error printed; detector plus anomaly results on the same images.
**Done when:** detector and anomaly routes run on one domain; the README flags it as small-sample.
**Resources:** 3–4 GB VRAM, about 10 min (🔴 if fine-tuning runs longer).

### Phase 8a — CutPaste/DRAEM synthetic defects and ablation (🟢 code; 🔴 runs)
**Goal:** measure whether synthetic data improves recall on real defects.
**Packages:** none.
**Files:** `src/defectlens/anomaly/synthetic.py`, `configs/experiment/ablation.yaml`, `scripts/ablation.py`,
`tests/test_synthetic.py`, `reports/ablation.md`.
**Commands:**
```bash
uv run python scripts/ablation.py -m synthetic=off,cutpaste,draem
```
**Expected:** an ablation table across backbone, shots, synthetic on/off and optionally CLAHE on/off.
**Done when:** real-defect recall is compared at a threshold chosen on `threshold_calibration`, even if the
effect is null. `threshold_evaluation` is read once per configuration, and that repeated use is disclosed.
**Resources:** 3–5 GB VRAM, 8 GB RAM, 2–4 h.

### Phase 8b — Stable Diffusion inpainting, optional (🔴)
**Goal:** a test of whether generated defects add anything beyond 8a.
**Packages:** `diffusers accelerate`.
**Files:** `src/defectlens/anomaly/sd_inpaint.py`, `scripts/generate_sd_defects.py`, `configs/synth/sd.yaml`.
**Commands:**
```bash
uv run python scripts/generate_sd_defects.py n=200
```
**Done when:** an extra row in the ablation table. Skip if 8a shows no benefit.
**Resources:** SD 1.5 inpainting, FP16, attention slicing: **4–6 GB VRAM, 8 GB RAM, 30–60 min** for 200 images.

### Phase 9 — Tracking (🟢 short clips)
**Goal:** stable defect IDs across frames.
**Packages:** `supervision`.
**Files:** `src/defectlens/tracking_video/{synth_video,track}.py`, `scripts/{make_conveyor_video,track_video}.py`,
`tests/test_track.py`, `reports/tracking.md`.
**Commands:**
```bash
uv run python scripts/make_conveyor_video.py
uv run python scripts/track_video.py
```
**Expected:** an annotated video plus MOTA and IDF1 from synthetic ground truth, labelled as synthetic.
**Done when:** IDs persist, and FPS is measured with the ONNX detector.
**Resources:** 2–3 GB VRAM, 4 GB RAM, under 5 min per clip.

---

## 6. Run summary

| Run | VRAM | RAM | Time | Who |
|---|---|---|---|---|
| Spikes (0c) | 1–2 GB | 4–6 GB | under 5 min each | 🟢 |
| Faster R-CNN | 3–4 GB | 6 GB | 20–40 min | 🔴 |
| D-FINE-S | 4–6 GB | 6–8 GB | 45–90 min | 🔴 |
| PatchCore, 4 categories | **1.5 GiB (measured)** | 6.2 GiB (measured) | about 4 min fit + score (measured) | 🔴 |
| AnomalyDINO, 4 categories | 2.7 GiB (measured) | not measured per run (1.9 GiB for the 450-image cashew run) | about 1.5 min fit + score (measured) | 🔴 |
| EfficientAD, 4 categories | 1.2 GiB (measured) | 2.4-2.7 GiB (measured per run) | **4.65 h (measured; 70 min per category)** | 🔴 overnight |
| Few-shot, 60 runs | at most 0.2 GiB (measured, 155 MiB at 10 shots) | not logged per run (smoke: 2.4 GB process max RSS) | about 4 min wall (inferred from file timestamps, not timed) | 🔴 |
| Thresholds, evaluation, gallery (4a) | not recorded | not recorded | 71 s + 67 s + 69 s (measured) | 🟢 |
| ORT CUDA benchmark (required) | 2–3 GB | 6 GB | 10–15 min | 🟢 |
| TensorRT benchmarks (optional) | 2–4 GB | 8 GB | 15–30 min | 🟢/🔴 |
| Ablation | 3–5 GB | 8 GB | 2–4 h | 🔴 |

Before long runs: plug in, set a performance power mode, disable sleep, close browsers and overlays.

---

## 7. Top risks

| # | Risk | Evidence | Fallback |
|---|---|---|---|
| 1 | **TensorRT EP route is unclear (no MVP impact)** | ORT docs' table stops at 1.22 / TensorRT 10.9 / CUDA 12, while 1.31.1 wheels pull CUDA 13. PyPI metadata doesn't mention TensorRT, and `tensorrt` is sdist-only | ORT CUDA ships regardless. Phase 5b tries, in order, in a separate env and within its time box: `TensorrtExecutionProvider` in `onnxruntime-gpu` → NVIDIA's `onnxruntime-ep-nv-tensorrt-rtx-cu12` 0.5.0 (bundled runtime, Linux and Windows wheels, FP16/QDQ support unconfirmed) → native TensorRT → older pinned ORT. If none works, record why and stop |
| 2 | INT8 ruins D-FINE accuracy (no MVP impact) | Implicit path deprecated; transformer INT8 has a history of regressions | Exclude sensitive nodes from QDQ, use partial quantization; otherwise report the loss |
| 3 | D-FINE ONNX export fails on ORT CUDA (**0c: exports and matches, with the three caveats in `reports/spike_notes.md`; D-FINE stays primary**) | No official documented HF `small` export found; Dynamo path depends on `onnxscript` | Caught in 0c. Next: RT-DETRv2, then upstream `Peterande/D-FINE` export. This is the deployment blocker to watch |
| 4 | Library versions drift | Fast-moving; PyPI torch already ahead of cu130 index | `uv.lock` frozen at 0b; upgrades only as a deliberate step |
| 5 | Windows vs WSL2 file and memory issues | `/mnt/c` slow; shared GPU memory is slow past VRAM | ext4 repo, 12 GB cap, 8 GB swap, 2–4 workers. Native Windows only if WSL GPU breaks |
| 6 | VRAM limits | PatchCore bank ≈ 4.3 GB before model | AMP, accumulation, patch subsampling, CPU coreset; D-FINE-N |
| 7 | 16 GB RAM pressure | WSL cap leaves 12 GB | Few workers, one job at a time, Docker stopped during training |
| 8 | NEU-DET download fails | IEEE DataPort needs a subscription; no plain HF mirror | Official Google Drive link via `gdown`; else manual browser download into `data/raw/`; else Baidu link. A community Kaggle copy is unverified |
| 9 | VisA download fails | Official AWS bucket, no account needed | anomalib auto-download; DagsHub mirror |
| 10 | AnomalyDINO API differs in 2.7.0 | Docs show two encoder names; `v2.7.0` source unread; substring check accepts both; ONNX untested | Settled from installed source and a forward pass in 0c. Few-shot uses saved image lists, so export isn't required |
| 11 | Thermal throttling | Laptop, 109 W cap | Plugged in, raised stand, runs resume from checkpoints |
| 12 | Docker GPU in WSL | Integration appears off | Fixed in 0a; CPU image as a working fallback for the demo |
| 13 | Dataset terms stay unresolved | NEU-DET terms unstated; VisA sources conflict | Document as open in `docs/DATA_TERMS.md`; no redistribution; no assumption on weights; publishing weights stays your decision |

---

## 8. Verification

- **Each phase:** `uv run pytest` (under 60 s, CPU; tests above 5 s carry `@pytest.mark.slow`, are skipped by default and run by `pytest -m slow` and in CI), `uv run ruff check .`, `uv run mypy src`, plus that phase's
  "done when" list.
- **Data integrity:** no image in two subsets; split hashes logged with every run. Tests block threshold selection
  on `threshold_evaluation` or `test`, and block evaluation without a frozen threshold file.
- **Metrics:** hand-computed fixtures for detection and threshold code before any model is compared.
- **Number provenance:** report tables are generated from MLflow or pasted output, never typed by hand.
- **MVP end-to-end:** fresh clone → `uv sync` → `docker run` → `curl /inspect` on a steel image and a VisA image
  returns boxes and a heatmap, served by `CUDAExecutionProvider` → CI green after your push. TensorRT isn't part
  of this check.

---

## 9. Sources

- [anomalib releases](https://www.github.com/openvinotoolkit/anomalib/releases), [AnomalyDINO docs](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/models/image/anomaly_dino.html), [anomalib on PyPI](https://pypi.org/project/anomalib/)
- [onnxruntime-gpu on PyPI](https://pypi.org/project/onnxruntime-gpu/), [ORT TensorRT EP docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html), [TensorRT-RTX EP plugin](https://pypi.org/project/onnxruntime-ep-nv-tensorrt-rtx-cu12/), [tensorrt on PyPI](https://pypi.org/project/tensorrt/)
- [TensorRT 10.1.0 release notes](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/release-notes-10/10.1.0.html)
- [PyTorch cu130 wheels](https://download.pytorch.org/whl/cu130/torch/), [torchvision cu130 wheels](https://download.pytorch.org/whl/cu130/torchvision/), [torchvision 0.29.1 metadata](https://pypi.org/project/torchvision/0.29.1/), [torch on PyPI](https://pypi.org/project/torch/), [transformers on PyPI](https://pypi.org/project/transformers/)
- [Microsoft: WSL `.wslconfig` reference](https://learn.microsoft.com/en-us/windows/wsl/wsl-config), [WSL September 2023 update](https://devblogs.microsoft.com/commandline/windows-subsystem-for-linux-september-2023-update/)
- [anomalib `anomaly_dino/torch_model.py`, default branch](https://raw.githubusercontent.com/open-edge-platform/anomalib/main/src/anomalib/models/image/anomaly_dino/torch_model.py), [`lightning_model.py`, default branch](https://raw.githubusercontent.com/open-edge-platform/anomalib/main/src/anomalib/models/image/anomaly_dino/lightning_model.py)
- [D-FINE nano ONNX card](https://huggingface.co/onnx-community/dfine-nano-coco-ONNX)
- [NEU database page (Kechen Song)](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm), [IEEE DataPort NEU-DET](https://ieee-dataport.org/documents/neu-det), [dronefreak/NEU-DET card](https://huggingface.co/datasets/dronefreak/NEU-DET)
- [VisA on AWS Open Data](https://registry.opendata.aws/visa), [anomalib VisA datamodule](https://anomalib.readthedocs.io/en/latest/markdown/guides/reference/data/datamodules/image/visa.html)
- [NVIDIA: CUDA performance on WSL2](https://developer.nvidia.com/blog/leveling-up-cuda-performance-on-wsl2-with-new-enhancements), [WSL shared GPU memory discussion](https://github.com/microsoft/WSL/discussions/10452), [WSL2 setup guide (secondary source)](https://insiderllm.com/pdfs/wsl2-local-ai-windows-guide.pdf)
- [MLflow backend store docs](https://mlflow.org/docs/latest/self-hosting/architecture/backend-store/)
