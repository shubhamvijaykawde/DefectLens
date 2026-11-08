Hand-written after reading the tables above; every number is from them. One machine (RTX 4060 Laptop under WSL2), one
run per latency row, one seed; nothing here was tuned on `test`.

**Export.** Four of the five models export with the Dynamo exporter at opset 20 and batch 1. Faster R-CNN does not:
`torch.export` stops on a data-dependent size inside torchvision's detection head (`GuardOnDataDependentSymNode`). The
legacy TorchScript exporter does export it (165.3 MiB) and ORT then reproduces PyTorch's val metrics, so it counts as
exported, by a different route. Not exported: nothing. EfficientAD is exported for all four VisA categories; PatchCore
and AnomalyDINO only for pcb1 (their export works the same way, the other categories were not needed for this gate).

**Accuracy, D-FINE-S (the required gate).**
- **With TF32 off the ONNX model reproduces PyTorch fp32 on the metrics, but not slot for slot on every image.** AP@50 and
  AP@50:95 are identical to four decimals on val (0.7103 / 0.4345) and test (0.7308 / 0.4447), the top-10 overlap of
  (query, class) pairs is 1.0000 and the top-100 overlap is 0.9998 (val) and 0.9996 (test). The slot-wise tolerance of
  1e-3 that I fixed before the measurement is **not** met by every image: 233 of 270 (val) and 234 of 270 (test) are within
  it, the others have a max abs difference up to 3.1 (logits) in some low-ranked slot. The spike's 7e-5 was one random
  input. I have not looked at what those slots are; the sorted top-100 scores agree to 4.4e-4 (val) and 1.2e-3 (test).
- **TF32 on changes the raw outputs but not the metrics.** With ORT `use_tf32=1` no image is within 1e-3, the top-100 overlap
  falls to 0.71, the largest difference between sorted top-100 scores is 0.20 (val) and 0.10 (test), and precision at 0.5
  moves by 0.0017 (val) and 0.0012 (test). AP@50:95 differs from TF32 off by -0.0000 [-0.0013, +0.0009] on val and +0.0001 [-0.0008, +0.0011]
  on test (paired image bootstrap, 150 resamples, 270 images each): no difference is detectable, and differences of about 0.001
  can not be ruled out. So Phase 0c's finding holds for the raw outputs and does not extend to detection quality.
- **PyTorch's own defaults do the same.** The cuDNN default (TF32 convolutions) and the AMP path used in Phase 2 are as far
  from strict fp32 in raw outputs as ORT with TF32 on (0 of 270 images within 1e-3, top-100 overlap 0.71 and 0.60), with the
  same AP. The Phase 2 AMP numbers are reproduced at batch 1: test AP@50 0.7304 (stored 0.7304), AP@50:95 0.4445 (stored
  0.4442); val 0.7109 / 0.4345 (stored 0.7105 / 0.4346).
- **Default for D-FINE: TF32 off** (decided on val and latency, test is only reported). TF32 on saves 2.0 ms at p50 (30.89
  against 28.87 ms), about 7%, and breaks element-wise comparison with PyTorch.

**Accuracy, the other models.** Faster R-CNN (TorchScript route): with TF32 off, val AP@50 0.7473 and AP@50:95 0.3991 equal
PyTorch strict; with TF32 on 0.7490 and 0.3985 (no interval computed). EfficientAD, PatchCore and AnomalyDINO image scores
on `threshold_calibration`: TF32 off differs from PyTorch strict by at most 9.3e-5 relative (efficientad candle), Spearman 1.0 and
identical AUROC in all six rows. TF32 on differs by up to 8.3e-3 relative, Spearman >= 0.99992, AUROC changes by at most 0.0008
(PatchCore 0.9192 to 0.9184). The AUROCs here are agreement checks on the selection split, not results; the AnomalyDINO value
equals the Phase 3 saved scores (0.9304), PatchCore 0.9192 against a saved 0.9188 and EfficientAD 0.9720 against 0.9716
(the saved scores come from PyTorch's cuDNN default, this reference is strict fp32).

**Latency (batch 1, p50, model only).** ORT CUDA is **not uniformly faster than PyTorch eager here**. Like for like (ORT TF32 off
against PyTorch strict fp32): D-FINE 30.89 against 27.45 ms (ORT slower), EfficientAD 12.95 against 13.77, PatchCore 10.44 against
10.88, AnomalyDINO 28.71 against 36.45, Faster R-CNN 77.01 against 77.56. TF32 matters most for the convolution-heavy models:
Faster R-CNN 77.01 to 45.94 ms, EfficientAD 12.95 to 9.03 ms. fp16 autocast in PyTorch is faster than PyTorch fp32 for Faster R-CNN (34.51
against 47.27), EfficientAD (8.43 against 10.69) and AnomalyDINO (29.01 against 36.64) and slower for D-FINE (36.70 against 28.25)
and PatchCore (14.77 against 11.24). I did not look into why
ORT is slower on D-FINE; the timed call includes the host-device copies, which PyTorch's row also includes. Everything in the
p50/p95 columns is a single process per row, so run-to-run variation is not measured (p95 is within 18% of p50 in every row).
The ONNX models are reproducible deployment artefacts; this table does not show them to be a speed-up.

**Resources.** Process peak RAM is 1,657 to 2,673 MiB per benchmark row (about 2,000 MiB for D-FINE; importing torch is part
of it); verification runs peaked at 3,081 MiB (D-FINE, 270 images with five variants) and 2,799 MiB (AnomalyDINO). Device memory
per row (device-wide, includes the CUDA context, other GPU users add noise) is 560-572 MiB for D-FINE, 574-642 MiB for
EfficientAD, 912-960 MiB for PatchCore, 1,278-1,332 MiB for Faster R-CNN and **4,942-4,960 MiB for AnomalyDINO under ORT**
(1,444-1,922 MiB under PyTorch). The 450 MB memory bank is stored in the ONNX file (515.5 MiB) and the ORT session needs about
2.6 times the PyTorch fp32 memory; on this 8 GB laptop, with about 1 GB already used by Windows, that leaves little room for a second
model in the same process. Not investigated.

**Limits and not measured.**
- Model latency only; no letterbox, decode, NMS or heatmap rendering, so API latency in Phase 6 will be higher.
- Batch 1 only (the exported graphs are fixed at batch 1); no throughput-vs-batch curve.
- ORT node placement was not inspected, so nodes ORT put on the CPU are included in the latency without being identified.
- The Phase 4a thresholds were set on scores from PyTorch's cuDNN-default path. ORT with TF32 off matches strict fp32 to 1e-5
  relative, TF32 on to up to 8e-3. How many calibration images change side of a frozen threshold between these variants was not
  measured; Phase 6 should check it before serving with a different precision than the thresholds were calibrated on.
- `test` was read for D-FINE a second time (Phase 2 scored it once per run) to measure export fidelity, not to choose a setting.
- The bootstrap used 150 resamples (each costs about 0.3 s per variant), so interval ends are coarse.
