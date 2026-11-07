Hand-written; every number is from the tables above or from the pasted job output of the 60 runs.
Five nested draws per category, scored on `threshold_calibration` (75-100 images), one seed per
draw: `±` is the spread over draws, not a confidence interval.

**Few-shot AnomalyDINO (image AUROC)**
- More normal images help, with diminishing returns. The mean over the four categories rises from
  0.7700 (1 shot) to 0.8095 (5) to 0.8186 (10), against 0.8959 with all normal images. The step from
  5 to 10 shots (+0.009) is smaller than the spread over draws (0.014-0.016), so it is not shown to
  be a gain; the step from 1 to 5 (+0.040) is larger than the 5-shot spread.
- **Ten shots stay below full data in every category:** gaps of 0.051 (cashew), 0.061 (candle),
  0.140 (pcb1) and 0.057 (macaroni2, 0.040 at 5 shots). Ten-shot AnomalyDINO (mean 0.8186) is below
  full-data PatchCore (0.8899) and EfficientAD (0.9099) in the mean, but note those use 450-900
  normal images, and only AnomalyDINO was run few-shot.
- **A single image can break the 1-shot model.** pcb1 1-shot averages 0.7039 ± 0.1747 because two
  of the five draws score about chance (0.5080 and 0.5200) while the others score 0.80-0.85;
  macaroni2 has two draws near chance too (0.5124, 0.5384). The 1-shot image is part of the 5-shot
  set, and 5 shots already cut the spread (pcb1 ± 0.0251, macaroni2 ± 0.0532). Why those images
  fail was not examined.
- **More images do not always help within a draw.** macaroni2 draw 2 scores 0.7828 at 5 shots and
  0.6932 at 10 (the 5 images are a subset of the 10); the macaroni2 mean at 10 shots (0.6768) is
  below the 5-shot mean (0.6934), within the spread (± 0.048-0.053).
- **Pixel-level quality saturates earlier than image-level ranking.** Pixel AUROC averages 0.9514
  / 0.9661 / 0.9675 for 1 / 5 / 10 shots (full data 0.9775) and AUPRO 0.7767 / 0.8398 / 0.8412
  (full data 0.8620). On cashew the 10-shot AUPRO (0.9262 ± 0.0071) is above the full-data run
  (0.9100); both are single configurations on 75 images, so this is not shown to be a real
  effect.
- **Cost.** Each of the 60 runs took at most 1 s to fit and 1-2 s to score, peak VRAM 106 / 122 /
  155 MiB for 1 / 5 / 10 shots (memory banks of 324 / 1,620 / 3,240 rows). The whole job took about
  4 minutes of wall time (inferred from file timestamps, not timed); per-run RAM was not logged
  because the runs share one process.
