Hand-written; every number is from the tables above. One seed, one split, 50 anomalous and 25-50
normal images per VisA category, so only large gaps mean anything.

- **Thresholds chosen on 75-100 images carried over only partly.** On VisA the cost per image was
  higher on the evaluation half than on the selection half for 31 of 36 thresholds. At fn_fp_ratio 10
  the total over the four categories rose from 104 / 96 / 99 (PatchCore / EfficientAD / AnomalyDINO)
  to 152 / 146 / 129, against 175 for the trivial policy.
- **A zero-miss selection is set by one image.** 30 of the 36 selections missed no anomalous image
  on the selection half, because at these ratios the cheapest threshold sits just below the
  lowest-scoring anomalous image there. 13 of those 30 missed at least one on the evaluation half. At
  ratio 50 the evaluation misses add up to 7 (3 PatchCore, 2 AnomalyDINO, 2 EfficientAD) where the
  selection had none, and each costs 50, so all three models end above the trivial policy
  (201 / 226 / 274 against 175).
- **A threshold cannot fix a weak ranking.** On macaroni2 (image AUROC 0.72-0.83) AnomalyDINO ties
  the trivial policy at every ratio (it flags all 50 normal images) and PatchCore is worse than it
  (64 / 56 / 96 against 50). Where the ranking is good (pcb1 EfficientAD, cashew AnomalyDINO) the
  threshold cuts the cost to half of flagging everything or less (25 against 50; 6 against 25).
- **Model order on VisA.** AnomalyDINO has the lowest total evaluation cost at all three ratios
  (105 / 129 / 201), EfficientAD is next at 5 and 10, PatchCore last. The totals are driven by a
  handful of misses worth 5-50 each and by one threshold per category, so the order is not shown to
  be meaningful.
- **NEU-DET.** All nine thresholds beat the trivial policy by a wide margin (for example D-FINE-S at
  ratio 10: 1,578 against 6,280). The matched Faster R-CNN has the lowest test cost at every ratio
  (871 / 1,359 / 3,440), D-FINE-S is not the lowest at any, although it has the best AP@50:95; at IoU
  0.5 the cost follows recall and AP@50 more than AP@50:95, as in Phase 2. The gaps between detectors
  (for example 871 against 918 at ratio 5) are single-seed and not shown to be meaningful. Test cost
  per box was not higher than val cost in any of the nine cases, matching Phase 2 where test scores exceeded val;
  one split of 270 images each cannot say why.
- **Assumptions.** The ratios are assumed, the VisA class balance is the benchmark's (about 50%
  anomalous), and the NEU-DET val split also chose the checkpoints, so its selection costs are
  optimistic.
- **Protocol.** `evaluate_thresholds.py` was run once on the frozen file above. Before it, the same
  code path was rehearsed on the selection splits only (it reproduced all 45 frozen counts exactly).
  The error gallery re-read the evaluation splits afterwards (read-only, no threshold touched). No
  threshold was changed after the evaluation.
