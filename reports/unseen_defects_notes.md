Hand-written after reading the tables above; every number is from them. One seed, one category, one detector
recipe, fixed before the runs and not changed after them.

**Verdict on the headline claim (the anomaly path catches defect types the detector never saw): weakly
supports it, decides it for one of four types, contradicts it for none, and cannot decide it overall.**

- **`missing` is the clear case.** The detector, trained without it, flags 0/7 [0.00, 0.35] at the 5% budget; PatchCore
  flags 6/7 [0.49, 0.97] with the same 2/50 realised false alarms, and the intervals do not overlap. AUROC says the same
  (0.759 for the detector against 0.982 PatchCore, 0.926 AnomalyDINO, 0.936 EfficientAD). At the 10% budget the gap
  stays in the point estimates (2/7 against 6/7) but the intervals overlap.
- **`melt` depends on the anomaly model.** EfficientAD flags 24/28 [0.69, 0.94] against the detector's 15/28 [0.36, 0.70]
  at 5% (intervals just overlap; both have 6/50 realised false alarms). PatchCore (17/28) and AnomalyDINO (16/28) are
  within one or two images of the detector.
- **`bent` and `scratch` cannot be decided.** Every interval overlaps. On `scratch` the detector (10/13) is ahead of
  PatchCore (9/13) and AnomalyDINO (8/13) and behind EfficientAD (13/13); on `bent` it is behind PatchCore and
  AnomalyDINO (9/9) and level with EfficientAD (6/9). At 10% the detector catches 8/9 bent and 12/13 scratch.
- **Not one anomaly model wins everywhere.** PatchCore is best on `missing` (and ties AnomalyDINO on `bent`), EfficientAD on
  `melt` and `scratch`. Choosing the best model per type afterwards would overstate the result, so read the per-model columns,
  not the best of them. The most consistent is PatchCore: AUROC at or above the detector on all four types (the
  `melt` and `scratch` differences, 0.010 and 0.003, are ties), recall at or above it on three of four at 5%, and
  2/50 realised false alarms in every case. EfficientAD has the highest recall on `melt` and `scratch` but pays for it
  in false alarms (11/50 = 0.22 at the 10% budget) and its `bent` AUROC (0.896) is below the detector's (0.963).
  No AUROC interval was computed, so differences of a few hundredths are not distinguishable.

**Is the detector just weak?** No, not on the types it was trained on. At 5%, its recall on the seen types is 0.80
(bent run), 0.95 (melt), 0.93 (missing) and 0.95 (scratch) and 0.88 in the control run. In the control run that is above
every anomaly model (PatchCore 0.74, AnomalyDINO 0.66, EfficientAD 0.82); in the bent run EfficientAD is ahead (0.85
against 0.80). So the anomaly models are a
complement for types the detector has not seen, not a better detector. In all four runs the detector's recall on the
held-out type is below its recall on the other types (0.67 against 0.80, 0.54 against 0.95, 0.00 against 0.93, 0.77
against 0.95); the intervals are disjoint for `melt` and `missing`. This is the pattern the claim predicts, but it
does not prove that novelty is the cause: the four types differ in difficulty and in size, and the seen set differs
between runs. The anomaly models, which never see any anomaly, show no consistent seen/unseen difference (PatchCore:
1.00 against 0.68 on bent, 0.61 against 0.91 on melt), so their recall tracks the type, not the held-out status.

**What would settle the open part.** Compare the detector's recall on each type when that type is trained on (the
control run) with when it is held out. This was done afterwards from the saved per-image scores, outside the protocol
written before the runs, so it is labelled exploratory: see `reports/unseen_followup.md`. It points the same way (held-out
recall lower for all four types) but does not separate novelty from the smaller training set of the held-out runs.

**Limits.**
- Small sample: 7 (`missing`), 9 (`bent`), 13 (`scratch`) and 28 (`melt`) unseen evaluation images. Intervals are wide and
  one image moves a recall by 4 to 14 points.
- The budget is matched on the 50 calibration normals only. On the 50 evaluation normals the realised false-alarm rate
  differs at the 5% budget: 2/50 for PatchCore, 5/50 for AnomalyDINO, 6/50 for EfficientAD and 2 to 6/50 for the
  detector (at 10%: 5, 5, 11 and 3 to 9). EfficientAD's rates (0.12 and 0.22) are above its budgets by more than their
  intervals. Recall comparisons across models carry that difference.
- An image counts as caught if anything is flagged, wherever. This helps the detector too (a box in the wrong place
  counts), so it cannot explain the detector's low recall.
- The detector trained on 23 to 47 images and 20 fixed epochs, with no validation set; a bigger or better-tuned
  detector might do better on unseen types. One seed per run, so run-to-run variation of the detector is not measured.
- Only pcb1. The result says nothing about other categories or about NEU-DET.
- `threshold_evaluation` has now been read twice (4a frozen evaluation; this); the 4a numbers did not change.
  The smoke runs before the real runs printed evaluation-half rows from a 3-iteration detector; no setting changed
  afterwards (see PLAN.md).
