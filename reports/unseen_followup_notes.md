Hand-written after reading the tables above; every number is from them. **Exploratory, outside the approved
protocol.** It does not change the Phase 4b verdict.

**What it shows.** On the same evaluation images, the detector catches fewer of a type's images when the type
was held out than when it was trained on, for all four types and both budgets as point estimates:
at 5%, 6/9 against 9/9 (bent), 15/28 against 24/28 (melt), 0/7 against 5/7 (missing), 10/13 against 13/13
(scratch). The paired outcomes only go one way: an image caught by the held-out detector but not by the
control is rare (2 images in total, both melt at 5%; none at 10%), while 3, 11, 5 and 3 images (bent, melt,
missing, scratch; 5%) are caught only by the control. AUROC on those images moves the same way, most for
`missing` (0.799 against 0.981) and `melt` (0.886 against 0.978).

**What it adds to Phase 4b.** The open question was whether the low held-out recall is novelty or just
type difficulty. The images are the same in both runs, so difficulty of the images is controlled: the
control detector catches 5/7 `missing` images at 5% and 7/7 at 10%, so `missing` is not simply undetectable
for this detector. Restricting to images that carry only the type (the held-out detector then knows
nothing about anything on the image) gives the same direction (`missing` 0/5 against 3/5 at 5%).

**What it does not show.**
- **Not novelty alone.** The two detectors differ in more than one type. The held-out detector trains on
  fewer anomalous images (23 for melt, 35 missing, 41 scratch, 42 bent, against 47 in the control), and each
  is a single 20-epoch run with one seed, so run-to-run variation is not measured. For `melt` the training
  set is half the size, so the largest and only clearly non-zero difference is also the most confounded one.
  `bent` and `scratch` lose the fewest training images and show differences of 3 images each (p = 0.25).
- **Weak evidence per type.** Exact McNemar p-values at 5% are 0.250 (bent), 0.022 (melt), 0.062 (missing)
  and 0.250 (scratch); at 10%, 1.000, 0.008, 0.062 and 1.000. They are not corrected for 8 tests, and with a
  Bonferroni threshold of 0.05 / 8 = 0.006 none passes. Read them as descriptive.
- **Small samples.** 7 to 28 images per type; `missing` has 7, so even 5 against 0 is a handful of images.
- **Negative control passes.** The anomaly models' scores are bit-identical across the five runs and they
  have no discordant image, so the pairing and the image lists are right (the held-out counts recomputed here
  also match the Phase 4b rows).
- Third read of `threshold_evaluation`, after the 4a frozen thresholds and the 4b runs; nothing was tuned
  from it, and no further read should feed a decision.

**Reading for the claim.** The pattern is what novelty hurting a supervised detector predicts, in a direction
that never reverses, and it weakens the "types just differ in difficulty" alternative. It does not isolate
novelty from training-set size, so Phase 4b's verdict (weakly supports; decided for `missing` only; cannot
decide overall) stands. Separating the two would need a control with matched training-set size, which is a
new run and so a new, disclosed decision.
