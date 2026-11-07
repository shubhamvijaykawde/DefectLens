Hand-written after looking at the figures and tables above. They are observations on a few
images; none of the explanations below was tested, and none fed back into a threshold.

**VisA**
- At fn_fp_ratio 10 the frozen thresholds miss 10 of the 600 anomalous-image decisions (12
  model-category cells x 50 images) but flag between 6 of 25 (AnomalyDINO, cashew) and 50 of 50
  (AnomalyDINO, macaroni2) normal images. The errors are overwhelmingly false alarms.
- **macaroni2** is the extreme case: the thresholds flag 45-50 of 50 normal images, i.e. nearly
  everything. In the EfficientAD figure the six lowest-scoring anomalous images (scores 0.076-0.103)
  are all caught, but their defects are small bright spots, and I could not see a defect in the six
  highest-scoring normal images (0.20-0.36, far above the 0.056 threshold). A threshold cannot help
  when the score ranking itself is weak.
- **Small defects.** Missed anomalies are few, so this is weak evidence. For candle the missed
  images have a smaller median defect area than the caught ones (0.09% against 0.16% for PatchCore
  and AnomalyDINO, n = 2; EfficientAD's single miss 0.00%, i.e. about a pixel after resizing), but
  for pcb1 PatchCore's single miss is not smaller (0.26% against 0.21%), so "small defects are
  missed" does not hold across categories.
- **False alarms in the candle / PatchCore figure** look like large, diffuse regions of the map
  across the candle glare and the background corners, not the compact spots seen on the real
  defects. Six images, not quantified.
- **pcb1 / EfficientAD:** the highest-scoring normal image (0.838, threshold 0.137) is a board
  rotated with the pins on top, and one more of the six shown is rotated too; the maps light up
  board edges and pins. Whether orientation drives these false alarms was not checked.

**NEU-DET** (D-FINE-S is drawn; the table below covers all three detectors at ratio 10)
- **Most misses are overlapped by a same-class box that did not reach IoU 0.5.** Of the 628 test
  boxes, Faster R-CNN misses 84 (74 have a kept same-class box with IoU >= 0.1), the matched
  Faster R-CNN 79 (73) and D-FINE-S 103 (75). So the dominant failure is box extent, not class
  confusion, in line with the near-empty off-diagonal of the Phase 2 confusion matrices.
- **Crazing and rolled-in scale** carry it: for D-FINE-S 45 of 99 crazing boxes are missed (32 with
  an overlapping same-class box) and 160 kept rolled-in-scale boxes match nothing; Faster R-CNN
  misses 29 and 20 of them and has 181 unmatched crazing boxes. In the figure these are large,
  diffuse regions where the ground truth is often several overlapping boxes and the detector draws
  boxes of a different extent (7 of the 12 worst D-FINE-S images are crazing). The annotation
  extent of such texture defects is itself ambiguous; I did not check how consistent the labels are.
- **Scratches are almost never missed** (1, 0 and 2 of 76 boxes) and patches rarely (9-11 of 136).
- **Thin inclusions** (inclusion_130, inclusion_125) are mixed: several narrow vertical boxes are
  matched at 0.7-0.8, while other narrow ground-truth boxes next to them are missed, and
  inclusion_125 also has several unmatched inclusion boxes at 0.38-0.41 inside the cluster.
