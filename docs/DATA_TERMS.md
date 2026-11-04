# Dataset terms and provenance

This records what each source says, not a legal opinion. Retrieved **2026-11-04** unless noted.
Datasets are **not redistributed**: raw data is downloaded from the official source, versioned with
a local DVC remote, and never committed to git. The README links to the official sources.

## NEU-DET (steel surface defects, Pascal VOC boxes)

- **Source page:** <https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm>
  (NEU surface defect database, Kechen Song).
- **Download used:** the Google Drive file linked on that page for NEU-DET
  (id `1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k`; a Baidu-Pan mirror is also listed there). 27,607,059 bytes,
  SHA-256 `cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5` (computed by us; the
  publisher gives no checksum, so this detects corruption or later change, not source tampering).
- **Usage wording on the page:** "need to download the image-with-annotations file, called NEU-DET"
  (for detection work) and "We would appreciate it if you cite our works when using the database."
- **License:** the page states **none**. A third-party dataset card also says none. Unresolved.
- **Citations the page lists:**
  1. Y. Bao, K. Song, J. Liu, Y. Wang, Y. Yan, H. Yu, X. Li, "Triplet-Graph Reasoning Network for
     Few-shot Metal Generic Surface Defect Segmentation," IEEE TIM, 2021.
  2. K. Song, Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled
     steel strip surface defects," Applied Surface Science 285, 858-864, 2013.
  3. Y. He, K. Song, Q. Meng, Y. Yan, "An End-to-end Steel Surface Defect Detection Approach via
     Fusing Multiple Hierarchical Features," IEEE TIM 69(4), 1493-1504, 2020.
- **Open questions:** whether redistribution, commercial use, or publishing trained weights is
  allowed. A third-party card says the data is hosted on IEEE DataPort (subscription); this conflicts
  with the official page, which we treated as primary.

### Measured properties and cleaning (from the downloaded files)

- 1,800 images (200x200, 1 channel) and 1,800 XML files; 4,189 raw boxes over 6 classes.
- 3 exact duplicate boxes in the raw annotations (`crazing_120`, `inclusion_62`, `patches_198`)
  are dropped by `drop_duplicate_boxes`, leaving **4,186** boxes. Nothing else is altered.
- The XML `<filename>` is missing its extension for `patches_*` and `pitted_surface_*`; images are
  located by annotation stem + `.jpg`.
- 123 images contain boxes of more than one class. Splits are stratified by the class in the file
  name (300 images per class), which is always among the image's box labels.
- There are no defect-free images, so image-level false-alarm rates on good steel are unmeasurable.

## VisA (Visual Anomaly)

- **Source:** AWS Open Data registry <https://registry.opendata.aws/visa>, object
  `s3://amazon-visual-anomaly/VisA_20220922.tar` (us-west-2). Archive SHA-256 verified against
  anomalib 2.7.0's download info (see `src/defectlens/data/download.py`).
- **License:** the registry page lists <https://creativecommons.org/licenses/by/4.0/> (CC BY 4.0),
  and the extracted archive's `LICENSE-DATASET` file is the CC BY 4.0 legal text. Earlier in this
  project, other sources were found that say CC BY-NC-SA 4.0; those were not re-read, so the conflict
  is **not fully resolved**. Attribution is required either way.
- **Requested citation:** "Visual Anomaly (VisA) was accessed on `DATE` from
  https://registry.opendata.aws/visa." (access date for this project: 2026-11-04) and Y. Zou,
  J. Jeong, L. Pemula, D. Zhang, O. Dabeer, "SPot-the-Difference Self-Supervised Pre-training for
  Anomaly Detection and Segmentation," arXiv:2207.14315, 2022.

## Not yet checked

- **MVTec AD** (if used) and **pretrained checkpoints** (D-FINE, DINOv2, WideResNet50): terms are
  recorded here when first used.
- **Trained weights:** no license or usage status is assumed. The README will say the terms are
  unresolved and that the repo's code license does not cover them. Publishing any weights is the
  repo owner's decision, after the terms above are checked.
