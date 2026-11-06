"""Score a COCO-results prediction file against a split, with the shared metric code.

Usage: uv run python scripts/evaluate.py preds=path/to/preds.json split=val [agnostic=true]
Predictions are ``[{"image_id", "category_id", "score", "bbox": [x, y, w, h]}, ...]`` in the
original image frame. Print-only: nothing is logged to MLflow.
"""

from __future__ import annotations

import sys
from pathlib import Path

from defectlens.config import load_config
from defectlens.data.coco import read_coco, read_predictions
from defectlens.eval.confusion import confusion_matrix
from defectlens.eval.detection import GroundTruth, evaluate_detections
from defectlens.eval.report import confusion_table, metrics_table, per_class_table


def main() -> int:
    args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
    if "preds" not in args:
        print(__doc__)
        return 2
    cfg = load_config()
    split = args.get("split", "val")
    agnostic = args.get("agnostic", "false").lower() == "true"
    images, names = read_coco(Path(cfg.data.processed_dir) / f"{split}.json")
    dets = read_predictions(Path(args["preds"]), images)
    gts = [GroundTruth.of(i.boxes, i.labels) for i in images]
    m = evaluate_detections(dets, gts, names, class_agnostic=agnostic)
    print(metrics_table({split: m}))
    if not agnostic:
        order = [names[i] for i in sorted(names)]
        print()
        print(per_class_table(m, order))
        print()
        print(confusion_table(confusion_matrix(dets, gts, len(order)), order))
    return 0


if __name__ == "__main__":
    sys.exit(main())
