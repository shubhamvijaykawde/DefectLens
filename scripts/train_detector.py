"""Train a detector. Usage: uv run python scripts/train_detector.py model=faster_rcnn [overrides]

Smoke test (about a minute): ... model=faster_rcnn trainer.fast_dev_run=true
"""

from __future__ import annotations

import sys

from defectlens.config import load_config
from defectlens.detection.trainer import run_training


def main() -> int:
    run_training(load_config(sys.argv[1:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
