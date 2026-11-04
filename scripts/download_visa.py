"""Download and extract VisA (1.93 GB) into data/raw/visa. Idempotent."""

from __future__ import annotations

import sys
from pathlib import Path

from defectlens.data.download import VISA_SHA256, VISA_URL, fetch_verified, safe_extract

ROOT = Path("data/raw/visa")
ARCHIVE = Path("data/raw/_downloads/VisA_20220922.tar")


def main() -> int:
    if (ROOT / "split_csv" / "1cls.csv").is_file():
        print(f"VisA already extracted at {ROOT}")
        return 0
    tar = fetch_verified(VISA_URL, VISA_SHA256, ARCHIVE)
    print(f"verified {tar} (sha256 ok)")
    safe_extract(tar, ROOT)
    print(f"extracted to {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
