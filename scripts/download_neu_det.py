"""Download and extract NEU-DET into data/raw/neu_det. Idempotent.

If the Google Drive download fails (quota, network), download the file manually from
https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm, save it as
data/raw/_downloads/NEU-DET.zip and rerun this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

from defectlens.data.download import (
    NEU_DET_GDRIVE_ID,
    NEU_DET_SHA256,
    safe_extract_zip,
    sha256_file,
)

ROOT = Path("data/raw/neu_det")
ARCHIVE = Path("data/raw/_downloads/NEU-DET.zip")


def main() -> int:
    if (ROOT / "NEU-DET" / "ANNOTATIONS").is_dir():
        print(f"NEU-DET already extracted at {ROOT}")
        return 0
    if not ARCHIVE.exists():
        import gdown

        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        got = gdown.download(id=NEU_DET_GDRIVE_ID, output=str(ARCHIVE), quiet=False)
        if got is None:
            print("gdown failed; download manually, see this script's docstring", file=sys.stderr)
            return 1
    actual = sha256_file(ARCHIVE)
    if actual != NEU_DET_SHA256:
        print(f"SHA-256 mismatch: expected {NEU_DET_SHA256}, got {actual}", file=sys.stderr)
        return 1
    safe_extract_zip(ARCHIVE, ROOT)
    print(f"verified and extracted to {ROOT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
