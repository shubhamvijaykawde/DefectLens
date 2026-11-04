"""Resumable, hash-verified dataset download with safe tar extraction."""

from __future__ import annotations

import hashlib
import http.client
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

CHUNK = 1 << 20

# Source: anomalib 2.7.0 ``anomalib.data.datamodules.image.visa.DOWNLOAD_INFO`` (installed copy).
VISA_URL = "https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar"
VISA_SHA256 = "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362"


def sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def _fetch_once(url: str, part: Path) -> None:
    """One attempt: append to ``part``, resuming from its current size."""
    start = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-"} if start else {})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 (fixed https URLs)
        if start and resp.status != 206:  # server ignored Range: restart cleanly
            start = 0
        with part.open("ab" if start else "wb") as out:
            while chunk := resp.read(CHUNK):
                out.write(chunk)


def download(url: str, dest: Path, retries: int = 20) -> Path:
    """Download ``url`` to ``dest``, resuming a partial ``.part`` file after dropped connections."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            _fetch_once(url, part)
            break
        except (OSError, http.client.HTTPException) as exc:
            if attempt == retries:
                raise
            print(
                f"download interrupted ({exc!r}); resuming, attempt {attempt}/{retries}", flush=True
            )
            time.sleep(min(2**attempt, 30))
    part.rename(dest)
    return dest


def safe_extract(archive: Path, out_dir: Path) -> None:
    """Extract a tar using the ``data`` filter, which rejects absolute paths, ``..`` and links
    that escape ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(out_dir, filter="data")


def fetch_verified(url: str, sha256: str, dest: Path) -> Path:
    """Download and verify; a corrupt file is deleted so a rerun starts clean."""
    path = download(url, dest)
    actual = sha256_file(path)
    if actual != sha256:
        path.unlink()
        msg = f"SHA-256 mismatch for {url}: expected {sha256}, got {actual}. File removed."
        raise ValueError(msg)
    return path


# NEU-DET: official Google Drive file linked from the NEU surface defect database page
# (https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm, retrieved 2026-11-04).
# The hash is of the file as served on that date (no publisher checksum exists); it detects
# corruption and later changes, not tampering at the source.
NEU_DET_GDRIVE_ID = "1qrdZlaDi272eA79b0uCwwqPrm2Q_WI3k"
NEU_DET_SHA256 = "cf04bb0b05d364f23e05031b191b20b908a0e5eb11466df74ee7fba98ec835d5"


def safe_extract_zip(archive: Path, out_dir: Path) -> None:
    """Extract a zip, refusing absolute paths and ``..`` components."""
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                msg = f"unsafe path in archive: {name}"
                raise ValueError(msg)
        zf.extractall(root)
