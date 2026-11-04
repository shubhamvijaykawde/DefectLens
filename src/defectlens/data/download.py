"""Resumable, hash-verified dataset download with safe tar extraction."""

from __future__ import annotations

import hashlib
import http.client
import tarfile
import time
import urllib.request
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
