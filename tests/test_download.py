import hashlib
import io
import tarfile
from pathlib import Path

import pytest

from defectlens.data.download import (
    download,
    fetch_verified,
    safe_extract,
    safe_extract_zip,
    sha256_file,
)


def test_sha256_file(tmp_path: Path) -> None:
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello")
    assert sha256_file(f) == hashlib.sha256(b"hello").hexdigest()


def test_download_and_verify_file_url(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    url = src.as_uri()
    dest = tmp_path / "out" / "got.bin"
    good = hashlib.sha256(b"payload").hexdigest()
    assert fetch_verified(url, good, dest).read_bytes() == b"payload"


def test_hash_mismatch_removes_file(tmp_path: Path) -> None:
    src = tmp_path / "src.bin"
    src.write_bytes(b"payload")
    dest = tmp_path / "got.bin"
    with pytest.raises(ValueError, match="mismatch"):
        fetch_verified(src.as_uri(), "0" * 64, dest)
    assert not dest.exists()


def test_download_skips_existing(tmp_path: Path) -> None:
    dest = tmp_path / "x.bin"
    dest.write_bytes(b"keep")
    assert download("file:///does/not/exist", dest).read_bytes() == b"keep"


def test_safe_extract_blocks_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("../escaped.txt")
        data = b"x"
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    with pytest.raises(tarfile.TarError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_ok(tmp_path: Path) -> None:
    archive = tmp_path / "ok.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo("d/f.txt")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"ok"))
    safe_extract(archive, tmp_path / "out")
    assert (tmp_path / "out" / "d" / "f.txt").read_text() == "ok"


def test_download_resumes_after_interruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import defectlens.data.download as dl

    src = tmp_path / "src.bin"
    src.write_bytes(b"0123456789")
    calls = {"n": 0}

    def flaky(url: str, part: Path) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            part.write_bytes(b"01234")  # partial data, then the connection drops
            raise BrokenPipeError
        part.write_bytes(src.read_bytes())

    monkeypatch.setattr(dl, "_fetch_once", flaky)
    monkeypatch.setattr(dl.time, "sleep", lambda _s: None)
    out = dl.download(src.as_uri(), tmp_path / "got.bin")
    assert out.read_bytes() == b"0123456789"
    assert calls["n"] == 2


def test_safe_extract_zip_rejects_traversal(tmp_path: Path) -> None:
    import zipfile

    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("../evil.txt", "x")
    with pytest.raises(ValueError, match="unsafe"):
        safe_extract_zip(z, tmp_path / "out")
    ok = tmp_path / "ok.zip"
    with zipfile.ZipFile(ok, "w") as zf:
        zf.writestr("a/b.txt", "x")
    safe_extract_zip(ok, tmp_path / "out2")
    assert (tmp_path / "out2" / "a" / "b.txt").read_text() == "x"
