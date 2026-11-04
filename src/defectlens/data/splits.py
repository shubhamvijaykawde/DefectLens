"""Deterministic stratified splits, saved to JSON with a content hash."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def stratified_split(
    items: Sequence[tuple[str, str]],
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, list[str]]:
    """Split ``(item_id, stratum)`` pairs so every stratum is divided by ``fractions``.

    Items are sorted before shuffling with a local ``random.Random(seed)``, so the result
    depends only on the ids, strata, fractions and seed. Per stratum, every part but the
    last gets ``floor(n * fraction + 0.5)`` items and the last takes the remainder.
    """
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        msg = f"fractions must sum to 1, got {sum(fractions.values())}"
        raise ValueError(msg)
    ids = [i for i, _ in items]
    if len(set(ids)) != len(ids):
        msg = "duplicate item ids"
        raise ValueError(msg)
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for item_id, stratum in items:
        by_stratum[stratum].append(item_id)
    rng = random.Random(seed)  # noqa: S311 (not cryptographic)
    out: dict[str, list[str]] = {name: [] for name in fractions}
    names = list(fractions)
    for stratum in sorted(by_stratum):
        group = sorted(by_stratum[stratum])
        rng.shuffle(group)
        start = 0
        for k, name in enumerate(names):
            end = (
                len(group)
                if k == len(names) - 1
                else start + math.floor(len(group) * fractions[name] + 0.5)
            )
            out[name].extend(group[start:end])
            start = end
    return {name: sorted(v) for name, v in out.items()}


def split_hash(parts: Mapping[str, Sequence[str]]) -> str:
    """SHA-256 over the canonical JSON of the id lists; log it with every run."""
    canon = json.dumps({k: sorted(v) for k, v in sorted(parts.items())}, separators=(",", ":"))
    return hashlib.sha256(canon.encode()).hexdigest()


def assert_disjoint(parts: Mapping[str, Sequence[str]]) -> None:
    """Raise if any id appears twice, within or across the named subsets."""
    seen: dict[str, str] = {}
    for name, ids in parts.items():
        for i in ids:
            if i in seen:
                msg = f"{i!r} is in both {seen[i]!r} and {name!r}"
                raise ValueError(msg)
            seen[i] = name


def save_splits(parts: Mapping[str, Sequence[str]], meta: Mapping[str, Any], path: Path) -> str:
    """Write ``{"meta": ..., <subset>: [...]}`` deterministically; returns the split hash."""
    assert_disjoint(parts)
    digest = split_hash(parts)
    doc = {"meta": {**meta, "split_hash": digest}, **{k: sorted(v) for k, v in parts.items()}}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return digest


def load_splits(path: Path) -> dict[str, Any]:
    """Load a splits file and verify its stored hash against its contents."""
    doc: dict[str, Any] = json.loads(path.read_text())
    parts = {k: v for k, v in doc.items() if k != "meta"}
    if split_hash(parts) != doc["meta"]["split_hash"]:
        msg = f"{path}: split hash mismatch (file edited by hand?)"
        raise ValueError(msg)
    return doc
