"""Build NEU-DET COCO files and the fixed splits for NEU-DET and VisA. Deterministic."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from defectlens.data.coco import subset_coco, to_coco, write_coco
from defectlens.data.neu_det import (
    drop_duplicate_boxes,
    image_class,
    load_records,
    validate_records,
)
from defectlens.data.splits import save_splits, stratified_split
from defectlens.data.visa_splits import make_visa_splits, read_one_cls


def prepare_neu_det(cfg: DictConfig) -> int:
    raw = Path(cfg.raw_dir)
    records, dropped = drop_duplicate_boxes(load_records(raw))
    for note in dropped:
        print("  CLEANED:", note)
    problems = validate_records(records, raw / "IMAGES")
    print(f"NEU-DET: {len(records)} images, {sum(len(r.boxes) for r in records)} boxes")
    for p in problems:
        print("  PROBLEM:", p)
    if problems:
        return 1
    strata = [(r.image_id, image_class(r)) for r in records]
    parts = stratified_split(strata, dict(cfg.fractions), cfg.seed)
    digest = save_splits(
        parts,
        {"dataset": "neu_det", "seed": cfg.seed, "fractions": dict(cfg.fractions)},
        Path(cfg.splits_file),
    )
    coco = to_coco(records)
    out = Path(cfg.processed_dir)
    write_coco(coco, out / "all.json")
    for name, ids in parts.items():
        write_coco(subset_coco(coco, set(ids)), out / f"{name}.json")
    cls = {r.image_id: c for r, (_, c) in zip(records, strata, strict=True)}
    for name, ids in parts.items():
        print(f"  {name}: {len(ids)} images {dict(sorted(Counter(cls[i] for i in ids).items()))}")
    print(f"  split_hash {digest}")
    return 0


def prepare_visa(cfg: DictConfig) -> int:
    for cat in cfg.categories:
        rows = read_one_cls(Path(cfg.csv), cat)
        parts, labels = make_visa_splits(rows, cfg.seed)
        missing = [p for ps in parts.values() for p in ps if not (Path(cfg.raw_dir) / p).is_file()]
        if missing:
            print(f"VisA {cat}: {len(missing)} image files missing, e.g. {missing[0]}")
            return 1
        meta = {"dataset": "visa", "category": cat, "seed": cfg.seed, "source": "1cls.csv"}
        digest = save_splits(parts, meta, Path(cfg.splits_dir) / f"visa_{cat}.json")
        counts = {n: dict(Counter(labels[i] for i in ids)) for n, ids in parts.items()}
        print(f"VisA {cat}: {counts} split_hash {digest[:12]}")
    return 0


def main() -> int:
    neu = OmegaConf.load("configs/data/neu_det.yaml")
    visa = OmegaConf.load("configs/data/visa.yaml")
    assert isinstance(neu, DictConfig) and isinstance(visa, DictConfig)
    return prepare_neu_det(neu) or prepare_visa(visa)


if __name__ == "__main__":
    sys.exit(main())
