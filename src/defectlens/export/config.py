"""Config of Phase 5a: ``configs/export/default.yaml`` plus command-line ``key=value`` overrides."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf

EXPORT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "export" / "default.yaml"
# Shortcuts: `models=dfine_s,efficientad` means `export.models=[dfine_s,efficientad]`, and so on.
SHORTCUTS = {
    "models": "export.models",
    "providers": "benchmark.providers",
    "category": "benchmark.category",
}


def load_export_config(args: Sequence[str] = (), path: Path = EXPORT_CONFIG) -> DictConfig:
    """The export config with overrides applied (later wins).

    A value is read as a list when it has a comma or when the key is a list in the file, so
    ``models=dfine_s`` and ``models=dfine_s,efficientad`` both work.
    """
    base = OmegaConf.load(path)
    dotlist = []
    for a in args:
        key, sep, value = a.partition("=")
        if not sep:
            msg = f"expected key=value, got {a!r}"
            raise ValueError(msg)
        key = SHORTCUTS.get(key, key)
        is_list = isinstance(OmegaConf.select(base, key), ListConfig)
        as_list = ("," in value or is_list) and not value.startswith("[")
        dotlist.append(f"{key}={'[' + value + ']' if as_list else value}")
    merged = OmegaConf.merge(base, OmegaConf.from_dotlist(dotlist))
    assert isinstance(merged, DictConfig)
    return merged
