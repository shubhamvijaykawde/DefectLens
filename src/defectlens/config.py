"""Compose the Hydra config without changing the working directory or creating output dirs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def load_config(overrides: Sequence[str] = (), config_dir: Path = CONFIG_DIR) -> DictConfig:
    """``configs/config.yaml`` plus Hydra overrides, e.g. ``["model=classical", "seed=1"]``."""
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="config", overrides=list(overrides))
