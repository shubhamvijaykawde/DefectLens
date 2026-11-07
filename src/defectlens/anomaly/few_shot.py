"""Fixed few-shot draws of VisA training images, saved to disk and reused.

For every category and draw ``d`` one seeded permutation of the sorted ``train`` paths is stored
(only its first ``max_shots`` entries); the k-shot set of draw ``d`` is the first ``k`` of them, so
within a draw the 1-shot image is part of the 5-shot set, which is part of the 10-shot set. Draws
differ between categories and between ``d``; a category's draws depend only on the seed and its
name, so adding a category to the file later never changes the existing ones. Only ``train``
(normal) paths are ever drawn.

The file is created once and then only read: asking for more draws or shots than it holds is an
error rather than a silent re-draw, because re-drawing would change every number that used it.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

DEFAULT_DRAWS = 5
DEFAULT_MAX_SHOTS = 10


def make_draws(
    train: Mapping[str, Sequence[str]], seed: int, n_draws: int, max_shots: int
) -> dict[str, list[list[str]]]:
    """``{category: [first max_shots paths of the permutation for each draw]}``."""
    out: dict[str, list[list[str]]] = {}
    for category in sorted(train):
        ordered = sorted(train[category])
        if len(ordered) < max_shots:
            msg = f"{category}: {len(ordered)} training images, fewer than max_shots={max_shots}"
            raise ValueError(msg)
        draws = []
        for d in range(n_draws):
            rng = np.random.default_rng([seed, zlib.crc32(category.encode()), d])
            draws.append([ordered[i] for i in rng.permutation(len(ordered))[:max_shots]])
        out[category] = draws
    return out


def save_draws(
    path: Path, draws: Mapping[str, list[list[str]]], seed: int, n_draws: int, max_shots: int
) -> None:
    meta = {"seed": seed, "n_draws": n_draws, "max_shots": max_shots, "scheme": "nested prefixes"}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta, "draws": draws}, indent=1, sort_keys=True) + "\n")


def load_draws(path: Path) -> tuple[dict[str, int], dict[str, list[list[str]]]]:
    """``(meta, draws)`` from a file written by ``save_draws``."""
    data = json.loads(path.read_text())
    return dict(data["meta"]), {k: [list(d) for d in v] for k, v in data["draws"].items()}


def load_or_create_draws(
    path: Path,
    train: Mapping[str, Sequence[str]],
    seed: int,
    n_draws: int = DEFAULT_DRAWS,
    max_shots: int = DEFAULT_MAX_SHOTS,
) -> dict[str, list[list[str]]]:
    """Saved draws for every category in ``train``: read from ``path``, created if missing.

    Existing categories are only checked (their paths must be in ``train``), never re-drawn;
    categories absent from the file are drawn with the file's own ``n_draws``/``max_shots`` and
    appended.
    """
    if path.is_file():
        meta, draws = load_draws(path)
    else:
        meta, draws = {"seed": seed, "n_draws": n_draws, "max_shots": max_shots}, {}
    for category, per_draw in draws.items():
        known = set(train.get(category, ()))
        stray = [p for d in per_draw for p in d if p not in known]
        if stray:
            msg = (
                f"{path}: {len(stray)} saved paths of {category} are not in its train split, "
                f"e.g. {stray[0]}"
            )
            raise ValueError(msg)
    missing = {c: train[c] for c in train if c not in draws}
    if missing:
        draws.update(make_draws(missing, meta["seed"], meta["n_draws"], meta["max_shots"]))
        save_draws(path, draws, meta["seed"], meta["n_draws"], meta["max_shots"])
    return draws


def select(
    draws: Mapping[str, list[list[str]]], category: str, draw: int, shots: int
) -> list[str]:
    """The ``shots`` training paths of draw ``draw`` for ``category``."""
    if category not in draws:
        msg = f"no saved draws for category {category!r} (have {sorted(draws)})"
        raise ValueError(msg)
    per_draw = draws[category]
    if not 0 <= draw < len(per_draw):
        msg = (
            f"draw {draw} requested but only {len(per_draw)} are saved: ask for fewer draws "
            "(deleting the file would re-draw and change every result that used it)"
        )
        raise ValueError(msg)
    saved = per_draw[draw]
    if not 1 <= shots <= len(saved):
        msg = f"{shots} shots requested but the saved draws hold at most {len(saved)}"
        raise ValueError(msg)
    return saved[:shots]


def finished_run_names(tracking_uri: str, experiment: str, phase: str) -> set[str]:
    """Names of FINISHED MLflow runs of ``phase`` (empty when the store or experiment is new)."""
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return set()
    runs = mlflow.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.phase = '{phase}' and attributes.status = 'FINISHED'",
        output_format="list",
    )
    return {r.info.run_name for r in runs}


def run_few_shot(
    model: str,
    categories: Sequence[str],
    shots: Sequence[int],
    n_draws: int,
    overrides: Sequence[str] = (),
    skip_done: bool = True,
) -> list[str]:
    """Fit ``model`` on every (category, draw, shots) of the saved draws; one MLflow run each.

    Scored on ``threshold_calibration`` like every Phase 3 run, tagged ``phase=3c``. Runs whose
    name is already FINISHED in MLflow are skipped when ``skip_done`` (resume after an
    interruption).
    Returns the run names that were executed. ``peak_ram_mib`` is not logged: all fits share this
    process, so the figure would not belong to any one of them.
    """
    from defectlens.anomaly.datamodule import load_split_paths
    from defectlens.anomaly.runner import run_anomaly
    from defectlens.config import load_config

    base = ["data=visa", f"model={model}", "+phase=3c", *overrides]
    cfg0 = load_config(base)
    splits = Path(cfg0.data.splits_dir)
    train = {c: load_split_paths(splits, c, "train") for c in categories}
    draws = load_or_create_draws(splits / "few_shot_draws.json", train, int(cfg0.seed))
    done = (
        finished_run_names(cfg0.mlflow.tracking_uri, cfg0.experiment, "3c") if skip_done else set()
    )
    executed = []
    for category in categories:
        for d in range(n_draws):
            for k in shots:
                cfg = load_config([*base, f"model.shots={k}", f"+draw={d}", f"+run_suffix=_d{d}"])
                name = f"{model}_{category}_{k}shot_d{d}"
                paths = select(draws, category, d, k)  # also validates draws and shots
                if name in done:
                    print(f"skip {name} (already finished)", flush=True)
                    continue
                run_anomaly(cfg, category, train_paths=paths, log_ram=False)
                executed.append(name)
    return executed
