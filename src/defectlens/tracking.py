"""MLflow helpers: SQLite backend, and config + git hash + dataset version on every run."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow

DEFAULT_TRACKING_URI = "sqlite:///mlruns/mlflow.db"
DEFAULT_ARTIFACT_ROOT = "mlruns/artifacts"


def git_commit_hash(repo: Path | None = None) -> str:
    """Current commit hash (read-only git), with ``+dirty`` if the tree has changes.

    Returns ``"no-commit"`` when there is no commit yet or git is unavailable.
    """
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=cwd
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "no-commit"
    return f"{sha}+dirty" if dirty else sha


def dataset_version(dvc_file: Path | None) -> str:
    """Dataset version = md5 recorded in a DVC ``.dvc`` file, or ``"unversioned"``."""
    if dvc_file is None or not dvc_file.is_file():
        return "unversioned"
    for line in dvc_file.read_text().splitlines():
        stripped = line.strip().lstrip("- ")
        if stripped.startswith("md5:"):
            return stripped.split(":", 1)[1].strip()
    return "unversioned"


def setup_tracking(
    experiment: str,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
) -> None:
    """Point MLflow at the local SQLite store and select (creating if needed) an experiment."""
    if tracking_uri.startswith("sqlite:///"):
        db_path = Path(tracking_uri.removeprefix("sqlite:///"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(experiment) is None:
        artifact_uri = Path(artifact_root).resolve().as_uri()
        mlflow.create_experiment(experiment, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment)


@contextmanager
def tracked_run(
    experiment: str,
    config: Mapping[str, Any],
    *,
    seed: int,
    dvc_file: Path | None = None,
    run_name: str | None = None,
    tracking_uri: str = DEFAULT_TRACKING_URI,
    artifact_root: str = DEFAULT_ARTIFACT_ROOT,
    repo: Path | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run that always records seed, git hash, dataset version and full config."""
    setup_tracking(experiment, tracking_uri, artifact_root)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "git_commit": git_commit_hash(repo),
                "dataset_version": dataset_version(dvc_file),
            }
        )
        mlflow.log_param("seed", seed)
        mlflow.log_dict(json.loads(json.dumps(config, default=str)), "config.json")
        yield run
