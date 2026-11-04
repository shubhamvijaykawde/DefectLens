from pathlib import Path

import mlflow

from defectlens.tracking import dataset_version, git_commit_hash, tracked_run


def test_dataset_version_reads_md5(tmp_path: Path) -> None:
    f = tmp_path / "x.dvc"
    f.write_text("outs:\n- md5: abc123.dir\n  size: 1\n  path: x\n")
    assert dataset_version(f) == "abc123.dir"


def test_dataset_version_missing(tmp_path: Path) -> None:
    assert dataset_version(tmp_path / "nope.dvc") == "unversioned"
    assert dataset_version(None) == "unversioned"


def test_git_hash_outside_repo(tmp_path: Path) -> None:
    assert git_commit_hash(tmp_path) == "no-commit"


def test_tracked_run_logs_provenance(tmp_path: Path) -> None:
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    with tracked_run(
        "test-exp",
        {"lr": 0.1, "nested": {"a": 1}},
        seed=42,
        tracking_uri=uri,
        artifact_root=str(tmp_path / "art"),
        repo=tmp_path,
    ) as run:
        mlflow.log_metric("m", 1.0)
    client = mlflow.MlflowClient(uri)
    data = client.get_run(run.info.run_id).data
    assert data.params["seed"] == "42"
    assert data.tags["git_commit"] == "no-commit"
    assert data.tags["dataset_version"] == "unversioned"
    assert data.metrics["m"] == 1.0
    assert client.download_artifacts(run.info.run_id, "config.json")
