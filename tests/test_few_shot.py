"""Saved few-shot draws and the few-shot runner (tiny CPU fixtures, no weight downloads)."""

import json
from pathlib import Path

import mlflow
import pytest

from defectlens.anomaly import few_shot
from defectlens.anomaly.datamodule import load_split_paths

TRAIN = {
    "a": [f"a/{i:03d}.jpg" for i in range(30)],
    "b": [f"b/{i:03d}.jpg" for i in range(30)],
}


def test_draws_are_deterministic_nested_and_from_train() -> None:
    d1 = few_shot.make_draws(TRAIN, seed=42, n_draws=5, max_shots=10)
    d2 = few_shot.make_draws(TRAIN, seed=42, n_draws=5, max_shots=10)
    assert d1 == d2
    assert len(d1["a"]) == 5 and all(len(d) == 10 for d in d1["a"])
    for category, per_draw in d1.items():
        assert all(len(set(d)) == 10 and set(d) <= set(TRAIN[category]) for d in per_draw)
    one, five, ten = (few_shot.select(d1, "a", 0, k) for k in (1, 5, 10))
    assert one == five[:1] and five == ten[:5]  # nested within a draw
    assert d1["a"][0] != d1["a"][1] and d1["a"][0] != d1["b"][0]  # draws and categories differ
    assert few_shot.make_draws(TRAIN, seed=43, n_draws=5, max_shots=10) != d1


def test_a_categorys_draws_do_not_depend_on_the_others() -> None:
    both = few_shot.make_draws(TRAIN, 42, 3, 10)
    only_b = few_shot.make_draws({"b": TRAIN["b"]}, 42, 3, 10)
    assert both["b"] == only_b["b"]


def test_too_few_training_images_is_an_error() -> None:
    with pytest.raises(ValueError, match="fewer than max_shots"):
        few_shot.make_draws({"a": ["x", "y"]}, 0, 1, 10)


def test_load_or_create_writes_once_then_reuses_and_extends(tmp_path: Path) -> None:
    path = tmp_path / "splits" / "draws.json"
    first = few_shot.load_or_create_draws(path, {"a": TRAIN["a"]}, seed=1, n_draws=2, max_shots=4)
    text = path.read_text()
    assert json.loads(text)["meta"] == {
        "seed": 1, "n_draws": 2, "max_shots": 4, "scheme": "nested prefixes",
    }
    # reading again, even with other arguments, returns the saved draws and does not rewrite
    again = few_shot.load_or_create_draws(path, {"a": TRAIN["a"]}, seed=99, n_draws=5, max_shots=9)
    assert again == first and path.read_text() == text
    # a new category is drawn with the file's own settings; the existing one is untouched
    both = few_shot.load_or_create_draws(path, TRAIN, seed=99, n_draws=5, max_shots=9)
    assert both["a"] == first["a"] and len(both["b"]) == 2 and len(both["b"][0]) == 4
    assert few_shot.load_draws(path)[1] == both


def test_saved_paths_must_be_in_the_train_split(tmp_path: Path) -> None:
    path = tmp_path / "draws.json"
    saved = few_shot.load_or_create_draws(path, {"a": TRAIN["a"]}, 1, n_draws=1, max_shots=3)
    shrunk = [p for p in TRAIN["a"] if p != saved["a"][0][0]]  # a saved image left the split
    with pytest.raises(ValueError, match="not in its train split"):
        few_shot.load_or_create_draws(path, {"a": shrunk}, seed=1)


def test_select_rejects_out_of_range_requests() -> None:
    draws = few_shot.make_draws(TRAIN, 0, 2, 5)
    with pytest.raises(ValueError, match="only 2 are saved"):
        few_shot.select(draws, "a", 2, 1)
    with pytest.raises(ValueError, match="at most 5"):
        few_shot.select(draws, "a", 0, 6)
    with pytest.raises(ValueError, match="at most 5"):
        few_shot.select(draws, "a", 0, 0)
    with pytest.raises(ValueError, match="no saved draws"):
        few_shot.select(draws, "zzz", 0, 1)


def _overrides(tiny: dict[str, Path], tmp_path: Path) -> list[str]:
    return [
        f"data.raw_dir={tiny['root']}",
        f"data.csv={tiny['csv']}",
        f"data.splits_dir={tiny['splits']}",
        f"data.dvc_file={tmp_path / 'missing.dvc'}",
        "model.backbone=resnet18",
        "model.pre_trained=false",
        "model.image_size=[64,64]",
        "model.coreset_ratio=0.5",
        "model.num_neighbors=1",
        "model.projection_dim=16",
        "trainer.device=cpu",
        "trainer.num_workers=0",
        f"mlflow.tracking_uri=sqlite:///{tmp_path}/m.db",
        f"mlflow.artifact_root={tmp_path / 'art'}",
    ]


def test_run_few_shot_uses_saved_draws_logs_tags_and_resumes(
    tiny_visa: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    over = _overrides(tiny_visa, tmp_path)
    done = few_shot.run_few_shot("patchcore", ["cat"], [3], 2, over)
    assert done == ["patchcore_cat_3shot_d0", "patchcore_cat_3shot_d1"]

    # the draws file sits next to the split files and holds only train images
    _, draws = few_shot.load_draws(tiny_visa["splits"] / "few_shot_draws.json")
    train = set(load_split_paths(tiny_visa["splits"], "cat", "train"))
    assert set(draws) == {"cat"} and all(set(d) <= train for d in draws["cat"])

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/m.db")
    exp = mlflow.get_experiment_by_name("defectlens")
    assert exp is not None
    runs = {
        r.info.run_name: r
        for r in mlflow.search_runs([exp.experiment_id], output_format="list")
    }
    assert set(runs) == set(done)
    r = runs["patchcore_cat_3shot_d1"]
    assert r.data.tags["phase"] == "3c" and r.data.tags["shots"] == "3"
    assert r.data.tags["draw"] == "1" and r.data.tags["eval_split"] == "threshold_calibration"
    assert r.data.metrics["n_train"] == 3.0 and r.data.metrics["calib/image_auroc"] >= 0.0
    assert "peak_ram_mib" not in r.data.metrics  # process-wide in a shared process

    # a second call resumes: everything is finished, nothing runs; skip_done=False repeats
    assert few_shot.run_few_shot("patchcore", ["cat"], [3], 2, over) == []
    assert few_shot.run_few_shot("patchcore", ["cat"], [3], 1, over, skip_done=False) == [
        "patchcore_cat_3shot_d0"
    ]


def test_finished_run_names_on_a_new_store(tmp_path: Path) -> None:
    assert few_shot.finished_run_names(f"sqlite:///{tmp_path}/x.db", "defectlens", "3c") == set()


