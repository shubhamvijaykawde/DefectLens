"""Phase 6c: the generated README tables (defectlens.eval.readme_tables)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from defectlens.eval.readme_tables import (
    BLOCKS,
    ReportError,
    build_blocks,
    counts_only,
    read_table,
    strip_n,
    unmarked_names,
    update_readme,
)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"

DOC = """# Report

## Section A

### Image AUROC

| category | m1 | m2 |
|---|---|---|
| cashew | 0.9 | 0.8 |
| **mean** | 0.5 | 0.4 |

## Section B

### Image AUROC

| category | m1 | m2 |
|---|---|---|
| cashew | 0.1 | 0.2 |

## Section C

Text, no table here.

### Empty

Only a sentence.
"""


def test_read_table_parses_header_and_rows() -> None:
    t = read_table(DOC, "Image AUROC")
    assert t.header == ["category", "m1", "m2"]
    assert t.rows == [["cashew", "0.9", "0.8"], ["**mean**", "0.5", "0.4"]]
    assert t.column("m2") == ["0.8", "0.4"]
    assert t.row("category", "cashew") == ["cashew", "0.9", "0.8"]


def test_under_picks_the_heading_of_the_right_section() -> None:
    assert read_table(DOC, "Image AUROC", under="Section B").rows == [["cashew", "0.1", "0.2"]]


@pytest.mark.parametrize(
    "heading, under",
    [("Nope", None), ("Image AUROC", "Section Z"), ("Empty", None), ("Section C", None)],
)
def test_missing_heading_or_table_raises(heading: str, under: str | None) -> None:
    with pytest.raises(ReportError):
        read_table(DOC, heading, under=under)


def test_unknown_column_or_ambiguous_row_raises() -> None:
    t = read_table(DOC, "Image AUROC")
    with pytest.raises(ReportError):
        t.pick(["nope"])
    with pytest.raises(ReportError):
        t.row("category", "absent")


def test_cell_helpers() -> None:
    assert counts_only("6/9 = 0.67 [0.35, 0.88]") == "6/9"
    assert counts_only("0.250") == "0.250"
    assert strip_n("0.8962 ± 0.0274 (n=5)") == "0.8962 ± 0.0274"


def test_render_round_trips() -> None:
    t = read_table(DOC, "Image AUROC")
    assert read_table("## X\n\n" + t.render() + "\n", "X").rows == t.rows


def test_update_readme_replaces_only_between_markers_and_is_idempotent() -> None:
    readme = "head\n<!-- generated:a -->\nOLD\n<!-- /generated:a -->\ntail\n"
    once = update_readme(readme, {"a": "NEW"})
    assert once == "head\n<!-- generated:a -->\nNEW\n<!-- /generated:a -->\ntail\n"
    assert update_readme(once, {"a": "NEW"}) == once


def test_update_readme_needs_exactly_one_marker_pair_in_order() -> None:
    with pytest.raises(ReportError):
        update_readme("no markers", {"a": "x"})
    with pytest.raises(ReportError):
        update_readme("<!-- /generated:a -->\n<!-- generated:a -->", {"a": "x"})
    twice = "<!-- generated:a -->\n<!-- /generated:a -->\n" * 2
    with pytest.raises(ReportError):
        update_readme(twice, {"a": "x"})


def test_unmarked_names_finds_a_typo() -> None:
    assert unmarked_names("<!-- generated:detectors -->\n<!-- generated:typo -->") == ["typo"]


# ---- the real reports and the committed README ---------------------------------------------


@pytest.fixture(scope="module")
def blocks() -> dict[str, str]:
    return build_blocks(REPORTS)


def test_every_block_is_built_and_is_a_table(blocks: dict[str, str]) -> None:
    assert set(blocks) == set(BLOCKS)
    for name, body in blocks.items():
        assert body.lstrip().startswith("|"), name
        assert "Generated from" in body, name


def numbers(line: str) -> list[str]:
    return re.findall(r"\d[\d,]*\.?\d*", line)


def test_every_number_in_a_generated_table_is_in_a_report(blocks: dict[str, str]) -> None:
    """The tables only select and rearrange: no digit string may appear that no report has."""
    corpus = "\n".join(
        p.read_text() for p in sorted(REPORTS.glob("*")) if p.suffix in {".md", ".json"}
    )
    missing: list[str] = []
    for name, body in blocks.items():
        for line in body.splitlines():
            if not line.startswith("|"):
                continue
            for tok in numbers(line):
                tok = tok.rstrip(".")
                if tok.replace(",", "") not in corpus and tok not in corpus:
                    missing.append(f"{name}: {tok}")
    assert not missing, missing


def test_readme_generated_tables_are_up_to_date(blocks: dict[str, str]) -> None:
    readme = (ROOT / "README.md").read_text()
    assert not unmarked_names(readme)
    assert update_readme(readme, blocks) == readme, (
        "README.md is out of date: run `uv run python scripts/build_report_tables.py`"
    )


def test_detector_table_has_the_four_detectors_from_the_report(blocks: dict[str, str]) -> None:
    rows = [ln for ln in blocks["detectors"].splitlines() if ln.startswith("| ")][1:]
    assert len(rows) == 4
    src = read_table((REPORTS / "detection_comparison.md").read_text(), "Headline metrics")
    test_rows = src.where("split", "test").rows
    assert [r[2] for r in test_rows] == [ln.split("|")[2].strip() for ln in rows]


# ---- leftovers of the rewrite and the licence -----------------------------------------------


def test_readme_has_no_leftover_placeholders() -> None:
    readme = (ROOT / "README.md").read_text()
    for needle in ("@@", "TODO: set once", "<repository URL>", "<!-- generated:TODO"):
        assert needle not in readme, needle


def test_readme_sections_are_unique_and_in_order() -> None:
    readme = (ROOT / "README.md").read_text()
    top = re.findall(r"^## (.+)$", readme, flags=re.M)
    assert len(top) == len(set(top)), top
    assert top == [
        "Results",
        "Architecture",
        "Quickstart",
        "Reproducibility",
        "Limitations",
        "Licence",
        "Dataset terms and citations",
    ]


def test_license_is_mit_for_the_owner_and_readme_limits_its_scope() -> None:
    text = (ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Shubham Vijay Kawde" in text
    readme = (ROOT / "README.md").read_text()
    assert "https://github.com/shubhamvijaykawde/DefectLens" in readme
    assert "actions/workflows/ci.yml" in readme
    licence = readme.split("## Licence")[1].split("## Dataset terms")[0]
    assert "code only" in licence.lower() or "only" in licence
    for word in ("datasets", "weights", "VisA-derived"):
        assert word in licence, word


def test_every_local_image_in_the_readme_exists() -> None:
    readme = (ROOT / "README.md").read_text()
    paths = re.findall(r"!\[[^\]]*\]\((?!https?://)([^)\s]+)\)", readme)
    assert paths
    missing = [p for p in paths if not (ROOT / p).is_file()]
    assert not missing, missing
