"""Phase 6c: the result tables of README.md, generated from ``reports/``.

Nothing here computes a metric. Every cell is a string copied from a table that an earlier script
wrote into ``reports/*.md`` (itself generated from MLflow runs or result files) or a field of
``reports/docker_smoke.json``; the only transformations are choosing rows and columns, renaming
headers, pivoting a long table into a wide one and cutting an interval off a ``k/n = p [lo, hi]``
cell. A table that is missing, or a row or column that is not there, raises instead of printing a
gap.

README.md holds each generated table between two marker lines::

    <!-- generated:detectors -->
    ...
    <!-- /generated:detectors -->

``scripts/build_report_tables.py`` rewrites what is between the markers; ``--check`` only compares.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path


class ReportError(ValueError):
    """A report table the README needs is missing or does not have the expected shape."""


# ---- reading a table out of a markdown report ---------------------------------------------------


@dataclass(frozen=True)
class Table:
    header: list[str]
    rows: list[list[str]]

    def index(self, column: str) -> int:
        if column not in self.header:
            raise ReportError(f"no column {column!r} in {self.header}")
        return self.header.index(column)

    def column(self, column: str) -> list[str]:
        i = self.index(column)
        return [r[i] for r in self.rows]

    def row(self, column: str, value: str) -> list[str]:
        """The one row whose ``column`` equals ``value``."""
        i = self.index(column)
        hits = [r for r in self.rows if r[i] == value]
        if len(hits) != 1:
            raise ReportError(f"{len(hits)} rows with {column} == {value!r}, wanted exactly 1")
        return hits[0]

    def where(self, column: str, value: str) -> Table:
        i = self.index(column)
        return Table(self.header, [r for r in self.rows if r[i] == value])

    def pick(self, columns: Sequence[str]) -> Table:
        idx = [self.index(c) for c in columns]
        return Table(list(columns), [[r[i] for i in idx] for r in self.rows])

    def renamed(self, mapping: dict[str, str]) -> Table:
        return Table([mapping.get(h, h) for h in self.header], self.rows)

    def map_cells(self, fn: Callable[[str], str]) -> Table:
        return Table(self.header, [[fn(c) for c in r] for r in self.rows])

    def render(self) -> str:
        lines = ["| " + " | ".join(self.header) + " |", "|" + "---|" * len(self.header)]
        lines += ["| " + " | ".join(r) + " |" for r in self.rows]
        return "\n".join(lines)


def _split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_heading(line: str) -> bool:
    return line.startswith("#")


def _heading_text(line: str) -> str:
    return line.lstrip("#").strip()


def read_table(text: str, heading: str, *, under: str | None = None) -> Table:
    """The first table below the heading ``heading`` (exact text, any level).

    ``under`` names an earlier heading: the search for ``heading`` starts after it. This tells apart
    two sections of one report that reuse a heading ("Image AUROC" in the three-model and the
    few-shot part). The table must come before the next heading of any level.
    """
    lines = text.splitlines()
    start = 0
    if under is not None:
        hits = [i for i, ln in enumerate(lines) if _is_heading(ln) and _heading_text(ln) == under]
        if not hits:
            raise ReportError(f"heading {under!r} not found")
        start = hits[0] + 1
    at = [
        i
        for i in range(start, len(lines))
        if _is_heading(lines[i]) and _heading_text(lines[i]) == heading
    ]
    if not at:
        raise ReportError(f"heading {heading!r} not found" + (f" under {under!r}" if under else ""))
    block: list[str] = []
    for ln in lines[at[0] + 1 :]:
        if _is_heading(ln):
            break
        if ln.startswith("|"):
            block.append(ln)
        elif block:
            break
    if len(block) < 3:
        raise ReportError(f"no table below heading {heading!r}")
    header = _split_row(block[0])
    rows = [_split_row(ln) for ln in block[2:]]
    for r in rows:
        if len(r) != len(header):
            raise ReportError(f"row {r} does not have {len(header)} cells (heading {heading!r})")
    return Table(header, rows)


# ---- small cell transformations -----------------------------------------------------------------

_COUNT = re.compile(r"^(\d+/\d+) = [0-9.]+ \[[0-9., ]+\]$")


def counts_only(cell: str) -> str:
    """``6/9 = 0.67 [0.35, 0.88]`` -> ``6/9``; other cells stay as they are."""
    m = _COUNT.match(cell)
    return m.group(1) if m else cell


def strip_bold(cell: str) -> str:
    return cell.replace("**", "")


def strip_n(cell: str) -> str:
    """``0.8962 ± 0.0274 (n=5)`` -> ``0.8962 ± 0.0274``."""
    return re.sub(r" \(n=\d+\)$", "", cell)


NAMES = {
    "classical": "classical (class-agnostic)",
    "faster_rcnn": "Faster R-CNN",
    "faster_rcnn_aug": "Faster R-CNN + scale/shift",
    "dfine_s": "D-FINE-S",
    "patchcore": "PatchCore",
    "anomaly_dino": "AnomalyDINO",
    "efficientad": "EfficientAD",
}


def nice(cell: str) -> str:
    return NAMES.get(cell, cell)


def source_line(*paths: str) -> str:
    srcs = ", ".join(f"`{p}`" for p in paths)
    return (
        f"*Generated from {srcs} by `scripts/build_report_tables.py`; "
        "edit the reports, not this table.*"
    )


def _block(table: Table, *sources: str, note: str = "") -> str:
    out = table.render() + "\n\n" + source_line(*sources)
    return out + ("\n\n" + note if note else "")


# ---- the blocks ----------------------------------------------------------------------------------


def detectors(reports: Path) -> str:
    """NEU-DET test split, one row per detector."""
    src = "reports/detection_comparison.md"
    t = read_table((reports / "detection_comparison.md").read_text(), "Headline metrics")
    t = t.where("split", "test")
    t = t.pick(["detector", "AP@50", "AP@50:95", "recall@0.5", "precision@0.5", "FP/img@0.5"])
    t = Table(t.header, [[nice(r[0]), *r[1:]] for r in t.rows])
    t = t.renamed({"detector": "detector (test)"})
    return _block(t, src)


def anomaly_models(reports: Path) -> str:
    """Image AUROC per category plus the three means, full training data, calibration half."""
    src = "reports/anomaly_comparison.md"
    text = (reports / "anomaly_comparison.md").read_text()
    sec = "Three models, full training data"
    image = read_table(text, "Image AUROC", under=sec).map_cells(strip_bold)
    pixel = read_table(text, "Pixel AUROC", under=sec).map_cells(strip_bold)
    pro = read_table(text, "AUPRO", under=sec).map_cells(strip_bold)
    if not (image.header == pixel.header == pro.header):
        raise ReportError("the three anomaly tables have different columns")
    body = [r for r in image.rows if r[0] != "mean"]
    means = [
        ["**mean image AUROC**", *image.row("category", "mean")[1:]],
        ["mean pixel AUROC", *pixel.row("category", "mean")[1:]],
        ["mean AUPRO", *pro.row("category", "mean")[1:]],
    ]
    table = Table(["image AUROC"] + image.header[1:], body + means)
    return _block(table, src)


def few_shot(reports: Path) -> str:
    """AnomalyDINO image AUROC with 1, 5 and 10 normal images, mean ± std over 5 draws."""
    src = "reports/anomaly_comparison.md"
    t = read_table(
        (reports / "anomaly_comparison.md").read_text(), "Image AUROC", under="Few-shot AnomalyDINO"
    )
    t = t.map_cells(strip_n).map_cells(strip_bold)
    t = t.renamed({"category": "AnomalyDINO, image AUROC"})
    note = "Each cell: mean ± sample standard deviation over 5 saved draws (n = 5)."
    return _block(t, src, note=note)


def _pivot_costs(
    long: Table, name_col: str, ratio_col: str, value_col: str
) -> tuple[list[str], dict[str, dict[str, str]]]:
    ratios: list[str] = []
    cells: dict[str, dict[str, str]] = {}
    for r in long.rows:
        name = r[long.index(name_col)]
        ratio = r[long.index(ratio_col)]
        value = r[long.index(value_col)]
        if ratio not in ratios:
            ratios.append(ratio)
        cells.setdefault(name, {})[ratio] = value
    return ratios, cells


def cost_thresholds(reports: Path) -> str:
    """Evaluation cost with the frozen thresholds, VisA (four categories summed) and NEU-DET."""
    src = "reports/thresholds.md"
    text = (reports / "thresholds.md").read_text()

    visa = read_table(text, "VisA, total over the four categories")
    ratios, cells = _pivot_costs(visa, "model", "fn_fp_ratio", "eval cost")
    trivial = {r[visa.index("fn_fp_ratio")]: r[visa.index("trivial (eval)")] for r in visa.rows}
    rows = [[nice(m), *[cells[m][x] for x in ratios]] for m in cells]
    rows.append(["best constant policy", *[trivial[x] for x in ratios]])
    visa_t = Table(["VisA, evaluation cost (4 categories)", *[f"ratio {x}" for x in ratios]], rows)

    neu_cells: dict[str, dict[str, str]] = {}
    neu_triv: dict[str, str] = {}
    neu_ratios: list[str] = []
    for ratio in ("5", "10", "50"):
        t = read_table(text, f"NEU-DET, fn_fp_ratio = {ratio}")
        neu_ratios.append(ratio)
        for r in t.rows:
            neu_cells.setdefault(r[t.index("detector")], {})[ratio] = r[t.index("eval cost")]
            neu_triv[ratio] = r[t.index("trivial")]
    neu_rows = [[nice(d), *[neu_cells[d][x] for x in neu_ratios]] for d in neu_cells]
    neu_rows.append(["best constant policy", *[neu_triv[x] for x in neu_ratios]])
    neu_t = Table(["NEU-DET, test cost", *[f"ratio {x}" for x in neu_ratios]], neu_rows)
    return (
        visa_t.render()
        + "\n\n"
        + neu_t.render()
        + "\n\n"
        + source_line(src)
        + "\n\nCost = ratio x missed + false alarms; the ratios 5, 10 and 50 are assumptions. "
        "Thresholds were frozen on the selection splits and evaluated once."
    )


def unseen_types(reports: Path) -> str:
    """Leave-one-defect-type-out on VisA pcb1: images caught / images, 5% false-alarm budget."""
    src1, src2 = "reports/unseen_defects.md", "reports/unseen_followup.md"
    t = read_table(
        (reports / "unseen_defects.md").read_text(),
        "Recall on the held-out type, false-alarm budget 5% on calibration normals",
    ).map_cells(counts_only)
    t = t.renamed({n: nice(n) for n in t.header})
    t = t.renamed({"held-out type": "held-out type (never seen by the detector)"})
    f = read_table((reports / "unseen_followup.md").read_text(), "Detector recall, budget 5%")
    f = f.pick(["type", "images", "held out", "trained on (control)", "p (exact)"])
    f = f.map_cells(counts_only)
    f = f.renamed(
        {
            "held out": "detector, type held out",
            "trained on (control)": "detector, type trained on",
            "p (exact)": "McNemar p (uncorrected)",
        }
    )
    return (
        t.render()
        + "\n\n"
        + f.render()
        + "\n\n"
        + source_line(src1, src2)
        + "\n\nImages caught / images that contain the type, evaluation half, at a 5% false-alarm"
        " budget set on the calibration normals; confidence intervals are in the reports. The"
        " second table is the exploratory follow-up (outside the approved protocol)."
    )


_LATENCY_VARIANTS = {
    "torch_fp32_strict": "PyTorch fp32 (strict)",
    "ort_cuda_tf32_off": "ORT CUDA, TF32 off",
    "ort_cuda_tf32_on": "ORT CUDA, TF32 on",
}


def onnx_tf32(reports: Path) -> str:
    """D-FINE-S accuracy of ONNX Runtime against PyTorch on val, and model latency per model."""
    src = "reports/benchmark.md"
    text = (reports / "benchmark.md").read_text()
    acc = read_table(text, "val (270 images, selection split)")
    acc = acc.pick(["variant", "AP@50", "AP@50:95", "images within atol", "top-100 overlap"])
    acc = acc.renamed(
        {"variant": "D-FINE-S on NEU-DET val", "images within atol": "images within 1e-3"}
    )

    lat = read_table(text, "Latency")
    order: list[str] = []
    p50: dict[str, dict[str, str]] = {}
    dev: dict[str, str] = {}
    for r in lat.rows:
        model, variant = r[lat.index("model")], r[lat.index("variant")]
        if model not in order:
            order.append(model)
        p50.setdefault(model, {})[variant] = r[lat.index("p50 ms")]
        if variant == "ort_cuda_tf32_off":
            dev[model] = r[lat.index("device MiB")]
    cols = list(_LATENCY_VARIANTS)
    lat_rows = [[m, *[p50[m][c] for c in cols], dev[m]] for m in sorted(order)]
    head = ["model, p50 ms (batch 1)", *_LATENCY_VARIANTS.values(), "ORT device MiB (TF32 off)"]
    lat_t = Table(head, lat_rows)
    note = (
        "`torch_fp32_strict` is PyTorch fp32 with TF32 off; `faster_rcnn` is the exported"
        " `faster_rcnn_aug` checkpoint; the anomaly models were timed on pcb1; device memory is"
        " the device-wide peak of the row and includes the CUDA context. Model only, batch 1,"
        " one run per row."
    )
    return acc.render() + "\n\n" + lat_t.render() + "\n\n" + source_line(src) + "\n\n" + note


def served_models(reports: Path) -> str:
    """What the API serves, with the TF32 setting the rule picked."""
    src = "reports/serve_models.md"
    text = (reports / "serve_models.md").read_text()
    prod = read_table(text, "Served products")
    rule = read_table(text, "TF32 rule applied")
    flips = {r[0]: r[rule.index("flips TF32 on")] for r in rule.rows}
    prod = prod.pick(["product", "model", "TF32", "device MiB"])
    rows = [[r[0], nice(r[1]), r[2], r[3], flips[r[0]]] for r in prod.rows]
    t = Table(["product", "model", "TF32", "device MiB", "flips with TF32 on"], rows)
    note = (
        "Flips with TF32 on: VisA, images on the other side of the frozen threshold than under"
        " PyTorch; steel, image-ratio pairs with a different number of kept boxes (no image is"
        " defect-free, so alarm flips cannot occur). The rule and the reasoning are in the report."
    )
    return _block(t, src, note=note)


def docker_facts(reports: Path) -> str:
    """The GPU image, from the recorded smoke test."""
    src = "reports/docker_smoke.json"
    j = json.loads((reports / "docker_smoke.json").read_text())
    gpu, cpu = j["gpu_container"], j["container_without_gpu"]
    steel, pcb1 = gpu["inspect"]["steel"], gpu["inspect"]["visa_pcb1"]
    size = j["image_size_bytes"]
    rows = [
        [
            "image size (`docker image inspect .Size`)",
            f"{size:,} bytes ({j['image_size_gib']} GiB)",
        ],
        ["files inside the image (`du -sx /`)", f"{j['image_filesystem_mib']:,} MiB"],
        ["smoke checks passed", str(len(j["checks_passed"]))],
        ["provider with `--gpus=all`", gpu["health"]["provider"]],
        ["provider without a GPU", cpu["health"]["provider"] + " (with a fallback reason)"],
        [
            "first `/inspect` steel, with model load",
            f"{steel['total_ms']:.0f} ms (load {steel['model_load_ms']:.0f} ms)",
        ],
        [
            "first `/inspect` visa_pcb1, with model load",
            f"{pcb1['total_ms']:.0f} ms (load {pcb1['model_load_ms']:.0f} ms)",
        ],
    ]
    rows.insert(0, ["date of the run", j["date"]])
    return _block(Table(["Docker image, one smoke run", "measured"], rows), src)


BLOCKS: dict[str, Callable[[Path], str]] = {
    "detectors": detectors,
    "anomaly": anomaly_models,
    "few_shot": few_shot,
    "thresholds": cost_thresholds,
    "unseen": unseen_types,
    "onnx": onnx_tf32,
    "served": served_models,
    "docker": docker_facts,
}


# ---- putting the blocks into README.md -----------------------------------------------------------


def _markers(name: str) -> tuple[str, str]:
    return f"<!-- generated:{name} -->", f"<!-- /generated:{name} -->"


def build_blocks(reports: Path) -> dict[str, str]:
    return {name: fn(reports) for name, fn in BLOCKS.items()}


def update_readme(readme: str, blocks: dict[str, str]) -> str:
    """Replace what is between each pair of marker lines. A missing marker pair is an error."""
    for name, body in blocks.items():
        start, end = _markers(name)
        if readme.count(start) != 1 or readme.count(end) != 1:
            raise ReportError(f"README.md must contain exactly one {start} ... {end} pair")
        if readme.index(start) > readme.index(end):
            raise ReportError(f"markers of {name!r} are out of order")
        head, rest = readme.split(start)
        _, tail = rest.split(end)
        readme = f"{head}{start}\n{body}\n{end}{tail}"
    return readme


def unmarked_names(readme: str) -> list[str]:
    """Marker pairs in the README that no block fills (a typo would leave a stale table)."""
    found = re.findall(r"<!-- generated:([a-z_]+) -->", readme)
    return [n for n in found if n not in BLOCKS]
