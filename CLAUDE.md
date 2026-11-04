# DefectLens — project brief for Claude Code

Industrial visual inspection: a supervised detector for known defects (NEU-DET) plus an
anomaly detector for unseen defects (VisA), exported to ONNX and served as an API.
Portfolio project; the code, tests and README matter as much as the numbers.

## Hardware and environment (hard constraints)
- Laptop: Lenovo LOQ, Ryzen 7 7435HS, 16 GB RAM, RTX 4060 Laptop GPU (8 GB VRAM).
- Work inside WSL2 Ubuntu (if available) with the NVIDIA CUDA driver from Windows.
- Everything is free: no paid cloud, no paid APIs.
- Every training config must fit in 8 GB VRAM: use small/nano model variants, mixed precision,
  gradient accumulation instead of large batches. State expected VRAM and runtime for each run.
- Never start a training run longer than ~5 minutes yourself. Write the script and the
  command, then stop; I run long jobs and paste back the output.

## Stack
- Python 3.11, uv for env and deps (`uv add`, `uv run`). No pip install into system Python.
- PyTorch + torchvision (CUDA build), Hugging Face Transformers (D-FINE / RT-DETRv2), timm.
- anomalib: PatchCore (wide_resnet50_2 baseline), AnomalyDINO (DINOv2-small, few-shot),
  EfficientAD (speed). EfficientAD has its own network; do not try to give it a DINOv2 backbone.
- OpenCV, scikit-image, Albumentations, pycocotools.
- Hydra for configs (`configs/`), MLflow (local `mlruns/`) for every experiment, DVC for data.
- ONNX, ONNX Runtime GPU (CUDA + TensorRT execution providers).
- FastAPI, Gradio, Docker. pytest, ruff, mypy, GitHub Actions.

## Repo layout
```
src/defectlens/
  data/        # dataset download, conversion to COCO, datamodules
  preprocess/  # OpenCV: resize, CLAHE, ROI crop, calibration
  detection/   # classical baseline, Faster R-CNN, D-FINE
  anomaly/     # anomalib wrappers, few-shot, synthetic defects
  eval/        # metrics, cost-based thresholds, error gallery
  export/      # ONNX export, ORT/TensorRT benchmarks
  serve/       # FastAPI app, Gradio demo
configs/       # Hydra configs
scripts/       # thin CLI entry points
tests/         # pytest, fast (<60 s total, CPU only, tiny fixtures)
reports/       # results tables and figures that go into the README
```

## Working rules
- Work phase by phase from PLAN.md. At the start of a phase, restate the goal and the
  "done when" checklist, then propose the file list before writing code.
- Never invent metrics. Every number in README/reports must come from an MLflow run or a
  script output I pasted. Leave a TODO placeholder if a number is not measured yet.
- MLflow uses the SQLite backend: tracking URI `sqlite:///mlruns/mlflow.db`, artifacts in
  `mlruns/artifacts` (the file backend is in maintenance mode). All of it stays under git-ignored `mlruns/`.
- Fix random seeds; log config, git commit hash and dataset version with every run.
- Keep train/val/test splits fixed and saved to disk; never tune thresholds on the test set.
- Add or update tests with every module. `uv run pytest`, `uv run ruff check .` and
  `uv run mypy src` must pass before a phase counts as done.
- Do not commit data, model weights or mlruns/ (they go through DVC or .gitignore).
- Git and GitHub are mine only. Never run git commands that change the repo (add, commit,
  push, pull, merge, rebase, reset, checkout, branch, remote, tag, stash) and never use the
  gh CLI or the GitHub API. Read-only `git status`, `git diff` and `git log` are fine.
  At the end of a phase, tell me which files changed and suggest a commit message; I commit
  and push myself. Writing workflow files (e.g. .github/workflows/ci.yml) is fine.
- At the end of each phase, update PLAN.md (status + measured results) and the README section.
- Never add Co-Authored-By or any Claude attribution lines to suggested commit messages.
