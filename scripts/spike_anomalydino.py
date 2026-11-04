"""Spike: settle the AnomalyDINO encoder name from installed anomalib source, then run it.

Usage: uv run python scripts/spike_anomalydino.py category=cashew shots=5 [encoder_only=true]

Steps 1-4 use no dataset. Step 5 (end-to-end smoke AUROC) needs data/raw/visa.
The AUROC printed here is a smoke value, not a reported result.
"""

from __future__ import annotations

import inspect
import re
import sys
from pathlib import Path

import anomalib
import torch
from anomalib.models.image.anomaly_dino import AnomalyDINO, lightning_model, torch_model

from defectlens.seed import seed_everything

EXPECTED_ANOMALIB = "2.7.0"
VISA_ROOT = Path("data/raw/visa")


def parse_args(argv: list[str]) -> dict[str, str]:
    """Parse ``key=value`` arguments."""
    out: dict[str, str] = {}
    for arg in argv:
        key, sep, value = arg.partition("=")
        if not sep:
            msg = f"expected key=value, got {arg!r}"
            raise SystemExit(msg)
        out[key] = value
    return out


def settle_encoder_name() -> str:
    """Steps 1-4. Returns the encoder name proven by a real forward pass."""
    # 1. version and source path
    print(f"[1] anomalib {anomalib.__version__} (expected {EXPECTED_ANOMALIB})")
    assert anomalib.__version__ == EXPECTED_ANOMALIB
    for mod in (lightning_model, torch_model):
        print(f"    source: {inspect.getsourcefile(mod)}")

    # 2. encoder_name defaults from the installed source, confirmed with inspect.signature
    names = set()
    for label, cls in (
        ("lightning_model.AnomalyDINO", lightning_model.AnomalyDINO),
        ("torch_model.AnomalyDINOModel", torch_model.AnomalyDINOModel),
    ):
        default = inspect.signature(cls.__init__).parameters["encoder_name"].default
        src = inspect.getsource(cls.__init__)
        in_src = re.search(r'encoder_name: str = "([^"]+)"', src)
        print(f"[2] {label}: signature default {default!r}; source text {in_src and in_src[1]!r}")
        names.add(default)
    assert len(names) == 1, f"defaults disagree: {names}"
    encoder_name = names.pop()

    # 3. timm resolves the name (weights download on first use)
    model = torch_model.AnomalyDINOModel(encoder_name=encoder_name)
    print(f"[3] instantiated AnomalyDINOModel; timm backbone {model.encoder_name!r}")

    # 4. real forward pass; patch grid is 18x18 at 252 px with patch size 14
    model = model.cuda().eval()
    x = torch.randn(2, 3, 252, 252, device="cuda")
    with torch.no_grad():
        feats = model.extract_features(x)
    print(
        f"[4] feature shape {tuple(feats.shape)} (B, patches, dim), patch_size {model.patch_size}"
    )
    return encoder_name


def end_to_end(encoder_name: str, category: str, shots: int, seed: int) -> float:
    """Step 5: fit a k-shot memory bank and test on the VisA category. Returns image AUROC."""
    from anomalib.data import Visa
    from anomalib.engine import Engine

    class FewShotVisa(Visa):
        """Visa whose train set is a fixed random k-shot subset, applied on every setup()."""

        def _setup(self, _stage: str | None = None) -> None:
            super()._setup(_stage)
            samples = self.train_data.samples
            self.train_data.samples = samples.sample(n=shots, random_state=seed).reset_index(
                drop=True
            )

    datamodule = FewShotVisa(
        root=VISA_ROOT,
        category=category,
        train_batch_size=min(shots, 8),
        eval_batch_size=8,
        num_workers=2,
    )
    datamodule.prepare_data()
    print(f"[5] {category}: using {shots}-shot train subset (seed {seed})")

    model = AnomalyDINO(encoder_name=encoder_name)
    engine = Engine(accelerator="gpu", devices=1, logger=False, default_root_dir="outputs/spike")
    torch.cuda.reset_peak_memory_stats()
    engine.fit(model=model, datamodule=datamodule)
    bank = tuple(model.model.memory_bank.shape)
    print(f"[5] memory bank {bank} (expect {shots} x 324 patches = {shots * 324})")
    assert bank[0] == shots * 324, "k-shot subset was not applied"
    results = engine.test(model=model, datamodule=datamodule)
    peak = torch.cuda.max_memory_allocated() / 2**20
    print(f"[5] test results: {results}")
    print(f"[5] peak VRAM (torch allocator) {peak:.0f} MiB")
    return float(results[0]["image_AUROC"])


def main() -> int:
    args = parse_args(sys.argv[1:])
    seed = int(args.get("seed", 42))
    seed_everything(seed)
    encoder_name = settle_encoder_name()
    if args.get("encoder_only") == "true":
        return 0
    auroc = end_to_end(
        encoder_name, args.get("category", "cashew"), int(args.get("shots", 5)), seed
    )
    print(f"[5] SMOKE image AUROC {auroc:.4f} (not a reported result)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
