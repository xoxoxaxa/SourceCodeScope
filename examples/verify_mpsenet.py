"""Visualise MP-SENet (https://github.com/yxlu-0102/MP-SENet) in both modes.

Usage::

    git clone https://github.com/yxlu-0102/MP-SENet external/MP-SENet
    uv run python examples/verify_mpsenet.py

The script contains nothing specific to MP-SENet beyond loading the model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch

import sourcecodescope as scs


def load_model(repo: Path, checkpoint: Path | None) -> tuple[torch.nn.Module, dict[str, int]]:
    """Build ``MPNet`` from the cloned repository."""
    if not (repo / "models" / "model.py").is_file():
        raise SystemExit(
            f"MP-SENet was not found in {repo}.\n"
            f"Run: git clone https://github.com/yxlu-0102/MP-SENet {repo}"
        )
    sys.path.insert(0, str(repo))
    try:
        importlib.import_module("pesq")
    except ImportError:
        # pesq is only used by the training metric and needs a C compiler.
        stub = types.ModuleType("pesq")
        stub.__dict__["pesq"] = None
        sys.modules["pesq"] = stub
    config = json.loads((repo / "config.json").read_text(encoding="utf-8"))
    hparams = importlib.import_module("env").AttrDict(config)
    model = importlib.import_module("models.model").MPNet(hparams)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["generator"])
    return model, config


def main() -> None:
    """Trace the model once and write ``mpsenet_<mode>.html`` for both modes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("external/MP-SENet"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="e.g. best_ckpt/g_best_vb")
    parser.add_argument("--frames", type=int, default=64, help="number of STFT frames (T)")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config = load_model(args.repo, args.checkpoint)
    model = model.to(args.device)
    bins = config["n_fft"] // 2 + 1
    generator = torch.Generator().manual_seed(0)
    noisy_amp = torch.rand(1, bins, args.frames, generator=generator).to(args.device)
    noisy_pha = (torch.rand(1, bins, args.frames, generator=generator) * 2 - 1) * torch.pi
    inputs = (noisy_amp, noisy_pha.to(args.device))
    names = {"noisy_amp": ("B", "F", "T"), "noisy_pha": ("B", "F", "T")}

    for mode in ("flow", "explainer"):
        path = scs.visualize(
            model, inputs, mode=mode, out=args.out / f"mpsenet_{mode}.html", dim_names=names
        )
        print(f"{mode}: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
