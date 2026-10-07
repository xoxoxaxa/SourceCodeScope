"""Visualise two unrelated architectures to check that nothing is model specific.

Usage::

    uv run python examples/verify_generic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision.models import resnet18

import sourcecodescope as scs


def main() -> None:
    """Write flow and explainer pages for ResNet-18 and ``nn.TransformerEncoder``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("out"))
    args = parser.parse_args()

    generator = torch.Generator().manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", "."]
    cases: dict[str, tuple[nn.Module, torch.Tensor]] = {
        "resnet18": (resnet18(weights=None), torch.randn(1, 3, 96, 96, generator=generator)),
        "transformer_encoder": (
            nn.TransformerEncoder(layer, num_layers=3),
            torch.randn(1, len(tokens), 64, generator=generator),
        ),
    }
    names = {"src": ("B", "T", "C"), "x": ("B", "C", "H", "W")}
    labels = {"T": tokens}
    for name, (model, inputs) in cases.items():
        for mode in ("flow", "explainer"):
            target = args.out / f"{name}_{mode}.html"
            path = scs.visualize(
                model, inputs, mode=mode, out=target, dim_names=names, axis_labels=labels
            )
            print(f"{name} / {mode}: {path} ({path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
