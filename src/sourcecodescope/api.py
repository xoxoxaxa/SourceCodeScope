"""High-level entry point: trace a model and write its visualisation."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch import nn

from sourcecodescope.render import build_html
from sourcecodescope.tracer import TraceOptions, trace

MAX_HTML_BYTES = 20 * 1024 * 1024
_MIN_TENSOR_ELEMS = 64


def visualize(
    model: nn.Module,
    inputs: Any,
    *,
    mode: str = "explainer",
    out: str | Path = "model.html",
    depth: int = 2,
    dim_names: Mapping[str, Sequence[str]] | None = None,
    axis_labels: Mapping[str, Sequence[str]] | None = None,
    attention_modules: Sequence[str] | None = None,
    max_tensor_elems: int = 4096,
) -> Path:
    """Run ``model`` on ``inputs`` once and write an interactive HTML view.

    Args:
        model: Any ``nn.Module``.  It is run in eval mode without gradients and
            is left unchanged.
        inputs: A tensor, a tuple of positional arguments or a dict of keyword
            arguments for ``model``.
        mode: ``"explainer"`` (rich view with values) or ``"flow"`` (structure
            and shapes).  The page can switch between both afterwards.
        out: Path of the HTML file to write.
        depth: Number of hierarchy levels that start out expanded.
        dim_names: Axis names per input, e.g. ``{"noisy_amp": ("B", "F", "T")}``.
        axis_labels: Position labels per axis name, e.g. ``{"T": ["The", "cat"]}``.
        attention_modules: Module paths to tag as attention when automatic
            detection misses them.
        max_tensor_elems: Upper bound of stored samples per tensor.  It is
            halved automatically while the page would exceed 20 MB.

    Returns:
        The path of the written HTML file.
    """
    budget = max_tensor_elems
    while True:
        options = TraceOptions(
            max_tensor_elems=budget,
            attention_modules=tuple(attention_modules or ()),
            dim_names=dict(dim_names or {}),
            axis_labels=dict(axis_labels or {}),
        )
        page = build_html(trace(model, inputs, options), mode, depth)
        if len(page.encode("utf-8")) <= MAX_HTML_BYTES or budget <= _MIN_TENSOR_ELEMS:
            break
        budget //= 2
        warnings.warn(
            f"HTML output exceeds 20 MB; retrying with max_tensor_elems={budget}.",
            stacklevel=2,
        )
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding="utf-8")
    return target
