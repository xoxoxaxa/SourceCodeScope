"""Summarise tensors into compact, JSON-friendly records."""

from __future__ import annotations

import base64
import math

import torch
from torch.nn import functional

from sourcecodescope.ir import Preview, TensorRecord

HIST_BINS = 24
MAX_MAPS = 8
STRIP_WIDTH = 256
_MAP_RANK = 3


def summarize(
    tensor: torch.Tensor, tid: int, max_elems: int, *, keep_batch: bool = False
) -> TensorRecord:
    """Build a :class:`TensorRecord` for ``tensor``.

    The first dimension is treated as the batch and only its first entry is
    previewed, unless ``keep_batch`` is set (used for weight matrices).
    """
    data = tensor.detach()
    record = TensorRecord(
        id=tid,
        shape=list(data.shape),
        dtype=str(data.dtype).removeprefix("torch."),
        numel=data.numel(),
    )
    if data.numel() == 0:
        return record
    try:
        if data.is_complex():
            parts = [("abs", data.abs()), ("phase", data.angle())]
        else:
            parts = [("value", data)]
        parts = [(label, part.to(torch.float32)) for label, part in parts]
        _fill_stats(record, parts[0][1])
        for label, part in parts:
            view = part.unsqueeze(0) if keep_batch else part
            record.previews.append(_preview(label, view, max_elems))
    except (RuntimeError, TypeError, NotImplementedError):
        # Exotic tensors (sparse, quantised ...) keep their shape only.
        record.previews.clear()
    return record


def _fill_stats(record: TensorRecord, values: torch.Tensor) -> None:
    """Store mean / std / min / max and a histogram of the finite values."""
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return
    low = float(finite.min())
    high = float(finite.max())
    record.stats = {
        "mean": float(finite.mean()),
        "std": float(finite.std()) if finite.numel() > 1 else 0.0,
        "min": low,
        "max": high,
    }
    hist = torch.histc(finite, bins=HIST_BINS, min=low, max=high)
    record.hist = [int(count) for count in hist.tolist()]


def _preview(label: str, values: torch.Tensor, max_elems: int) -> Preview:
    """Reduce ``values`` to at most ``max_elems`` 8-bit samples."""
    view = values[0] if values.ndim > 1 else values
    while view.ndim > _MAP_RANK:
        view = view[0]
    view = torch.nan_to_num(view.reshape(-1) if view.ndim == 0 else view, 0.0, 0.0, 0.0)
    channels: list[int] = []
    if view.ndim == 1:
        layout = "strip"
        width = max(1, min(view.shape[0], STRIP_WIDTH, max_elems))
        small = functional.interpolate(view[None, None], size=width, mode="area")[0, 0]
    elif view.ndim == _MAP_RANK - 1:
        layout = "grid"
        small = _shrink(view[None], max_elems)[0]
    else:
        layout = "maps"
        channels = _pick(view.shape[0], MAX_MAPS)
        small = _shrink(view[channels], max(1, max_elems // len(channels)))
    low = float(small.min())
    high = float(small.max())
    span = (high - low) or 1.0
    quantised = ((small - low) / span * 255.0).round().clamp(0, 255).to(torch.uint8)
    payload = base64.b64encode(bytes(quantised.flatten().tolist())).decode("ascii")
    return Preview(label, layout, list(small.shape), low, high, payload, channels)


def _shrink(stack: torch.Tensor, budget: int) -> torch.Tensor:
    """Average-pool a ``[n, h, w]`` stack so that ``h * w <= budget``."""
    height, width = stack.shape[-2], stack.shape[-1]
    if height * width <= budget:
        return stack
    scale = math.sqrt(budget / (height * width))
    new_height = max(1, int(height * scale))
    new_width = max(1, min(width, budget // new_height))
    return functional.adaptive_avg_pool2d(stack[None], (new_height, new_width))[0]


def _pick(total: int, limit: int) -> list[int]:
    """Choose up to ``limit`` evenly spaced indices out of ``total``."""
    if total <= limit:
        return list(range(total))
    step = (total - 1) / (limit - 1)
    return sorted({round(index * step) for index in range(limit)})
