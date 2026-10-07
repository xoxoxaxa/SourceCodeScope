"""Intermediate representation shared by the tracer and the renderers.

The graph is hierarchical: every node has an optional parent, and edges only
connect leaves (inputs, outputs, operations and leaf modules).  Renderers lift
edge endpoints to whichever ancestor is currently visible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IR_VERSION = 1


@dataclass
class Preview:
    """A down-sampled, 8-bit quantised view of a tensor.

    ``layout`` is ``"strip"`` (``shape == [w]``), ``"grid"`` (``[h, w]``) or
    ``"maps"`` (``[n, h, w]``, one map per entry of ``channels``).
    """

    label: str
    layout: str
    shape: list[int]
    lo: float
    hi: float
    data: str
    channels: list[int] = field(default_factory=list)


@dataclass
class TensorRecord:
    """Shape, statistics and previews of one tensor value."""

    id: int
    shape: list[int]
    dtype: str
    numel: int
    stats: dict[str, float] | None = None
    hist: list[int] = field(default_factory=list)
    previews: list[Preview] = field(default_factory=list)


@dataclass
class Node:
    """A module call, an operation, a model input/output or a repeat group.

    ``kind`` is one of ``input``, ``output``, ``module``, ``op`` or ``repeat``;
    ``category`` is the display family (``conv``, ``attention``, ``shape`` ...).
    """

    id: int
    kind: str
    name: str
    path: str
    type: str
    category: str
    parent: int | None = None
    params: int = 0
    count: int = 1
    inputs: list[int] = field(default_factory=list)
    outputs: list[int] = field(default_factory=list)
    attrs: dict[str, str] = field(default_factory=dict)
    attention: int | None = None
    weight: int | None = None


@dataclass(frozen=True)
class Edge:
    """A tensor flowing from one leaf node to another."""

    src: int
    dst: int
    tensor: int


@dataclass
class Graph:
    """The complete trace of one forward pass."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    tensors: list[TensorRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        data = asdict(self)
        data["version"] = IR_VERSION
        return data

    def to_json(self) -> str:
        """Serialise the graph to a compact JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))

    def save(self, path: str | Path) -> Path:
        """Write the graph to ``path`` as JSON and return the path."""
        target = Path(path)
        target.write_text(self.to_json(), encoding="utf-8")
        return target

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Graph:
        """Rebuild a graph from :meth:`to_dict` output."""
        tensors = []
        for item in data.get("tensors", []):
            record = TensorRecord(item["id"], item["shape"], item["dtype"], item["numel"])
            record.stats = item.get("stats")
            record.hist = list(item.get("hist", []))
            record.previews = [Preview(**preview) for preview in item.get("previews", [])]
            tensors.append(record)
        return cls(
            nodes=[Node(**item) for item in data.get("nodes", [])],
            edges=[Edge(**item) for item in data.get("edges", [])],
            tensors=tensors,
            meta=dict(data.get("meta", {})),
        )

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        """Read a graph previously written with :meth:`save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
