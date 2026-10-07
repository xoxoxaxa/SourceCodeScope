"""End-to-end checks on small models."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

import sourcecodescope as scs
from sourcecodescope.ir import Graph

BLOCKS = 3
BUDGET = 64


class Block(nn.Module):
    """Residual block with hand-written attention."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention and an MLP, each with a skip connection."""
        y = self.norm(x)
        scores = self.query(y) @ self.key(y).transpose(-2, -1) / y.shape[-1] ** 0.5
        x = x + scores.softmax(dim=-1) @ y
        return x + self.mlp(x)


class Tiny(nn.Module):
    """Two inputs, two outputs, repeated blocks and a built-in attention layer."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Conv1d(3, 16, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([Block(16) for _ in range(BLOCKS)])
        self.attn = nn.MultiheadAttention(16, 4, batch_first=True)
        self.head = nn.Linear(16, 5)

    def forward(
        self, signal: torch.Tensor, bias: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and the pooled features."""
        x = self.embed(signal).permute(0, 2, 1) + bias
        for block in self.blocks:
            x = block(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        pooled = x.mean(dim=1)
        return self.head(pooled), pooled


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return torch.randn(2, 3, 12, generator=generator), torch.randn(2, 12, 16, generator=generator)


def _trace() -> Graph:
    return scs.trace(Tiny(), _inputs())


def test_inputs_outputs_and_edges() -> None:
    graph = _trace()
    kinds = [node.kind for node in graph.nodes]
    assert kinds.count("input") == len(_inputs())
    assert kinds.count("output") == len(_inputs())
    ids = {node.id for node in graph.nodes}
    assert all(edge.src in ids and edge.dst in ids for edge in graph.edges)
    names = {node.name for node in graph.nodes if node.kind == "op"}
    assert {"permute", "add", "matmul", "softmax", "mean"} <= names


def test_repeated_blocks_are_bundled() -> None:
    graph = _trace()
    groups = [node for node in graph.nodes if node.kind == "repeat"]
    assert [(group.name, group.count) for group in groups] == [("blocks", BLOCKS)]
    members = [node for node in graph.nodes if node.parent == groups[0].id]
    assert [member.path for member in members] == ["blocks.0", "blocks.1", "blocks.2"]


def test_attention_is_detected() -> None:
    graph = _trace()
    tensors = {tensor.id: tensor for tensor in graph.tensors}
    builtin = next(node for node in graph.nodes if node.path == "attn")
    assert builtin.attention is not None
    assert tensors[builtin.attention].shape == [2, 4, 12, 12]
    manual = [node for node in graph.nodes if node.name == "softmax"]
    assert len(manual) == BLOCKS
    assert all(node.category == "attention" and node.attention is not None for node in manual)


def test_model_is_left_untouched() -> None:
    model = Tiny().train()
    inputs = _inputs()
    scs.trace(model, inputs)
    assert model.training
    assert all(module.training for module in model.modules())
    hooks = sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
    assert hooks == 0
    model.eval()
    first = model(*inputs)
    scs.trace(model, inputs)
    second = model(*inputs)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_previews_respect_the_budget() -> None:
    graph = scs.trace(Tiny(), _inputs(), scs.TraceOptions(max_tensor_elems=BUDGET))
    flow = {tid for node in graph.nodes for tid in (*node.inputs, *node.outputs)}
    for tensor in graph.tensors:
        if tensor.id not in flow:
            continue
        for preview in tensor.previews:
            size = 1
            for dim in preview.shape:
                size *= dim
            assert size <= BUDGET


def test_ir_round_trip(tmp_path: Path) -> None:
    graph = _trace()
    restored = Graph.load(graph.save(tmp_path / "ir.json"))
    assert restored.to_dict() == graph.to_dict()


def test_visualize_writes_self_contained_html(tmp_path: Path) -> None:
    for mode in ("flow", "explainer"):
        path = scs.visualize(
            Tiny(),
            _inputs(),
            mode=mode,
            out=tmp_path / f"{mode}.html",
            dim_names={"signal": ("B", "C", "T")},
        )
        page = path.read_text(encoding="utf-8")
        assert 'id="scs-data"' in page
        assert ' src="http' not in page
        assert "<link" not in page
        assert f'"mode": "{mode}"' in page


def test_dict_inputs_and_bad_mode(tmp_path: Path) -> None:
    signal, bias = _inputs()
    graph = scs.trace(Tiny(), {"signal": signal, "bias": bias})
    assert [node.name for node in graph.nodes if node.kind == "input"] == ["signal", "bias"]
    try:
        scs.render(graph, mode="other", out=tmp_path / "x.html")
    except ValueError:
        return
    raise AssertionError("an unknown mode must be rejected")
