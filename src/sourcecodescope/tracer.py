"""Record one forward pass of a model as a hierarchical graph.

Module calls are observed with forward hooks; everything else (operators,
``torch.*`` functions, tensor methods) is observed with a
:class:`~torch.overrides.TorchFunctionMode`.  Nothing in the model is patched,
and all hooks are removed when the trace ends.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional
from torch.overrides import TorchFunctionMode

from sourcecodescope.capture import summarize
from sourcecodescope.classify import is_leaf_module, module_category, op_category
from sourcecodescope.ir import Edge, Graph, Node
from sourcecodescope.postprocess import finalize

_ALIAS_OPS = frozenset(
    {
        "contiguous",
        "detach",
        "to",
        "type",
        "type_as",
        "float",
        "double",
        "half",
        "cpu",
        "cuda",
        "clone",
        "data",
        "requires_grad_",
    }
)
_MATMUL_OPS = frozenset({"matmul", "bmm", "mm", "einsum"})
_WEIGHT_CATEGORIES = frozenset({"conv", "linear"})
_DUNDER = {
    "__add__": "add",
    "__radd__": "add",
    "__iadd__": "add_",
    "__sub__": "sub",
    "__rsub__": "rsub",
    "__isub__": "sub_",
    "__mul__": "mul",
    "__rmul__": "mul",
    "__imul__": "mul_",
    "__truediv__": "div",
    "__rtruediv__": "rdiv",
    "__itruediv__": "div_",
    "__matmul__": "matmul",
    "__rmatmul__": "matmul",
    "__pow__": "pow",
    "__rpow__": "rpow",
    "__neg__": "neg",
    "__getitem__": "getitem",
    "__setitem__": "setitem",
}
_SDPA_PARAMS = ("query", "key", "value", "attn_mask", "dropout_p", "is_causal", "scale")
_ATTENTION_RANK = 3
_ATTENTION_BUDGET = 4
_MAX_WEIGHT_ELEMS = 20_000_000
_MAX_TEXT = 160


@dataclass(frozen=True)
class TraceOptions:
    """Settings of one trace.

    ``attention_modules`` lists module paths (as in ``named_modules``) to tag
    as attention when automatic detection misses them.  ``dim_names`` maps an
    input name to its axis names and ``axis_labels`` maps an axis name to the
    labels of its positions.
    """

    max_tensor_elems: int = 4096
    attention_modules: tuple[str, ...] = ()
    dim_names: Mapping[str, Sequence[str]] = field(default_factory=dict)
    axis_labels: Mapping[str, Sequence[str]] = field(default_factory=dict)


class _TensorTable:
    """Remembers which node produced which tensor value."""

    def __init__(self, graph: Graph, max_elems: int) -> None:
        self._graph = graph
        self._max_elems = max_elems
        self._ids: dict[int, int] = {}
        # Keeps traced tensors alive so that ``id()`` values are never reused.
        self._keep: list[torch.Tensor] = []
        self.producers: dict[int, int] = {}

    def lookup(self, tensor: torch.Tensor) -> int | None:
        """Return the record id of ``tensor`` if it was seen before."""
        return self._ids.get(id(tensor))

    def known(self, tensors: Iterable[torch.Tensor]) -> list[int]:
        """Return the distinct record ids of the already-seen ``tensors``."""
        found = (self._ids.get(id(tensor)) for tensor in tensors)
        return list(dict.fromkeys(tid for tid in found if tid is not None))

    def define(self, tensor: torch.Tensor, producer: int) -> int:
        """Record a new value of ``tensor`` produced by node ``producer``."""
        tid = self.snapshot(tensor)
        self._keep.append(tensor)
        self._ids[id(tensor)] = tid
        self.producers[tid] = producer
        return tid

    def snapshot(self, tensor: torch.Tensor, scale: int = 1, *, keep_batch: bool = False) -> int:
        """Record ``tensor`` as side data that is not part of the data flow."""
        tid = len(self._graph.tensors)
        budget = self._max_elems * scale
        self._graph.tensors.append(summarize(tensor, tid, budget, keep_batch=keep_batch))
        return tid

    def alias(self, tensor: torch.Tensor, tid: int) -> None:
        """Treat ``tensor`` as the same value as record ``tid``."""
        self._keep.append(tensor)
        self._ids[id(tensor)] = tid


class Tracer(TorchFunctionMode):
    """Runs a model once and records what happened.

    The tracer is itself the function mode that receives every torch API call
    made during the forward pass.
    """

    def __init__(self, model: nn.Module, options: TraceOptions) -> None:
        super().__init__()
        self._model = model
        self._options = options
        self.graph = Graph()
        self._table = _TensorTable(self.graph, options.max_tensor_elems)
        self._names = {id(module): name for name, module in model.named_modules()}
        self._calls: dict[int, int] = {}
        self._edges: set[Edge] = set()
        self._stack: list[Node] = []
        self._leaf_depth = 0
        self._paused = False
        self._side: torch.Tensor | None = None
        self._softmax: dict[int, Node] = {}

    def run(self, inputs: Any) -> Graph:
        """Trace ``model(inputs)`` and return the raw graph."""
        args, kwargs = _as_call(inputs)
        modes = [(module, module.training) for module in self._model.modules()]
        handles = []
        for module in self._model.modules():
            handles.append(module.register_forward_pre_hook(self._enter, with_kwargs=True))
            handles.append(
                module.register_forward_hook(self._exit, with_kwargs=True, always_call=True)
            )
        self._model.eval()
        try:
            self._add_inputs(args, kwargs)
            with torch.no_grad(), self:
                output = self._model(*args, **kwargs)
            self._add_outputs(output)
        finally:
            for handle in handles:
                handle.remove()
            for module, training in modes:
                module.training = training
        self.graph.meta.update(
            model=type(self._model).__name__,
            torch=torch.__version__,
            dim_names={name: list(dims) for name, dims in self._options.dim_names.items()},
            axis_labels={name: list(tags) for name, tags in self._options.axis_labels.items()},
        )
        return self.graph

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        """Entry point used by torch for every intercepted call."""
        del types
        return self.on_function(func, args, kwargs or {})

    def on_function(self, func: Callable[..., Any], args: tuple[Any, ...], kwargs: Any) -> Any:
        """Execute one torch API call and record it as an operation."""
        if self._paused:
            return func(*args, **kwargs)
        result, weights = _call(func, args, kwargs)
        if self._leaf_depth:
            if weights is not None:
                self._side = weights
            return result
        self._record_op(_func_name(func), (args, kwargs), result, weights)
        return result

    @contextmanager
    def _pause(self) -> Iterator[None]:
        """Stop recording while the tracer itself uses torch."""
        previous = self._paused
        self._paused = True
        try:
            yield
        finally:
            self._paused = previous

    def _add_node(self, kind: str, name: str, type_name: str, category: str) -> Node:
        parent = self._stack[-1] if self._stack else None
        path = name if parent is None or kind == "module" else f"{parent.path}.{name}"
        node = Node(
            id=len(self.graph.nodes),
            kind=kind,
            name=name.rsplit(".", 1)[-1] if kind == "module" else name,
            path=path,
            type=type_name,
            category=category,
            parent=None if parent is None else parent.id,
        )
        self.graph.nodes.append(node)
        return node

    def _connect(self, node: Node, tids: Iterable[int]) -> None:
        for tid in tids:
            source = self._table.producers.get(tid)
            edge = Edge(-1 if source is None else source, node.id, tid)
            if source is None or source == node.id or edge in self._edges:
                continue
            self._edges.add(edge)
            self.graph.edges.append(edge)

    def _add_inputs(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        for label, tensor in _named_inputs(self._model, args, kwargs):
            node = self._add_node("input", label, "input", "input")
            node.outputs = [self._table.define(tensor, node.id)]
            dims = self._options.dim_names.get(label)
            if dims:
                node.attrs["dims"] = ",".join(dims)

    def _add_outputs(self, output: Any) -> None:
        for label, tensor in _named(output, "output"):
            node = self._add_node("output", label, "output", "output")
            node.inputs = self._table.known([tensor])
            self._connect(node, node.inputs)

    def _enter(self, module: nn.Module, args: Any, kwargs: Any) -> None:
        if self._leaf_depth:
            self._leaf_depth += 1
            return
        with self._pause():
            node = self._module_node(module)
            node.inputs = self._table.known(_tensors((args, kwargs)))
            self._stack.append(node)
            if is_leaf_module(module):
                self._leaf_depth = 1
                self._connect(node, node.inputs)

    def _exit(self, module: nn.Module, _args: Any, _kwargs: Any, output: Any) -> None:
        if self._leaf_depth > 1:
            self._leaf_depth -= 1
            return
        with self._pause():
            node = self._stack.pop()
            tensors = list(_tensors(output))
            if self._leaf_depth == 0:
                node.outputs = self._table.known(tensors)
                return
            self._leaf_depth = 0
            node.outputs = [self._table.define(tensor, node.id) for tensor in tensors]
            self._finish_leaf(module, node)

    def _module_node(self, module: nn.Module) -> Node:
        type_name = type(module).__name__
        base = self._names.get(id(module), type_name) or type_name
        calls = self._calls.get(id(module), 0) + 1
        self._calls[id(module)] = calls
        tagged = base in self._options.attention_modules
        category = "attention" if tagged else module_category(module)
        node = self._add_node(
            "module", base if calls == 1 else f"{base}#{calls}", type_name, category
        )
        node.params = sum(parameter.numel() for parameter in module.parameters())
        config = module.extra_repr()
        if config:
            node.attrs["config"] = config[:_MAX_TEXT]
        if isinstance(module, nn.PReLU):
            node.attrs["slope"] = f"{float(module.weight.mean()):.4g}"
        return node

    def _finish_leaf(self, module: nn.Module, node: Node) -> None:
        if self._side is not None:
            node.attention = self._table.snapshot(self._side, _ATTENTION_BUDGET)
            self._side = None
        weight = getattr(module, "weight", None)
        if (
            node.category in _WEIGHT_CATEGORIES
            and isinstance(weight, torch.Tensor)
            and weight.numel() <= _MAX_WEIGHT_ELEMS
        ):
            matrix = weight.reshape(weight.shape[0], -1)
            node.weight = self._table.snapshot(matrix, keep_batch=True)
        if not node.outputs:
            return
        if isinstance(module, nn.Softmax):
            self._softmax[node.outputs[0]] = node
        elif node.category == "dropout" and node.inputs and node.inputs[0] in self._softmax:
            self._softmax[node.outputs[0]] = self._softmax[node.inputs[0]]

    def _record_op(
        self,
        name: str,
        call: tuple[tuple[Any, ...], Any],
        result: Any,
        weights: torch.Tensor | None,
    ) -> None:
        args, kwargs = call
        inputs = list(_tensors(call))
        outputs = list(dict.fromkeys(_tensors(result)))
        if name == "setitem" and inputs:
            outputs = inputs[:1]
        if not outputs:
            return
        known = self._table.known(inputs)
        if _is_alias(name, inputs, outputs):
            for tensor in outputs:
                if known and self._table.lookup(tensor) is None:
                    self._table.alias(tensor, known[0])
            return
        node = self._add_node("op", name, name, op_category(name))
        node.inputs = known
        self._connect(node, known)
        text = _format_call(args, kwargs)
        if text:
            node.attrs["args"] = text
        node.outputs = [self._table.define(tensor, node.id) for tensor in outputs]
        if weights is not None:
            node.category = "attention"
            node.attention = self._table.snapshot(weights, _ATTENTION_BUDGET)
        self._track_attention(node, name, outputs[0].ndim)

    def _track_attention(self, node: Node, name: str, rank: int) -> None:
        """Detect hand-written attention: a softmax whose result feeds a matmul."""
        first = node.inputs[0] if node.inputs else None
        if name == "softmax" and rank >= _ATTENTION_RANK:
            self._softmax[node.outputs[0]] = node
        elif name == "dropout" and first in self._softmax:
            self._softmax[node.outputs[0]] = self._softmax[first]
        elif name in _MATMUL_OPS and first in self._softmax:
            source = self._softmax[first]
            source.category = "attention"
            source.attention = first
            owner = self._stack[-1] if self._stack else None
            if owner is not None and owner.attention is None:
                owner.attention = first


def trace(model: nn.Module, inputs: Any, options: TraceOptions | None = None) -> Graph:
    """Run ``model`` on ``inputs`` once and return the recorded graph.

    ``inputs`` may be a tensor, a tuple / list of positional arguments or a
    dictionary of keyword arguments.
    """
    graph = Tracer(model, options or TraceOptions()).run(inputs)
    finalize(graph)
    return graph


def _as_call(inputs: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if isinstance(inputs, Mapping):
        return (), dict(inputs)
    if isinstance(inputs, (tuple, list)):
        return tuple(inputs), {}
    return (inputs,), {}


def _named(value: Any, prefix: str) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(label, tensor)`` for every tensor nested inside ``value``."""
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _named(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _named(item, f"{prefix}[{index}]")


def _tensors(value: Any) -> Iterator[torch.Tensor]:
    for _, tensor in _named(value, ""):
        yield tensor


def _named_inputs(
    model: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Iterator[tuple[str, torch.Tensor]]:
    try:
        bound = inspect.signature(model.forward).bind(*args, **kwargs).arguments
    except (TypeError, ValueError):
        bound = {f"input{index}": value for index, value in enumerate(args)} | kwargs
    for name, value in bound.items():
        yield from _named(value, name)


def _func_name(func: Callable[..., Any]) -> str:
    name = getattr(func, "__name__", None) or type(func).__name__
    if name in ("__get__", "__set__"):
        name = getattr(getattr(func, "__self__", None), "__name__", name)
    return _DUNDER.get(name, name)


def _is_inplace(name: str) -> bool:
    return name == "setitem" or (name.endswith("_") and not name.endswith("__"))


def _is_alias(name: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor]) -> bool:
    """Return True for calls that hand back an existing value unchanged."""
    if name in _ALIAS_OPS:
        return True
    if _is_inplace(name):
        return False
    return all(any(output is tensor for tensor in inputs) for output in outputs)


def _call(
    func: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[Any, torch.Tensor | None]:
    """Run ``func`` and also return attention weights when it computes any."""
    if func is functional.multi_head_attention_forward:
        options = dict(kwargs)
        wanted = options.get("need_weights", True)
        averaged = options.get("average_attn_weights", True)
        options.update(need_weights=True, average_attn_weights=False)
        output, weights = func(*args, **options)
        returned = None
        if wanted:
            returned = weights.mean(dim=-3) if averaged else weights
        return (output, returned), weights
    result = func(*args, **kwargs)
    if func is functional.scaled_dot_product_attention:
        return result, _sdpa_weights(dict(zip(_SDPA_PARAMS, args, strict=False)) | kwargs)
    return result, None


def _sdpa_weights(call: dict[str, Any]) -> torch.Tensor | None:
    """Recompute the weights of ``scaled_dot_product_attention``."""
    query, key, mask = call["query"], call["key"], call.get("attn_mask")
    scale = call.get("scale")
    try:
        factor = 1.0 / math.sqrt(query.shape[-1]) if scale is None else scale
        scores = query @ key.transpose(-2, -1) * factor
        if call.get("is_causal"):
            causal = torch.ones(scores.shape[-2:], dtype=torch.bool, device=scores.device).tril()
            scores = scores.masked_fill(~causal, -math.inf)
        if mask is not None and mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -math.inf)
        elif mask is not None:
            scores = scores + mask
        return scores.softmax(dim=-1)
    except RuntimeError:
        return None


def _format_call(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Render the non-tensor arguments of a call, e.g. ``0, 3, 2, 1``."""
    parts = [_format(value) for value in args if not _only_tensors(value)]
    parts += [
        f"{key}={_format(value)}"
        for key, value in kwargs.items()
        if value is not None and not _only_tensors(value)
    ]
    return ", ".join(parts)[:_MAX_TEXT]


def _only_tensors(value: Any) -> bool:
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_only_tensors(item) for item in value)
    return isinstance(value, torch.Tensor)


def _format(value: Any) -> str:
    if isinstance(value, slice):
        bounds = ["" if part is None else str(part) for part in (value.start, value.stop)]
        return ":".join(bounds if value.step is None else [*bounds, str(value.step)])
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(_format(item) for item in value) + ")"
    if value is Ellipsis:
        return "..."
    if isinstance(value, torch.Tensor):
        return "tensor"
    if isinstance(value, (bool, int, float, str, type(None), torch.dtype, torch.device)):
        return str(value)
    return type(value).__name__
