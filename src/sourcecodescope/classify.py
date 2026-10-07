"""Map modules and operations to display categories."""

from __future__ import annotations

from torch import nn

# torch.nn modules that only orchestrate children; everything else in
# torch.nn is drawn as a single leaf node.
_CONTAINERS: tuple[type[nn.Module], ...] = (
    nn.Sequential,
    nn.ModuleList,
    nn.ModuleDict,
    nn.Transformer,
    nn.TransformerEncoder,
    nn.TransformerDecoder,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
)


def _names(text: str) -> frozenset[str]:
    return frozenset(text.split())


_BY_SOURCE = {
    "conv": "conv",
    "linear": "linear",
    "batchnorm": "norm",
    "instancenorm": "norm",
    "normalization": "norm",
    "rnn": "rnn",
    "activation": "activation",
    "pooling": "pool",
    "dropout": "dropout",
    "padding": "pad",
    "sparse": "embedding",
    "flatten": "shape",
    "fold": "shape",
    "pixelshuffle": "shape",
    "channelshuffle": "shape",
    "upsampling": "resample",
}

_OP_GROUPS: dict[str, frozenset[str]] = {
    "shape": _names(
        "permute view reshape transpose cat concat concatenate stack squeeze unsqueeze "
        "flatten unflatten expand expand_as repeat chunk split unbind getitem setitem select "
        "narrow roll flip movedim moveaxis swapaxes t T mT H view_as reshape_as pad "
        "index_select gather tile repeat_interleave hstack vstack masked_fill where tril "
        "triu view_as_real view_as_complex real imag"
    ),
    "activation": _names(
        "relu relu6 gelu silu sigmoid tanh softmax log_softmax leaky_relu elu selu prelu "
        "softplus hardtanh mish hardswish hardsigmoid logsigmoid softsign glu celu"
    ),
    "norm": _names("layer_norm batch_norm group_norm instance_norm normalize rms_norm"),
    "conv": _names("conv1d conv2d conv3d conv_transpose1d conv_transpose2d conv_transpose3d"),
    "linear": _names("linear bilinear"),
    "dropout": _names("dropout dropout1d dropout2d dropout3d alpha_dropout"),
    "embedding": _names("embedding embedding_bag one_hot"),
    "resample": _names("interpolate upsample grid_sample pixel_shuffle"),
    "attention": _names("scaled_dot_product_attention multi_head_attention_forward"),
}


def is_leaf_module(module: nn.Module) -> bool:
    """Return True when ``module`` should be drawn without its internals."""
    builtin = type(module).__module__.startswith("torch.nn.")
    return builtin and not isinstance(module, _CONTAINERS)


def module_category(module: nn.Module) -> str:
    """Return the display category of a module instance."""
    if isinstance(module, nn.MultiheadAttention):
        return "attention"
    if isinstance(module, nn.Identity):
        return "other"
    source = type(module).__module__
    if source.startswith("torch.nn.modules."):
        return _BY_SOURCE.get(source.rsplit(".", 1)[1], "module")
    lowered = type(module).__name__.lower()
    if "attention" in lowered or "attn" in lowered:
        return "attention"
    return "module"


def op_category(name: str) -> str:
    """Return the display category of a traced operation name."""
    base = name.rstrip("_") if not name.startswith("_") else name
    for category, names in _OP_GROUPS.items():
        if base in names:
            return category
    if "pool" in base:
        return "pool"
    return "math"
