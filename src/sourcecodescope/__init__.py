"""SourceCodeScope: visualise any PyTorch model as an interactive HTML page."""

from sourcecodescope.api import visualize
from sourcecodescope.ir import Graph
from sourcecodescope.render import render
from sourcecodescope.tracer import TraceOptions, trace

__all__ = ["Graph", "TraceOptions", "render", "trace", "visualize"]
__version__ = "0.1.0"
