"""Turn a traced graph into a single, self-contained HTML file."""

from __future__ import annotations

import html
import json
import re
from importlib import resources
from pathlib import Path

from sourcecodescope.ir import Graph

MODES = ("flow", "explainer")
_TOKEN = re.compile(r"%%SCS_(\w+)%%")


def _asset(name: str) -> str:
    target = resources.files("sourcecodescope") / "assets"
    for part in name.split("/"):
        target = target / part
    return target.read_text(encoding="utf-8")


def _inline_script(source: str) -> str:
    """Make ``source`` safe to place inside a ``<script>`` element."""
    return source.replace("</script", "<\\/script").replace("<!--", "<\\!--")


def build_html(graph: Graph, mode: str = "explainer", depth: int = 2) -> str:
    """Return the HTML document that displays ``graph``.

    ``mode`` selects the initial view (``"flow"`` or ``"explainer"``) and
    ``depth`` how many levels of the module hierarchy start out expanded.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    config = {"mode": mode, "depth": max(0, depth)}
    parts = {
        "TITLE": html.escape(str(graph.meta.get("model", "model"))),
        "CSS": _asset("viewer.css"),
        "D3": _inline_script(_asset("vendor/d3.min.js")),
        "ELK": _inline_script(_asset("vendor/elk.bundled.js")),
        "APP": _inline_script(_asset("viewer.js")),
        "DATA": _inline_script(graph.to_json()),
        "CONFIG": _inline_script(json.dumps(config)),
    }
    return _TOKEN.sub(lambda match: parts[match.group(1)], _asset("template.html"))


def render(
    graph: Graph, *, mode: str = "explainer", out: str | Path = "model.html", depth: int = 2
) -> Path:
    """Write the HTML view of ``graph`` to ``out`` and return its path."""
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(build_html(graph, mode, depth), encoding="utf-8")
    return target
