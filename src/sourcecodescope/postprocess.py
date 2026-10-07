"""Clean-ups applied to a raw trace before it is rendered."""

from __future__ import annotations

import re

from sourcecodescope.ir import Graph, Node

_INDEX = re.compile(r"(?:(?<=\.)|^)\d+$")
_MIN_REPEAT = 2


def finalize(graph: Graph) -> None:
    """Drop noise and bundle repeated blocks, in place."""
    prune_isolated(graph)
    group_repeats(graph)


def prune_isolated(graph: Graph) -> None:
    """Remove operations that are not connected to anything."""
    connected = {edge.src for edge in graph.edges} | {edge.dst for edge in graph.edges}
    graph.nodes = [node for node in graph.nodes if node.kind != "op" or node.id in connected]


def group_repeats(graph: Graph) -> None:
    """Wrap runs of identical sibling blocks (``blocks.0``, ``blocks.1`` ...) in a repeat node.

    Only blocks that were called back to back are bundled, so a bundle never
    introduces a cycle when it is drawn collapsed.
    """
    parents = {node.parent for node in graph.nodes}
    next_id = max((node.id for node in graph.nodes), default=-1) + 1
    result: list[Node] = []
    runs: dict[int | None, list[Node]] = {}

    def close(parent: int | None) -> None:
        nonlocal next_id
        run = runs.pop(parent, [])
        if len(run) < _MIN_REPEAT:
            return
        group = _make_group(next_id, run)
        next_id += 1
        result.insert(result.index(run[0]), group)
        for member in run:
            member.parent = group.id

    for node in graph.nodes:
        run = runs.get(node.parent, [])
        repeatable = node.kind == "module" and node.id in parents and _INDEX.search(node.path)
        if run and not (repeatable and _same_series(run[-1], node)):
            close(node.parent)
        if repeatable:
            runs.setdefault(node.parent, []).append(node)
        result.append(node)
    for parent in list(runs):
        close(parent)
    graph.nodes = result


def _stem(path: str) -> str:
    return _INDEX.sub("", path)


def _same_series(first: Node, second: Node) -> bool:
    return first.type == second.type and _stem(first.path) == _stem(second.path)


def _make_group(node_id: int, run: list[Node]) -> Node:
    first = run[0]
    stem = _stem(first.path).rstrip(".")
    return Node(
        id=node_id,
        kind="repeat",
        name=stem.rsplit(".", 1)[-1] or first.type,
        path=stem,
        type=first.type,
        category="repeat",
        parent=first.parent,
        params=sum(member.params for member in run),
        count=len(run),
        inputs=list(first.inputs),
        outputs=list(run[-1].outputs),
    )
