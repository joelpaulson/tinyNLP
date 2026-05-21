"""Deterministic trace helpers for tinyNLP expressions."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from tinynlp.ir import Expr, Node, NodeId, OpKind


@dataclass(frozen=True)
class TraceEvent:
    """Stable structural trace event for one IR node."""

    node_id: NodeId
    op: OpKind
    inputs: tuple[NodeId, ...]
    name: str | None = None
    value: float | None = None


def trace_expression(expr: Expr) -> list[TraceEvent]:
    """Return deterministic structural trace events for an expression."""

    reachable = _reachable_node_ids(expr)
    return [_trace_node(node) for node in expr.graph.nodes if node.id in reachable]


def format_trace(events: Iterable[TraceEvent]) -> str:
    """Format trace events without object addresses or unordered metadata."""

    return "\n".join(_format_event(event) for event in events)


def _reachable_node_ids(expr: Expr) -> set[NodeId]:
    reachable: set[NodeId] = set()
    pending = [expr.id]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(expr.graph.node(node_id).inputs)
    return reachable


def _trace_node(node: Node) -> TraceEvent:
    return TraceEvent(
        node_id=node.id,
        op=node.op,
        inputs=node.inputs,
        name=node.name,
        value=node.value,
    )


def _format_event(event: TraceEvent) -> str:
    inputs = ", ".join(str(node_id) for node_id in event.inputs)
    parts = [f"{event.node_id}: {event.op.value}", f"inputs=[{inputs}]"]
    if event.name is not None:
        parts.append(f"name={event.name}")
    if event.value is not None:
        parts.append(f"value={event.value:g}")
    return " ".join(parts)
