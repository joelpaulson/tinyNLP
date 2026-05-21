"""Internal deterministic IR analysis helpers."""

from __future__ import annotations

from collections.abc import Sequence

from tinynlp.ir.core import Expr, Graph, Node, NodeId, OpKind, VariableRef


def reachable_node_ids(expr: Expr) -> set[NodeId]:
    """Return node ids reachable from an expression."""

    reachable: set[NodeId] = set()
    pending = [expr.id]
    while pending:
        node_id = pending.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        pending.extend(expr.graph.node(node_id).inputs)
    return reachable


def reachable_nodes(expr: Expr) -> tuple[Node, ...]:
    """Return reachable nodes in graph insertion order."""

    reachable = reachable_node_ids(expr)
    return tuple(node for node in expr.graph.nodes if node.id in reachable)


def variable_refs(expr: Expr) -> tuple[VariableRef, ...]:
    """Return reachable variable references in graph insertion order."""

    return tuple(
        VariableRef(name=_variable_name(node), node_id=node.id)
        for node in reachable_nodes(expr)
        if node.op is OpKind.VARIABLE
    )


def variable_refs_for_expressions(
    expressions: Sequence[Expr],
) -> tuple[VariableRef, ...]:
    """Return variables reachable from expressions in graph insertion order."""

    graph = require_non_empty_same_graph(expressions)
    reachable: set[NodeId] = set()
    for expr in expressions:
        reachable.update(reachable_node_ids(expr))

    return tuple(
        VariableRef(name=_variable_name(node), node_id=node.id)
        for node in graph.nodes
        if node.id in reachable and node.op is OpKind.VARIABLE
    )


def require_non_empty_same_graph(expressions: Sequence[Expr]) -> Graph:
    """Return the shared graph for a non-empty expression sequence."""

    if not isinstance(expressions, list | tuple):
        msg = "outputs must be a list or tuple of Expr objects"
        raise TypeError(msg)
    if not expressions:
        msg = "outputs must contain at least one expression"
        raise ValueError(msg)

    graph = expressions[0].graph
    for expr in expressions:
        if expr.graph is not graph:
            msg = "all expressions must belong to the same graph"
            raise ValueError(msg)
    return graph


def _variable_name(node: Node) -> str:
    if node.name is None:
        msg = f"variable node {node.id} is missing a name"
        raise ValueError(msg)
    return node.name
