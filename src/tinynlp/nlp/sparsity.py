"""Structural sparsity discovery for supported expressions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from tinynlp.ir import Expr, Node, NodeId, OpKind, VariableRef
from tinynlp.ir.analysis import (
    reachable_nodes,
    require_non_empty_same_graph,
    variable_refs_for_expressions,
)


@dataclass(frozen=True, order=True)
class SparsityEntry:
    """One structurally nonzero Jacobian position."""

    row: int
    column: int
    variable: VariableRef


@dataclass(frozen=True)
class SparsityTraceEvent:
    """Dependency propagation record for one reachable node."""

    row: int
    node_id: NodeId
    op: OpKind
    input_node_ids: tuple[NodeId, ...]
    dependencies: tuple[VariableRef, ...]


@dataclass(frozen=True)
class SparsityPattern:
    """Symbolic row-major Jacobian sparsity pattern."""

    shape: tuple[int, int]
    variables: tuple[VariableRef, ...]
    entries: tuple[SparsityEntry, ...]
    trace: tuple[SparsityTraceEvent, ...]


def expression_dependencies(expr: Expr) -> tuple[VariableRef, ...]:
    """Return symbolic variable dependencies for one expression."""

    dependencies, _trace = _dependencies_for_expr(expr, row=0)
    return dependencies


def jacobian_sparsity(outputs: Sequence[Expr]) -> SparsityPattern:
    """Return conservative symbolic sparsity for vector-output Jacobians."""

    require_non_empty_same_graph(outputs)
    output_tuple = tuple(outputs)
    variables = variable_refs_for_expressions(output_tuple)
    column_by_node = {
        variable.node_id: column for column, variable in enumerate(variables)
    }
    entries: list[SparsityEntry] = []
    trace: list[SparsityTraceEvent] = []

    for row, output in enumerate(output_tuple):
        dependencies, row_trace = _dependencies_for_expr(output, row=row)
        trace.extend(row_trace)
        for variable in dependencies:
            entries.append(
                SparsityEntry(
                    row=row,
                    column=column_by_node[variable.node_id],
                    variable=variable,
                )
            )

    return SparsityPattern(
        shape=(len(output_tuple), len(variables)),
        variables=variables,
        entries=tuple(entries),
        trace=tuple(trace),
    )


def format_sparsity(pattern: SparsityPattern) -> str:
    """Format a sparsity pattern and provenance trace deterministically."""

    lines = [f"SparsityPattern shape={pattern.shape}"]
    lines.append("variables:")
    lines.extend(
        f"  col={column} node={variable.node_id} name={variable.name}"
        for column, variable in enumerate(pattern.variables)
    )
    lines.append("entries:")
    lines.extend(
        "  "
        f"row={entry.row} col={entry.column} "
        f"node={entry.variable.node_id} name={entry.variable.name}"
        for entry in pattern.entries
    )
    lines.append("trace:")
    lines.extend(_format_trace_event(event) for event in pattern.trace)
    return "\n".join(lines)


def _dependencies_for_expr(
    expr: Expr,
    *,
    row: int,
) -> tuple[tuple[VariableRef, ...], tuple[SparsityTraceEvent, ...]]:
    dependencies_by_node: dict[NodeId, tuple[VariableRef, ...]] = {}
    trace: list[SparsityTraceEvent] = []

    for node in reachable_nodes(expr):
        dependencies = _node_dependencies(node, dependencies_by_node)
        dependencies_by_node[node.id] = dependencies
        trace.append(
            SparsityTraceEvent(
                row=row,
                node_id=node.id,
                op=node.op,
                input_node_ids=node.inputs,
                dependencies=dependencies,
            )
        )

    return dependencies_by_node[expr.id], tuple(trace)


def _node_dependencies(
    node: Node,
    dependencies_by_node: dict[NodeId, tuple[VariableRef, ...]],
) -> tuple[VariableRef, ...]:
    if node.op is OpKind.VARIABLE:
        return (VariableRef(name=_variable_name(node), node_id=node.id),)
    if node.op is OpKind.CONSTANT:
        return ()
    return _union_dependencies(
        dependencies_by_node[input_id] for input_id in node.inputs
    )


def _union_dependencies(
    dependency_groups: Sequence[tuple[VariableRef, ...]],
) -> tuple[VariableRef, ...]:
    by_node_id: dict[NodeId, VariableRef] = {}
    for group in dependency_groups:
        for variable in group:
            by_node_id[variable.node_id] = variable
    return tuple(by_node_id[node_id] for node_id in sorted(by_node_id))


def _variable_name(node: Node) -> str:
    if node.name is None:
        msg = f"variable node {node.id} is missing a name"
        raise ValueError(msg)
    return node.name


def _format_trace_event(event: SparsityTraceEvent) -> str:
    inputs = ", ".join(str(node_id) for node_id in event.input_node_ids)
    dependencies = ", ".join(
        _format_variable(variable) for variable in event.dependencies
    )
    return (
        f"  row={event.row} node={event.node_id} op={event.op.value} "
        f"inputs=[{inputs}] deps=[{dependencies}]"
    )


def _format_variable(variable: VariableRef) -> str:
    return f"{variable.name}@{variable.node_id}"
