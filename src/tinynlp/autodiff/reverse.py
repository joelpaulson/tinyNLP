"""Reverse-mode derivative construction for scalar expressions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from tinynlp.backends import KernelStep, build_kernel_plan, evaluate
from tinynlp.ir import Expr, Graph, NodeId, OpKind, VariableRef
from tinynlp.ir.analysis import variable_refs


@dataclass(frozen=True)
class GradientEntry:
    """Derivative expression for one variable."""

    variable: VariableRef
    derivative: Expr


@dataclass(frozen=True)
class DerivativeTraceEvent:
    """Visible reverse-mode contribution for one source operation."""

    source_node_id: NodeId
    source_op: OpKind
    input_node_ids: tuple[NodeId, ...]
    adjoint_node_id: NodeId
    target_node_ids: tuple[NodeId, ...]
    contribution_node_ids: tuple[NodeId, ...]


@dataclass(frozen=True)
class Gradient:
    """Scalar-output gradient represented by derivative expressions."""

    output: Expr
    entries: tuple[GradientEntry, ...]
    trace: tuple[DerivativeTraceEvent, ...]


def gradient(expr: Expr) -> Gradient:
    """Construct a visible reverse-mode gradient for a scalar expression."""

    variables = variable_refs(expr)
    if not variables:
        return Gradient(output=expr, entries=(), trace=())

    plan = build_kernel_plan(expr)
    graph = expr.graph
    adjoints: dict[NodeId, Expr] = {expr.id: graph.constant(1.0)}
    trace: list[DerivativeTraceEvent] = []

    for step in reversed(plan.steps):
        adjoint = adjoints.get(step.node_id)
        if adjoint is None:
            continue
        contributions = _step_contributions(graph, step, adjoint)
        trace.append(
            DerivativeTraceEvent(
                source_node_id=step.node_id,
                source_op=step.op,
                input_node_ids=step.inputs,
                adjoint_node_id=adjoint.id,
                target_node_ids=tuple(target for target, _ in contributions),
                contribution_node_ids=tuple(expr.id for _, expr in contributions),
            )
        )
        for target, contribution in contributions:
            adjoints[target] = _accumulate_adjoint(adjoints.get(target), contribution)

    entries = tuple(
        GradientEntry(
            variable=variable,
            derivative=adjoints.get(variable.node_id, graph.constant(0.0)),
        )
        for variable in variables
    )
    return Gradient(output=expr, entries=entries, trace=tuple(trace))


def evaluate_gradient(
    result: Gradient, values: Mapping[str, float]
) -> dict[str, float]:
    """Evaluate a gradient with explicit variable values."""

    _ensure_unique_variable_names(entry.variable for entry in result.entries)
    return {
        entry.variable.name: evaluate(entry.derivative, values)
        for entry in result.entries
    }


def format_derivative_trace(events: Iterable[DerivativeTraceEvent]) -> str:
    """Format derivative trace events with deterministic text."""

    return "\n".join(_format_derivative_event(event) for event in events)


def _step_contributions(
    graph: Graph,
    step: KernelStep,
    adjoint: Expr,
) -> tuple[tuple[NodeId, Expr], ...]:
    inputs = step.inputs
    if step.op is OpKind.NEG:
        return ((inputs[0], -adjoint),)
    if step.op is OpKind.ADD:
        return ((inputs[0], adjoint), (inputs[1], adjoint))
    if step.op is OpKind.SUB:
        return ((inputs[0], adjoint), (inputs[1], -adjoint))
    if step.op is OpKind.MUL:
        left = _expr(graph, inputs[0])
        right = _expr(graph, inputs[1])
        return ((inputs[0], adjoint * right), (inputs[1], adjoint * left))
    if step.op is OpKind.DIV:
        left = _expr(graph, inputs[0])
        right = _expr(graph, inputs[1])
        return (
            (inputs[0], adjoint / right),
            (inputs[1], -((adjoint * left) / (right * right))),
        )

    msg = f"unsupported operation for autodiff: {step.op!s}"
    raise ValueError(msg)


def _expr(graph: Graph, node_id: NodeId) -> Expr:
    return Expr(graph=graph, id=node_id)


def _accumulate_adjoint(existing: Expr | None, contribution: Expr) -> Expr:
    if existing is None:
        return contribution
    return existing + contribution


def _ensure_unique_variable_names(variables: Iterable[VariableRef]) -> None:
    seen: set[str] = set()
    for variable in variables:
        if variable.name in seen:
            msg = (
                "gradient cannot be evaluated as a dict because variable name "
                f"{variable.name!r} appears more than once"
            )
            raise ValueError(msg)
        seen.add(variable.name)


def _format_derivative_event(event: DerivativeTraceEvent) -> str:
    inputs = ", ".join(str(node_id) for node_id in event.input_node_ids)
    pairs = ", ".join(
        f"{target}<-{contribution}"
        for target, contribution in zip(
            event.target_node_ids,
            event.contribution_node_ids,
            strict=True,
        )
    )
    return (
        f"source={event.source_node_id} op={event.source_op.value} "
        f"inputs=[{inputs}] adjoint={event.adjoint_node_id} "
        f"contributions=[{pairs}]"
    )
