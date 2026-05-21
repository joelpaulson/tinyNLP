"""Deterministic kernel plans for expression evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from tinynlp.ir import Expr, Node, NodeId, OpKind


@dataclass(frozen=True)
class KernelStep:
    """One executable operation in a kernel plan."""

    node_id: NodeId
    op: OpKind
    inputs: tuple[NodeId, ...]


@dataclass(frozen=True)
class KernelPlanStats:
    """Small, deterministic plan statistics."""

    operation_counts: tuple[tuple[OpKind, int], ...]
    temporary_count: int

    def count(self, op: OpKind) -> int:
        return dict(self.operation_counts).get(op, 0)


@dataclass(frozen=True)
class KernelPlan:
    """Visible execution plan for a supported expression graph."""

    output: NodeId
    variables: tuple[tuple[NodeId, str], ...]
    constants: tuple[tuple[NodeId, float], ...]
    steps: tuple[KernelStep, ...]
    stats: KernelPlanStats


def build_kernel_plan(expr: Expr) -> KernelPlan:
    """Build a deterministic kernel plan for an expression."""

    reachable = _reachable_node_ids(expr)
    ordered_nodes = tuple(node for node in expr.graph.nodes if node.id in reachable)
    variables: list[tuple[NodeId, str]] = []
    constants: list[tuple[NodeId, float]] = []
    steps: list[KernelStep] = []
    counts: Counter[OpKind] = Counter()

    for node in ordered_nodes:
        counts[node.op] += 1
        if node.op is OpKind.VARIABLE:
            variables.append((node.id, _variable_name(node)))
        elif node.op is OpKind.CONSTANT:
            constants.append((node.id, _constant_value(node)))
        else:
            steps.append(KernelStep(node_id=node.id, op=node.op, inputs=node.inputs))

    return KernelPlan(
        output=expr.id,
        variables=tuple(variables),
        constants=tuple(constants),
        steps=tuple(steps),
        stats=KernelPlanStats(
            operation_counts=tuple(
                (op, counts[op]) for op in OpKind if counts.get(op, 0) > 0
            ),
            temporary_count=len(steps),
        ),
    )


def format_kernel_plan(plan: KernelPlan) -> str:
    """Render a kernel plan with stable, human-readable text."""

    lines = [f"KernelPlan output={plan.output}"]
    lines.append("variables:")
    lines.extend(f"  {node_id}: {name}" for node_id, name in plan.variables)
    lines.append("constants:")
    lines.extend(f"  {node_id}: {value:g}" for node_id, value in plan.constants)
    lines.append("steps:")
    lines.extend(_format_step(step) for step in plan.steps)
    lines.append("counts:")
    lines.extend(f"  {op.value}: {count}" for op, count in plan.stats.operation_counts)
    lines.append(f"temporaries: {plan.stats.temporary_count}")
    return "\n".join(lines)


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


def _variable_name(node: Node) -> str:
    if node.name is None:
        msg = f"variable node {node.id} is missing a name"
        raise ValueError(msg)
    return node.name


def _constant_value(node: Node) -> float:
    if node.value is None:
        msg = f"constant node {node.id} is missing a literal value"
        raise ValueError(msg)
    return node.value


def _format_step(step: KernelStep) -> str:
    inputs = ", ".join(str(node_id) for node_id in step.inputs)
    return f"  {step.node_id}: {step.op.value} inputs=[{inputs}]"
