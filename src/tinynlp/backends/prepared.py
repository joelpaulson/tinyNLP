"""Prepared KernelPlan execution for repeated CPU evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tinynlp.backends.kernel_plan import KernelPlan
from tinynlp.backends.reference import EvaluationError
from tinynlp.ir import NodeId, OpKind


@dataclass(frozen=True)
class PreparedKernelVariable:
    """Prepared variable load slot."""

    slot: int
    node_id: NodeId
    name: str


@dataclass(frozen=True)
class PreparedKernelConstant:
    """Prepared constant slot."""

    slot: int
    node_id: NodeId
    value: float


@dataclass(frozen=True)
class PreparedKernelStep:
    """Prepared executable step with slot-indexed inputs."""

    slot: int
    node_id: NodeId
    op: OpKind
    inputs: tuple[int, ...]


@dataclass(frozen=True)
class PreparedKernel:
    """Slot-indexed executable representation of one KernelPlan."""

    output_node: NodeId
    output_slot: int
    variables: tuple[PreparedKernelVariable, ...]
    constants: tuple[PreparedKernelConstant, ...]
    steps: tuple[PreparedKernelStep, ...]
    slot_count: int


class PreparedKernelBackend:
    """Dependency-free CPU backend for prepared KernelPlan execution."""

    name = "prepared-python"

    def execute(self, plan: KernelPlan, values: Mapping[str, float]) -> float:
        """Execute a KernelPlan by preparing it first."""

        return self.execute_prepared(prepare_kernel(plan), values)

    def execute_prepared(
        self,
        kernel: PreparedKernel,
        values: Mapping[str, float],
    ) -> float:
        """Execute a prepared kernel with explicit runtime values."""

        slots = [0.0] * kernel.slot_count
        for variable in kernel.variables:
            slots[variable.slot] = _variable_value(
                variable.node_id,
                variable.name,
                values,
            )
        for constant in kernel.constants:
            slots[constant.slot] = constant.value
        for step in kernel.steps:
            slots[step.slot] = _evaluate_prepared_step(step, slots)
        return slots[kernel.output_slot]


def prepare_kernel(plan: KernelPlan) -> PreparedKernel:
    """Prepare a KernelPlan for repeated slot-indexed execution."""

    slot_by_node: dict[NodeId, int] = {}
    variables: list[PreparedKernelVariable] = []
    constants: list[PreparedKernelConstant] = []
    steps: list[PreparedKernelStep] = []

    for node_id, name in plan.variables:
        slot = _allocate_slot(slot_by_node, node_id)
        variables.append(PreparedKernelVariable(slot=slot, node_id=node_id, name=name))
    for node_id, value in plan.constants:
        slot = _allocate_slot(slot_by_node, node_id)
        constants.append(
            PreparedKernelConstant(slot=slot, node_id=node_id, value=value)
        )
    for step in plan.steps:
        slot = _allocate_slot(slot_by_node, step.node_id)
        try:
            input_slots = tuple(slot_by_node[node_id] for node_id in step.inputs)
        except KeyError as exc:
            msg = f"node {step.node_id} input {exc.args[0]} has not been prepared"
            raise EvaluationError(msg) from exc
        steps.append(
            PreparedKernelStep(
                slot=slot,
                node_id=step.node_id,
                op=step.op,
                inputs=input_slots,
            )
        )

    try:
        output_slot = slot_by_node[plan.output]
    except KeyError as exc:
        msg = f"output node {plan.output} has not been prepared"
        raise EvaluationError(msg) from exc

    return PreparedKernel(
        output_node=plan.output,
        output_slot=output_slot,
        variables=tuple(variables),
        constants=tuple(constants),
        steps=tuple(steps),
        slot_count=len(slot_by_node),
    )


def _allocate_slot(slot_by_node: dict[NodeId, int], node_id: NodeId) -> int:
    if node_id in slot_by_node:
        msg = f"node {node_id} appears more than once in KernelPlan"
        raise EvaluationError(msg)
    slot = len(slot_by_node)
    slot_by_node[node_id] = slot
    return slot


def _evaluate_prepared_step(step: PreparedKernelStep, slots: list[float]) -> float:
    if step.op is OpKind.NEG:
        return -_input_value(step, slots, 0)
    if step.op is OpKind.ADD:
        return _input_value(step, slots, 0) + _input_value(step, slots, 1)
    if step.op is OpKind.SUB:
        return _input_value(step, slots, 0) - _input_value(step, slots, 1)
    if step.op is OpKind.MUL:
        return _input_value(step, slots, 0) * _input_value(step, slots, 1)
    if step.op is OpKind.DIV:
        denominator = _input_value(step, slots, 1)
        if denominator == 0.0:
            msg = f"division by zero at node {step.node_id}"
            raise EvaluationError(msg)
        return _input_value(step, slots, 0) / denominator

    msg = f"unsupported operation {step.op!s} at node {step.node_id}"
    raise EvaluationError(msg)


def _variable_value(
    node_id: NodeId,
    name: str,
    values: Mapping[str, float],
) -> float:
    if name not in values:
        msg = f"missing value for variable {name!r} at node {node_id}"
        raise EvaluationError(msg)
    return float(values[name])


def _input_value(
    step: PreparedKernelStep,
    slots: list[float],
    index: int,
) -> float:
    try:
        return slots[step.inputs[index]]
    except IndexError as exc:
        msg = f"node {step.node_id} is missing input {index}"
        raise EvaluationError(msg) from exc
