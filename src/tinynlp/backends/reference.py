"""CPU-first reference evaluator for tinyNLP expressions."""

from __future__ import annotations

from collections.abc import Mapping

from tinynlp.backends.kernel_plan import KernelPlan, KernelStep, build_kernel_plan
from tinynlp.ir import Expr, NodeId, OpKind


class EvaluationError(ValueError):
    """Raised when a symbolic expression cannot be evaluated."""


class PythonReferenceBackend:
    """Reference Python backend for kernel plan execution."""

    name = "python"

    def execute(self, plan: KernelPlan, values: Mapping[str, float]) -> float:
        results: dict[NodeId, float] = {}
        for node_id, name in plan.variables:
            results[node_id] = _variable_value(node_id, name, values)
        for node_id, value in plan.constants:
            results[node_id] = value
        for step in plan.steps:
            results[step.node_id] = _evaluate_step(step, results)

        try:
            return results[plan.output]
        except KeyError as exc:
            msg = f"output node {plan.output} has not been evaluated"
            raise EvaluationError(msg) from exc


def evaluate(expr: Expr, values: Mapping[str, float]) -> float:
    """Evaluate an expression with explicit variable bindings."""

    from tinynlp.backends.registry import get_backend

    plan = build_kernel_plan(expr)
    return get_backend("python").execute(plan, values)


def _evaluate_step(step: KernelStep, results: Mapping[NodeId, float]) -> float:
    if step.op is OpKind.NEG:
        return -_input_value(step, results, 0)
    if step.op is OpKind.ADD:
        return _input_value(step, results, 0) + _input_value(step, results, 1)
    if step.op is OpKind.SUB:
        return _input_value(step, results, 0) - _input_value(step, results, 1)
    if step.op is OpKind.MUL:
        return _input_value(step, results, 0) * _input_value(step, results, 1)
    if step.op is OpKind.DIV:
        denominator = _input_value(step, results, 1)
        if denominator == 0.0:
            msg = f"division by zero at node {step.node_id}"
            raise EvaluationError(msg)
        return _input_value(step, results, 0) / denominator

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
    step: KernelStep,
    results: Mapping[NodeId, float],
    index: int,
) -> float:
    try:
        input_id = step.inputs[index]
        return results[input_id]
    except IndexError as exc:
        msg = f"node {step.node_id} is missing input {index}"
        raise EvaluationError(msg) from exc
    except KeyError as exc:
        msg = f"node {step.node_id} input {input_id} has not been evaluated"
        raise EvaluationError(msg) from exc
