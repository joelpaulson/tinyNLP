"""CPU-first reference evaluator for tinyNLP expressions."""

from __future__ import annotations

from collections.abc import Mapping

from tinynlp.ir import Expr, Node, NodeId, OpKind


class EvaluationError(ValueError):
    """Raised when a symbolic expression cannot be evaluated."""


def evaluate(expr: Expr, values: Mapping[str, float]) -> float:
    """Evaluate an expression with explicit variable bindings."""

    results: dict[NodeId, float] = {}
    for node in expr.graph.nodes:
        results[node.id] = _evaluate_node(node, results, values)
        if node.id == expr.id:
            return results[node.id]

    msg = f"expression node {expr.id} is not present in its graph"
    raise EvaluationError(msg)


def _evaluate_node(
    node: Node,
    results: Mapping[NodeId, float],
    values: Mapping[str, float],
) -> float:
    if node.op is OpKind.VARIABLE:
        return _variable_value(node, values)
    if node.op is OpKind.CONSTANT:
        if node.value is None:
            msg = f"constant node {node.id} is missing a literal value"
            raise EvaluationError(msg)
        return node.value
    if node.op is OpKind.NEG:
        return -_input_value(node, results, 0)
    if node.op is OpKind.ADD:
        return _input_value(node, results, 0) + _input_value(node, results, 1)
    if node.op is OpKind.SUB:
        return _input_value(node, results, 0) - _input_value(node, results, 1)
    if node.op is OpKind.MUL:
        return _input_value(node, results, 0) * _input_value(node, results, 1)
    if node.op is OpKind.DIV:
        denominator = _input_value(node, results, 1)
        if denominator == 0.0:
            msg = f"division by zero at node {node.id}"
            raise EvaluationError(msg)
        return _input_value(node, results, 0) / denominator

    msg = f"unsupported operation {node.op!s} at node {node.id}"
    raise EvaluationError(msg)


def _variable_value(node: Node, values: Mapping[str, float]) -> float:
    if node.name is None:
        msg = f"variable node {node.id} is missing a name"
        raise EvaluationError(msg)
    if node.name not in values:
        msg = f"missing value for variable {node.name!r} at node {node.id}"
        raise EvaluationError(msg)
    return float(values[node.name])


def _input_value(node: Node, results: Mapping[NodeId, float], index: int) -> float:
    try:
        input_id = node.inputs[index]
        return results[input_id]
    except IndexError as exc:
        msg = f"node {node.id} is missing input {index}"
        raise EvaluationError(msg) from exc
    except KeyError as exc:
        msg = f"node {node.id} input {input_id} has not been evaluated"
        raise EvaluationError(msg) from exc
