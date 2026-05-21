import pytest

from tinynlp.backends.reference import EvaluationError, evaluate
from tinynlp.ir import Graph


def test_evaluates_scalar_expression() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = ((x + 2) * y) / 4

    assert evaluate(expr, {"x": 6, "y": 3}) == 6.0


def test_values_are_separate_from_symbolic_graph() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x * x + 1
    before = graph.nodes

    assert evaluate(expr, {"x": 2}) == 5.0
    assert evaluate(expr, {"x": 3}) == 10.0
    assert graph.nodes == before
    assert x.node.value is None


def test_missing_variable_raises_deterministic_error() -> None:
    graph = Graph()
    expr = graph.variable("x") + 1

    with pytest.raises(EvaluationError, match="missing value for variable 'x'"):
        evaluate(expr, {})


def test_int_and_float_constants_coerce_to_float() -> None:
    graph = Graph()
    expr = graph.constant(2) + graph.constant(0.5)

    result = evaluate(expr, {})

    assert result == 2.5
    assert isinstance(result, float)


def test_division_by_zero_raises_evaluation_error() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x / 0

    with pytest.raises(EvaluationError, match="division by zero at node 2"):
        evaluate(expr, {"x": 1})
