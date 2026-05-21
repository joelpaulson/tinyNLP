import pytest

from tinynlp.autodiff import (
    DerivativeTraceEvent,
    evaluate_jacobian,
    jacobian,
    verify_gradient,
    verify_jacobian,
)
from tinynlp.ir import Graph


def test_residual_like_jacobian_evaluates_expected_values() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    outputs = [x + y - 1, (x * x) + y - 2]

    result = jacobian(outputs)

    assert [variable.name for variable in result.variables] == ["x", "y"]
    assert evaluate_jacobian(result, {"x": 3, "y": 5}) == [
        [1.0, 1.0],
        [6.0, 1.0],
    ]


def test_jacobian_rejects_empty_outputs() -> None:
    with pytest.raises(ValueError, match="at least one expression"):
        jacobian([])


def test_jacobian_rejects_mixed_graph_outputs() -> None:
    left = Graph().variable("x")
    right = Graph().variable("y")

    with pytest.raises(ValueError, match="same graph"):
        jacobian([left, right])


def test_jacobian_rejects_non_list_tuple_outputs() -> None:
    graph = Graph()
    x = graph.variable("x")

    with pytest.raises(TypeError, match="list or tuple"):
        jacobian(expr for expr in [x])


def test_gradient_verification_passes_on_quadratic() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = (x * x) + (2 * x) + 1

    result = verify_gradient(expr, {"x": 3})

    assert result.passed is True
    assert len(result.checks) == 1
    assert result.checks[0].analytic == 8.0
    assert result.checks[0].passed is True


def test_jacobian_verification_passes_on_residual_like_outputs() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    outputs = [x + y - 1, (x * x) + y - 2]

    result = verify_jacobian(outputs, {"x": 3, "y": 5})

    assert result.passed is True
    assert [check.output_index for check in result.checks] == [0, 0, 1, 1]
    assert [check.variable.name for check in result.checks] == ["x", "y", "x", "y"]


def test_verification_failure_object_is_deterministic() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x * x * x

    result = verify_gradient(expr, {"x": 2}, step=0.1, tolerance=0.0)

    assert result.passed is False
    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.output_index == 0
    assert check.variable.name == "x"
    assert check.analytic == 12.0
    assert check.finite_difference == pytest.approx(12.01)
    assert check.error == pytest.approx(0.01)
    assert check.tolerance == 0.0
    assert check.passed is False


def test_jacobian_traces_preserve_output_row_provenance() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    outputs = [x + y - 1, x * y]

    result = jacobian(outputs)

    assert len(result.traces) == 2
    assert result.traces[0] != result.traces[1]
    assert all(
        isinstance(event, DerivativeTraceEvent)
        for row_trace in result.traces
        for event in row_trace
    )


def test_verification_requires_positive_step() -> None:
    graph = Graph()
    expr = graph.variable("x") + 1

    with pytest.raises(ValueError, match="step must be positive"):
        verify_gradient(expr, {"x": 1}, step=0.0)
