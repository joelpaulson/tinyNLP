import pytest

from tinynlp.autodiff import (
    DerivativeTraceEvent,
    evaluate_gradient,
    format_derivative_trace,
    gradient,
)
from tinynlp.backends import evaluate
from tinynlp.ir import Graph, NodeId


def test_affine_gradient_matches_hand_derived_values() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = (2 * x) + y - 3

    result = gradient(expr)

    assert evaluate_gradient(result, {"x": 4, "y": 5}) == {"x": 2.0, "y": 1.0}


def test_quadratic_gradient_matches_hand_derived_value() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = (x * x) + (2 * x) + 1

    result = gradient(expr)

    assert evaluate_gradient(result, {"x": 3}) == {"x": 8.0}


def test_product_negation_subtraction_and_division_gradients() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = ((x * -y) - x) / y

    result = gradient(expr)

    assert evaluate_gradient(result, {"x": 6, "y": 3}) == pytest.approx(
        {
            "x": -4.0 / 3.0,
            "y": 2.0 / 3.0,
        }
    )


def test_gradient_values_are_separate_from_symbolic_graph() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = (x * x) + 1
    original_nodes = graph.nodes

    result = gradient(expr)

    assert graph.nodes[: len(original_nodes)] == original_nodes
    assert x.node.value is None
    assert evaluate_gradient(result, {"x": 2}) == {"x": 4.0}
    assert evaluate_gradient(result, {"x": 3}) == {"x": 6.0}
    assert evaluate(expr, {"x": 3}) == 10.0


def test_gradient_variable_order_is_deterministic_and_excludes_unused() -> None:
    graph = Graph()
    _unused = graph.variable("unused")
    y = graph.variable("y")
    x = graph.variable("x")
    expr = (x * y) + x

    result = gradient(expr)

    assert [
        (entry.variable.name, entry.variable.node_id) for entry in result.entries
    ] == [
        ("y", NodeId(1)),
        ("x", NodeId(2)),
    ]
    assert evaluate_gradient(result, {"x": 2, "y": 5}) == {"y": 2.0, "x": 6.0}


def test_constant_output_has_empty_gradient() -> None:
    graph = Graph()
    expr = graph.constant(3)

    result = gradient(expr)

    assert result.entries == ()
    assert result.trace == ()
    assert evaluate_gradient(result, {}) == {}


def test_derivative_trace_format_is_deterministic() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x * x

    result = gradient(expr)
    formatted = format_derivative_trace(result.trace)

    assert result.trace
    assert all(isinstance(event, DerivativeTraceEvent) for event in result.trace)
    assert formatted == "\n".join(
        [
            "source=1 op=mul inputs=[0, 0] adjoint=2 contributions=[0<-3, 0<-4]",
        ]
    )
    assert "object at" not in formatted


def test_duplicate_variable_names_fail_for_dict_evaluation() -> None:
    graph = Graph()
    left = graph.variable("x")
    right = graph.variable("x")
    result = gradient(left + right)

    with pytest.raises(ValueError, match="variable name 'x' appears more than once"):
        evaluate_gradient(result, {"x": 1})
