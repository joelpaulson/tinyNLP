"""Canonical tinyNLP expression examples."""

from tinynlp.backends.reference import evaluate
from tinynlp.ir import Expr, Graph


def affine_expression() -> Expr:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    return (2 * x) + y - 3


def evaluate_affine(values: dict[str, float]) -> float:
    return evaluate(affine_expression(), values)


def quadratic_expression() -> Expr:
    graph = Graph()
    x = graph.variable("x")
    return (x * x) + (2 * x) + 1


def evaluate_quadratic(values: dict[str, float]) -> float:
    return evaluate(quadratic_expression(), values)


def residual_expressions() -> list[Expr]:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    return [x + y - 1, (x * x) + y - 2]


def evaluate_residuals(values: dict[str, float]) -> list[float]:
    return [evaluate(expr, values) for expr in residual_expressions()]
