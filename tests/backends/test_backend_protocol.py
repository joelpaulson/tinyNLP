from collections.abc import Mapping

import pytest

from tinynlp.backends import (
    EvaluationError,
    KernelPlan,
    build_kernel_plan,
    evaluate,
    get_backend,
    register_backend,
)
from tinynlp.ir import Graph


def test_python_backend_executes_kernel_plan_like_evaluate() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = ((x + 2) * y) / 4
    plan = build_kernel_plan(expr)

    backend = get_backend("python")

    assert backend.execute(plan, {"x": 6, "y": 3}) == 6.0
    assert backend.execute(plan, {"x": 6, "y": 3}) == evaluate(expr, {"x": 6, "y": 3})


def test_backend_protocol_keeps_values_separate_from_graph() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x * x + 1
    plan = build_kernel_plan(expr)
    before = graph.nodes
    backend = get_backend("python")

    assert backend.execute(plan, {"x": 2}) == 5.0
    assert backend.execute(plan, {"x": 3}) == 10.0
    assert graph.nodes == before
    assert x.node.value is None


def test_python_backend_missing_variable_error_is_deterministic() -> None:
    graph = Graph()
    expr = graph.variable("x") + 1
    plan = build_kernel_plan(expr)

    with pytest.raises(EvaluationError, match="missing value for variable 'x'"):
        get_backend("python").execute(plan, {})


def test_python_backend_division_by_zero_error_is_deterministic() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x / 0
    plan = build_kernel_plan(expr)

    with pytest.raises(EvaluationError, match="division by zero at node 2"):
        get_backend("python").execute(plan, {"x": 1})


def test_registry_rejects_unknown_backend_names() -> None:
    with pytest.raises(KeyError, match="unknown backend 'missing'"):
        get_backend("missing")


def test_registry_accepts_small_backend_protocol_implementations() -> None:
    class ConstantBackend:
        name = "constant-test"

        def execute(
            self,
            plan: KernelPlan,
            values: Mapping[str, float],
        ) -> float:
            return float(len(plan.steps) + len(values))

    graph = Graph()
    expr = graph.variable("x") + 1
    plan = build_kernel_plan(expr)

    register_backend(ConstantBackend())

    assert get_backend("constant-test").execute(plan, {"x": 2}) == 2.0
