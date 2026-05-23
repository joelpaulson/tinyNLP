import pytest

from tinynlp.backends import (
    EvaluationError,
    PreparedKernelBackend,
    build_kernel_plan,
    get_backend,
    prepare_kernel,
)
from tinynlp.ir import Graph


def test_prepared_kernel_matches_reference_backend_for_supported_ops() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = (((x + 2) * (y - 3)) / (x - y)) + (-x)
    plan = build_kernel_plan(expr)
    values = {"x": 5.0, "y": 1.5}

    reference = get_backend("python").execute(plan, values)
    prepared = PreparedKernelBackend().execute_prepared(prepare_kernel(plan), values)

    assert prepared == pytest.approx(reference)


def test_prepared_backend_is_registered_by_name() -> None:
    backend = get_backend("prepared-python")

    assert backend.name == "prepared-python"


def test_prepared_kernel_reports_missing_variables_deterministically() -> None:
    graph = Graph()
    x = graph.variable("x")
    plan = build_kernel_plan(x + 1)
    kernel = prepare_kernel(plan)

    with pytest.raises(EvaluationError, match="missing value for variable 'x'"):
        PreparedKernelBackend().execute_prepared(kernel, {})


def test_prepared_kernel_reports_division_by_zero_deterministically() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x / (x - x)
    plan = build_kernel_plan(expr)
    kernel = prepare_kernel(plan)

    with pytest.raises(EvaluationError, match="division by zero at node"):
        PreparedKernelBackend().execute_prepared(kernel, {"x": 2.0})


def test_prepared_kernel_handles_variable_and_constant_outputs() -> None:
    graph = Graph()
    x = graph.variable("x")
    constant = graph.constant(7.0)

    backend = PreparedKernelBackend()

    assert (
        backend.execute_prepared(prepare_kernel(build_kernel_plan(x)), {"x": 4.0})
        == 4.0
    )
    assert (
        backend.execute_prepared(
            prepare_kernel(build_kernel_plan(constant)),
            {"x": 4.0},
        )
        == 7.0
    )
