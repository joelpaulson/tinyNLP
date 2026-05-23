import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.ir import Graph
from tinynlp.nlp import Problem
from tinynlp.solvers import (
    LeastSquaresError,
    LeastSquaresStatus,
    format_least_squares_trace,
    format_normal_equations,
    solve_least_squares,
)


def _load_flagship_example() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "flagship_chain_modeling.py"
    spec = importlib.util.spec_from_file_location(
        "flagship_chain_modeling",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


flagship_example = _load_flagship_example()


def test_linear_residual_converges_with_visible_normal_equations() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([x - 3.0])

    result = solve_least_squares(problem, {"x": 0.0}, regularization=0.0)

    assert result.success
    assert result.status is LeastSquaresStatus.CONVERGED
    assert result.final_values["x"] == pytest.approx(3.0)
    assert len(result.trace) == 1

    record = result.trace[0]
    assert record.previous_residual_norm == pytest.approx(3.0)
    assert record.residual_norm == pytest.approx(0.0)
    assert record.least_squares_value == pytest.approx(0.0)
    assert record.gradient_norm == pytest.approx(3.0)
    assert record.step_norm == pytest.approx(3.0)
    assert record.accepted_step_length == 1.0
    assert record.linear_solve_residual_norm == pytest.approx(0.0)
    assert record.jacobian_shape == (1, 1)
    assert record.normal_equation_shape == (1, 1)
    assert record.normal_equations is not None
    assert record.normal_equations.coefficients == ((1.0,),)
    assert record.normal_equations.rhs == (3.0,)


def test_flagship_chain_reduces_residual_and_reports_objective_metric() -> None:
    case = flagship_example.flagship_chain_case(horizon=3)

    result = solve_least_squares(case.problem, case.values, max_iterations=8)

    assert result.success
    assert result.status is LeastSquaresStatus.CONVERGED
    assert result.trace[0].previous_residual_norm is not None
    assert result.trace[-1].residual_norm < result.trace[0].previous_residual_norm
    assert result.trace[-1].least_squares_value < (
        0.5 * result.trace[0].previous_residual_norm ** 2
    )
    assert all(record.tracked_objective_value is not None for record in result.trace)
    assert [value.name for value in result.trace[0].variables] == [
        "x0",
        "x1",
        "x2",
        "x3",
        "u0",
        "u1",
        "u2",
    ]


def test_least_squares_is_deterministic_and_preserves_input_values() -> None:
    case = flagship_example.flagship_chain_case(horizon=3)
    sentinel = object()
    values = dict(case.values)
    values["unused"] = "keep-me"
    values["sentinel"] = sentinel

    first = solve_least_squares(case.problem, values, max_iterations=8)
    second = solve_least_squares(case.problem, values, max_iterations=8)

    assert values["x0"] == case.values["x0"]
    assert values["unused"] == "keep-me"
    assert values["sentinel"] is sentinel
    assert first.final_values is not values
    assert first.final_values["unused"] == "keep-me"
    assert first.final_values["sentinel"] is sentinel
    assert first.final_values == second.final_values
    assert format_least_squares_trace(first.trace) == format_least_squares_trace(
        second.trace
    )


def test_least_squares_trace_and_normal_equation_reports_are_stable() -> None:
    case = flagship_example.flagship_chain_case(horizon=3)
    result = solve_least_squares(case.problem, case.values, max_iterations=8)
    trace_report = format_least_squares_trace(result.trace)
    normal_equations = result.trace[0].normal_equations
    assert normal_equations is not None
    normal_report = format_normal_equations(normal_equations)

    assert trace_report == format_least_squares_trace(result.trace)
    assert normal_report == format_normal_equations(normal_equations)
    assert "LeastSquaresTrace" in trace_report
    assert "previous_residual_norm=" in trace_report
    assert "least_squares_value=" in trace_report
    assert "gradient_norm=" in trace_report
    assert "linear_solve_residual_norm=" in trace_report
    assert "tracked_objective_value=" in trace_report
    assert "NormalEquationSystem" in normal_report
    assert "source_jacobian_shape=(3, 7)" in normal_report
    assert "regularization=1e-06" in normal_report
    assert "object at" not in trace_report
    assert "object at" not in normal_report


def test_least_squares_rejects_missing_values_and_invalid_options() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([x - 1.0])

    with pytest.raises(LeastSquaresError, match="missing values"):
        solve_least_squares(problem, {})
    with pytest.raises(LeastSquaresError, match="numeric"):
        solve_least_squares(problem, {"x": "bad"})
    with pytest.raises(LeastSquaresError, match="max_iterations"):
        solve_least_squares(problem, {"x": 0.0}, max_iterations=0)
    with pytest.raises(LeastSquaresError, match="max_iterations"):
        solve_least_squares(problem, {"x": 0.0}, max_iterations=True)
    with pytest.raises(LeastSquaresError, match="residual_tolerance"):
        solve_least_squares(problem, {"x": 0.0}, residual_tolerance="1e-8")
    with pytest.raises(LeastSquaresError, match="step_tolerance"):
        solve_least_squares(problem, {"x": 0.0}, step_tolerance=True)
    with pytest.raises(LeastSquaresError, match="gradient_tolerance"):
        solve_least_squares(problem, {"x": 0.0}, gradient_tolerance=0.0)
    with pytest.raises(LeastSquaresError, match="regularization"):
        solve_least_squares(problem, {"x": 0.0}, regularization=-1.0)
    with pytest.raises(LeastSquaresError, match="damping_steps"):
        solve_least_squares(problem, {"x": 0.0}, damping_steps=())
    with pytest.raises(LeastSquaresError, match="interval"):
        solve_least_squares(problem, {"x": 0.0}, damping_steps=(1.5,))
    with pytest.raises(LeastSquaresError, match="numeric"):
        solve_least_squares(problem, {"x": 0.0}, damping_steps=("0.5",))


def test_line_search_failure_returns_non_success_status() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([x - 1.0])

    result = solve_least_squares(
        problem,
        {"x": 0.0},
        damping_steps=(1e-300,),
    )

    assert not result.success
    assert result.status is LeastSquaresStatus.LINE_SEARCH_FAILED
    assert "no damping step" in result.message
    assert result.trace[0].step_norm is not None
    assert result.trace[0].accepted_step_length is None


def test_singular_normal_equations_raise_clear_error() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    problem = Problem.from_residuals([x + y - 1.0])

    with pytest.raises(LeastSquaresError, match="singular normal-equation"):
        solve_least_squares(problem, {"x": 0.0, "y": 0.0}, regularization=0.0)
