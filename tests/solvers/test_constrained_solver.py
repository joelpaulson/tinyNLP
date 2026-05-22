import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.ir import Graph
from tinynlp.nlp import Problem, assemble_residuals, build_assembly_contract
from tinynlp.solvers import (
    SolverError,
    SolverStatus,
    format_solver_trace,
    solve_constraints,
)


def _load_chain_example() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "chain_dynamics_problem.py"
    spec = importlib.util.spec_from_file_location("chain_dynamics_problem", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chain_example = _load_chain_example()


def test_linear_equality_problem_converges_in_one_correction() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    problem = Problem.from_residuals([x + y - 3])

    result = solve_constraints(problem, {"x": 0.0, "y": 0.0})

    assert result.success
    assert result.status is SolverStatus.CONVERGED
    assert result.final_values["x"] == pytest.approx(1.5)
    assert result.final_values["y"] == pytest.approx(1.5)
    assert len(result.trace) == 1
    assert result.trace[0].iteration == 0
    assert result.trace[0].residual_norm == pytest.approx(0.0)
    assert result.trace[0].step_norm == pytest.approx((1.5**2 + 1.5**2) ** 0.5)
    assert result.trace[0].accepted_step_length == 1.0
    assert result.trace[0].kkt_solve_residual_norm == pytest.approx(0.0)


def test_chain_dynamics_residuals_are_reduced_and_converge() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    initial_norm = _residual_norm(case.problem, case.values)

    result = solve_constraints(case.problem, case.values, max_iterations=5)

    assert result.success
    assert result.status is SolverStatus.CONVERGED
    assert result.trace[-1].residual_norm < 1e-8
    assert result.trace[-1].residual_norm < initial_norm
    assert len(result.trace) <= 5
    assert all(record.objective_value is not None for record in result.trace)
    assert [value.name for value in result.trace[0].variables] == [
        "x0",
        "x1",
        "x2",
        "x3",
        "u0",
        "u1",
        "u2",
    ]


def test_solver_trace_format_is_deterministic_and_address_free() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    result = solve_constraints(case.problem, case.values, max_iterations=5)

    first = format_solver_trace(result.trace)
    second = format_solver_trace(result.trace)

    assert first == second
    assert "SolverTrace" in first
    assert "iteration=0" in first
    assert "residual_norm=" in first
    assert "kkt_solve_residual_norm=" in first
    assert "objective_value=" in first
    assert "variables=[x0=" in first
    assert "object at" not in first


def test_solver_does_not_mutate_input_values_and_preserves_extra_values() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([x - 1])
    values = {"x": 0.0, "unused": 99.0}

    result = solve_constraints(problem, values)

    assert values == {"x": 0.0, "unused": 99.0}
    assert result.final_values["x"] == pytest.approx(1.0)
    assert result.final_values["unused"] == 99.0


def test_solver_reports_max_iterations_without_claiming_success() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)

    result = solve_constraints(
        case.problem,
        case.values,
        max_iterations=1,
        residual_tolerance=1e-16,
    )

    assert not result.success
    assert result.status is SolverStatus.MAX_ITERATIONS
    assert "max_iterations=1" in result.message
    assert len(result.trace) == 1


def test_solver_rejects_missing_values_and_invalid_options() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([x - 1])

    with pytest.raises(SolverError, match="missing values"):
        solve_constraints(problem, {})

    with pytest.raises(SolverError, match="max_iterations"):
        solve_constraints(problem, {"x": 0.0}, max_iterations=0)

    with pytest.raises(SolverError, match="residual_tolerance"):
        solve_constraints(problem, {"x": 0.0}, residual_tolerance=0.0)

    with pytest.raises(SolverError, match="damping_steps"):
        solve_constraints(problem, {"x": 0.0}, damping_steps=())

    with pytest.raises(SolverError, match="interval"):
        solve_constraints(problem, {"x": 0.0}, damping_steps=(1.5,))

    with pytest.raises(SolverError, match="numeric"):
        solve_constraints(problem, {"x": "not-a-number"})


def test_solver_raises_clear_error_for_singular_kkt_step() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([(x - x) + 1])

    with pytest.raises(SolverError, match="failed to compute constrained solver step"):
        solve_constraints(problem, {"x": 0.0})


def _residual_norm(problem: Problem, values: dict[str, float]) -> float:
    residuals = assemble_residuals(build_assembly_contract(problem), values)
    return sum(value.value * value.value for value in residuals.values) ** 0.5
