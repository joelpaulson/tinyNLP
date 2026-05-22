import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.ir import Graph
from tinynlp.nlp import Problem
from tinynlp.solvers import (
    SensitivityError,
    SolverResult,
    SolverStatus,
    format_sensitivity,
    implicit_sensitivity,
    solve_constraints,
)


def _load_parameter_example() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "parameter_sensitivity_problem.py"
    spec = importlib.util.spec_from_file_location(
        "parameter_sensitivity_problem", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


parameter_example = _load_parameter_example()


def test_parameterized_linear_example_matches_hand_derived_sensitivity() -> None:
    case = parameter_example.parameter_sensitivity_case()

    result = implicit_sensitivity(
        case.problem,
        case.values,
        parameter=case.parameter,
        solve_variables=case.solve_variables,
    )

    assert result.parameter.name == "p"
    assert result.sensitivities == pytest.approx(case.expected_sensitivities)
    assert [(entry.variable.name, entry.value) for entry in result.entries] == [
        ("x", pytest.approx(-1.0)),
        ("y", pytest.approx(2.0)),
    ]


def test_sensitivity_matches_central_finite_difference_solve() -> None:
    case = parameter_example.parameter_sensitivity_case()
    result = implicit_sensitivity(
        case.problem,
        case.values,
        parameter=case.parameter,
        solve_variables=case.solve_variables,
    )
    step = 1e-6

    plus = _solve_fixed_parameter(1.0 + step)
    minus = _solve_fixed_parameter(1.0 - step)
    finite_difference = {
        name: (float(plus[name]) - float(minus[name])) / (2.0 * step)
        for name in case.solve_variables
    }

    assert result.sensitivities == pytest.approx(finite_difference, abs=1e-6)


def test_sensitivity_trace_and_format_are_deterministic_and_address_free() -> None:
    case = parameter_example.parameter_sensitivity_case()

    result = implicit_sensitivity(
        case.problem,
        case.values,
        parameter=case.parameter,
        solve_variables=case.solve_variables,
    )
    first = format_sensitivity(result)
    second = format_sensitivity(result)

    assert first == second
    assert result.trace.parameter.name == "p"
    assert [variable.name for variable in result.trace.solve_variables] == ["x", "y"]
    assert result.trace.kkt_system.shape == (4, 4)
    assert result.trace.rhs == pytest.approx((0.0, 0.0, -1.0, 2.0))
    assert [
        (entry.row, entry.derivative, entry.rhs_value)
        for entry in result.trace.rhs_entries
    ] == [
        (0, pytest.approx(1.0), pytest.approx(-1.0)),
        (1, pytest.approx(-2.0), pytest.approx(2.0)),
    ]
    assert result.trace.kkt_solve_residual_norm == pytest.approx(0.0)
    assert "SensitivityResult parameter=p@" in first
    assert "kkt_shape=(4, 4)" in first
    assert "rhs_entries:" in first
    assert "object at" not in first


def test_successful_solver_result_can_provide_solution_values() -> None:
    case = parameter_example.parameter_sensitivity_case()
    solution = solve_constraints(case.problem, case.values)

    result = implicit_sensitivity(
        case.problem,
        solution,
        parameter=case.parameter,
        solve_variables=case.solve_variables,
    )

    assert solution.success
    assert result.sensitivities == pytest.approx(case.expected_sensitivities)


def test_non_success_solver_result_is_rejected() -> None:
    case = parameter_example.parameter_sensitivity_case()
    solution = SolverResult(
        status=SolverStatus.MAX_ITERATIONS,
        message="not converged",
        final_values=dict(case.values),
        trace=(),
    )

    with pytest.raises(SensitivityError, match="successful SolverResult"):
        implicit_sensitivity(case.problem, solution, parameter=case.parameter)


def test_sensitivity_rejects_invalid_parameter_and_solve_variable_inputs() -> None:
    case = parameter_example.parameter_sensitivity_case()

    missing_parameter = dict(case.values)
    del missing_parameter["p"]
    with pytest.raises(SensitivityError, match="missing value.*'p'"):
        implicit_sensitivity(case.problem, missing_parameter, parameter="p")

    with pytest.raises(SensitivityError, match="unknown sensitivity parameter"):
        implicit_sensitivity(case.problem, case.values, parameter="q")

    with pytest.raises(SensitivityError, match="unknown solve_variables"):
        implicit_sensitivity(
            case.problem,
            case.values,
            parameter="p",
            solve_variables=("x", "q"),
        )

    with pytest.raises(SensitivityError, match="duplicate"):
        implicit_sensitivity(
            case.problem,
            case.values,
            parameter="p",
            solve_variables=("x", "x"),
        )

    with pytest.raises(SensitivityError, match="must not include"):
        implicit_sensitivity(
            case.problem,
            case.values,
            parameter="p",
            solve_variables=("p",),
        )

    with pytest.raises(SensitivityError, match="at least one"):
        implicit_sensitivity(
            case.problem,
            case.values,
            parameter="p",
            solve_variables=(),
        )

    graph = Graph()
    p = graph.variable("p")
    parameter_only = Problem.from_residuals([p])
    with pytest.raises(SensitivityError, match="at least one"):
        implicit_sensitivity(parameter_only, {"p": 0.0}, parameter="p")


def test_sensitivity_rejects_unsatisfied_residuals_and_bad_numeric_values() -> None:
    case = parameter_example.parameter_sensitivity_case()

    with pytest.raises(SensitivityError, match="residual norm exceeds tolerance"):
        implicit_sensitivity(
            case.problem,
            {"x": 0.0, "y": 2.0, "p": 1.0},
            parameter="p",
        )

    with pytest.raises(SensitivityError, match="must be numeric"):
        implicit_sensitivity(
            case.problem,
            {"x": 2.0, "y": 2.0, "p": "1.0"},
            parameter="p",
        )

    with pytest.raises(SensitivityError, match="residual_tolerance"):
        implicit_sensitivity(
            case.problem,
            case.values,
            parameter="p",
            residual_tolerance=True,
        )


def test_singular_sensitivity_kkt_fails_clearly() -> None:
    graph = Graph()
    x = graph.variable("x")
    p = graph.variable("p")
    problem = Problem.from_residuals([(x - x) + p])

    with pytest.raises(SensitivityError, match="failed to solve sensitivity KKT"):
        implicit_sensitivity(
            problem,
            {"x": 0.0, "p": 0.0},
            parameter="p",
            solve_variables=("x",),
        )


def test_structurally_absent_parameter_column_gives_zero_sensitivity() -> None:
    graph = Graph()
    x = graph.variable("x")
    p = graph.variable("p")
    problem = Problem.from_residuals([x - 1], objective=p * p)

    result = implicit_sensitivity(
        problem,
        {"x": 1.0, "p": 3.0},
        parameter="p",
    )

    assert result.sensitivities == pytest.approx({"x": 0.0})
    assert result.trace.rhs == pytest.approx((0.0, -0.0))
    assert result.trace.rhs_entries[0].provenance is None
    assert "structural-zero" in format_sensitivity(result)


def test_parameter_sensitivity_example_runs_directly_from_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "examples" / "parameter_sensitivity_problem.py"

    result = subprocess.run(
        [sys.executable, str(module_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "parameter_sensitivity parameter=p" in result.stdout
    assert "sensitivity=-1" in result.stdout
    assert "sensitivity=2" in result.stdout


def _solve_fixed_parameter(parameter_value: float) -> dict[str, object]:
    problem, values = parameter_example.solve_only_problem(parameter_value)
    values["x"] = 2.0
    values["y"] = 2.0
    result = solve_constraints(problem, values, max_iterations=3)
    assert result.success
    return result.final_values
