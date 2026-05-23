import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.autodiff import evaluate_jacobian as evaluate_tinynlp_jacobian
from tinynlp.autodiff import jacobian
from tinynlp.bridges.casadi import (
    casadi_available,
    compare_expression,
    compare_problem_assembly,
    evaluate_expression,
    evaluate_jacobian,
    format_casadi_comparison,
)
from tinynlp.nlp import Problem


def test_casadi_bridge_import_does_not_require_casadi() -> None:
    module = importlib.import_module("tinynlp.bridges.casadi")

    assert isinstance(module.casadi_available(), bool)
    assert isinstance(casadi_available(), bool)


def test_expression_values_match_casadi_when_available() -> None:
    _require_casadi()
    examples = _load_example("canonical_expressions")
    affine = examples.affine_expression()
    quadratic = examples.quadratic_expression()

    assert evaluate_expression(affine, {"x": 2.0, "y": 4.0}) == pytest.approx(5.0)
    assert evaluate_expression(quadratic, {"x": 3.0}) == pytest.approx(16.0)

    comparison = compare_expression(quadratic, {"x": 3.0})
    assert comparison.passed is True
    assert comparison.max_error == pytest.approx(0.0)


def test_scalar_gradient_matches_casadi_when_available() -> None:
    _require_casadi()
    examples = _load_example("canonical_expressions")
    expression = examples.quadratic_expression()
    values = {"x": 3.0}

    casadi_jacobian = evaluate_jacobian([expression], values)
    tiny_jacobian = evaluate_tinynlp_jacobian(jacobian([expression]), values)

    _assert_matrix_close(casadi_jacobian, tiny_jacobian)
    _assert_matrix_close(casadi_jacobian, [[8.0]])


def test_residual_jacobian_matches_casadi_when_available() -> None:
    _require_casadi()
    examples = _load_example("canonical_expressions")
    residuals = examples.residual_expressions()
    values = {"x": 1.0, "y": 2.0}

    casadi_jacobian = evaluate_jacobian(residuals, values)
    tiny_jacobian = evaluate_tinynlp_jacobian(jacobian(residuals), values)

    _assert_matrix_close(casadi_jacobian, tiny_jacobian)
    _assert_matrix_close(casadi_jacobian, [[1.0, 1.0], [2.0, 1.0]])


def test_banded_problem_assembly_matches_casadi_when_available() -> None:
    _require_casadi()
    banded = _load_example("banded_residual_system")
    problem = Problem.from_residuals(
        banded.banded_residual_expressions(size=6),
        name="banded",
    )

    comparison = compare_problem_assembly(
        problem,
        banded.banded_values(size=6, value=2.0),
    )

    assert comparison.passed is True
    assert comparison.max_error == pytest.approx(0.0)
    assert len(comparison.residuals) == 5
    assert len(comparison.jacobian_entries) == 10


def test_chain_problem_assembly_matches_casadi_when_available() -> None:
    _require_casadi()
    chain = _load_example("chain_dynamics_problem")
    case = chain.chain_dynamics_case(horizon=3)

    comparison = compare_problem_assembly(case.problem, case.values)

    assert comparison.passed is True
    assert comparison.max_error == pytest.approx(0.0)
    assert len(comparison.residuals) == 3
    assert len(comparison.jacobian_entries) == 9


def test_casadi_comparison_reports_are_deterministic_when_available() -> None:
    _require_casadi()
    banded = _load_example("banded_residual_system")
    problem = Problem.from_residuals(
        banded.banded_residual_expressions(size=6),
        name="banded",
    )
    comparison = compare_problem_assembly(
        problem,
        banded.banded_values(size=6, value=2.0),
    )

    report = format_casadi_comparison(comparison)

    assert report == format_casadi_comparison(comparison)
    assert "CasadiProblemAssemblyComparison problem=banded" in report
    assert "passed=True" in report
    assert "tolerance=1e-09" in report
    assert "max_error=0" in report
    assert "residuals:" in report
    assert "jacobian_entries:" in report
    assert "variable=x0" in report
    assert "object at" not in report


def _require_casadi() -> ModuleType:
    return pytest.importorskip("casadi")


def _load_example(name: str) -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_matrix_close(
    actual: list[list[float]], expected: list[list[float]]
) -> None:
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert actual_row == pytest.approx(expected_row)
