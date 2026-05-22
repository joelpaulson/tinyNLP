import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.nlp import (
    build_assembly_contract,
    format_assembly_contract,
    format_residual_assembly,
    format_sparse_matrix,
    to_dense,
)
from tinynlp.solvers import format_kkt_system


def _load_chain_example() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_path / "chain_dynamics_problem.py"
    spec = importlib.util.spec_from_file_location("chain_dynamics_problem", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chain_example = _load_chain_example()


def test_chain_case_exposes_expected_dimensions_and_variable_order() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    jacobian = chain_example.assemble_chain_jacobian(case)
    kkt = chain_example.build_chain_kkt(case)

    assert case.problem.variable_dimension == 7
    assert case.problem.residual_dimension == 3
    assert [variable.name for variable in case.problem.variables] == [
        "x0",
        "x1",
        "x2",
        "x3",
        "u0",
        "u1",
        "u2",
    ]
    assert len(contract.jacobian_terms) == 9
    assert jacobian.shape == (3, 7)
    assert len(jacobian.entries) == 9
    assert kkt.shape == (10, 10)
    assert len(kkt.entries) == 25


def test_chain_residual_values_match_expected_default_point() -> None:
    residuals = chain_example.assemble_chain_residuals(
        chain_example.chain_dynamics_case(horizon=3)
    )

    assert [value.value for value in residuals.values] == pytest.approx(
        [0.115, 0.11595, 0.1168]
    )


def test_chain_jacobian_pattern_and_values_match_hand_derived_result() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    jacobian = chain_example.assemble_chain_jacobian(case)

    assert [(entry.row, entry.column) for entry in jacobian.entries] == [
        (0, 0),
        (0, 1),
        (0, 4),
        (1, 1),
        (1, 2),
        (1, 5),
        (2, 2),
        (2, 3),
        (2, 6),
    ]
    for row, expected in zip(
        to_dense(jacobian),
        _expected_chain_jacobian(case),
        strict=True,
    ):
        assert row == pytest.approx(expected)


def test_chain_objective_metric_is_zero_at_default_references() -> None:
    assert (
        chain_example.evaluate_chain_objective(
            chain_example.chain_dynamics_case(horizon=3)
        )
        == 0.0
    )


def test_chain_reports_are_deterministic_and_address_free() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    residuals = chain_example.assemble_chain_residuals(case)
    jacobian = chain_example.assemble_chain_jacobian(case)
    kkt = chain_example.build_chain_kkt(case)

    reports = [
        format_assembly_contract(contract),
        format_residual_assembly(residuals),
        format_sparse_matrix(jacobian),
        format_kkt_system(kkt),
    ]

    assert reports == [
        format_assembly_contract(contract),
        format_residual_assembly(residuals),
        format_sparse_matrix(jacobian),
        format_kkt_system(kkt),
    ]
    assert all("object at" not in report for report in reports)


def test_chain_case_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        chain_example.chain_dynamics_case(horizon=0)

    with pytest.raises(ValueError, match="horizon \\+ 1"):
        chain_example.chain_dynamics_case(horizon=3, references=(1.0, 1.1, 1.2))


def _expected_chain_jacobian(case):
    rows = []
    for index in range(case.horizon):
        values = [0.0] * case.problem.variable_dimension
        x_value = case.values[f"x{index}"]
        values[index] = -1.0 - case.dt * (case.a + (2.0 * case.c * x_value))
        values[index + 1] = 1.0
        values[case.horizon + 1 + index] = -case.dt * case.b
        rows.append(values)
    return rows
