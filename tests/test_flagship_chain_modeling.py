import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.nlp import (
    format_residual_assembly,
    format_sparse_matrix,
    jacobian_sparsity,
    to_dense,
)


def _load_example(name: str) -> ModuleType:
    examples_path = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


explicit_chain = _load_example("chain_dynamics_problem")
flagship_chain = _load_example("flagship_chain_modeling")


def test_flagship_chain_matches_explicit_chain_structure() -> None:
    explicit = explicit_chain.chain_dynamics_case(horizon=3)
    flagship = flagship_chain.flagship_chain_case(horizon=3)

    assert [variable.name for variable in flagship.problem.variables] == [
        variable.name for variable in explicit.problem.variables
    ]
    assert flagship.values == explicit.values
    assert flagship.references == explicit.references
    assert [block.name for block in flagship.problem.residual_blocks] == [
        "chain_dynamics"
    ]
    assert flagship.problem.variable_dimension == explicit.problem.variable_dimension
    assert flagship.problem.residual_dimension == explicit.problem.residual_dimension


def test_flagship_chain_matches_explicit_chain_values_and_jacobian() -> None:
    explicit = explicit_chain.chain_dynamics_case(horizon=3)
    flagship = flagship_chain.flagship_chain_case(horizon=3)

    explicit_residuals = explicit_chain.assemble_chain_residuals(explicit)
    flagship_residuals = flagship_chain.assemble_flagship_residuals(flagship)
    explicit_jacobian = explicit_chain.assemble_chain_jacobian(explicit)
    flagship_jacobian = flagship_chain.assemble_flagship_jacobian(flagship)

    assert [value.value for value in flagship_residuals.values] == pytest.approx(
        [value.value for value in explicit_residuals.values]
    )
    for flagship_row, explicit_row in zip(
        to_dense(flagship_jacobian),
        to_dense(explicit_jacobian),
        strict=True,
    ):
        assert flagship_row == pytest.approx(explicit_row)
    assert flagship_chain.evaluate_flagship_objective(flagship) == pytest.approx(
        explicit_chain.evaluate_chain_objective(explicit)
    )


def test_flagship_chain_sparsity_matches_explicit_chain() -> None:
    explicit = explicit_chain.chain_dynamics_case(horizon=3)
    flagship = flagship_chain.flagship_chain_case(horizon=3)

    explicit_pattern = jacobian_sparsity(explicit.problem.residuals)
    flagship_pattern = jacobian_sparsity(flagship.problem.residuals)

    assert flagship_pattern.shape == explicit_pattern.shape
    assert flagship_pattern.entries == explicit_pattern.entries
    assert [variable.name for variable in flagship_pattern.variables] == [
        variable.name for variable in explicit_pattern.variables
    ]


def test_flagship_chain_contract_report_is_deterministic_and_address_free() -> None:
    case = flagship_chain.flagship_chain_case(horizon=3)
    residuals = flagship_chain.assemble_flagship_residuals(case)
    jacobian = flagship_chain.assemble_flagship_jacobian(case)
    reports = [
        format_residual_assembly(residuals),
        format_sparse_matrix(jacobian),
    ]

    assert reports == [
        format_residual_assembly(residuals),
        format_sparse_matrix(jacobian),
    ]
    assert all("object at" not in report for report in reports)


def test_flagship_chain_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        flagship_chain.flagship_chain_case(horizon=0)
    with pytest.raises(ValueError, match="horizon \\+ 1"):
        flagship_chain.flagship_chain_case(horizon=3, references=(1.0, 1.1, 1.2))


def test_flagship_example_runs_directly_from_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "flagship_chain_modeling.py"

    result = subprocess.run(
        [sys.executable, str(module_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "flagship_chain horizon=3" in result.stdout
    assert "variables=7 residuals=3" in result.stdout
    assert "blocks=[chain_dynamics]" in result.stdout
    assert "jacobian_shape=(3, 7) jacobian_entries=9" in result.stdout
