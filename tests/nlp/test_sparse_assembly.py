import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.ir import Graph
from tinynlp.nlp import (
    AssemblyError,
    Problem,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
    format_residual_assembly,
    format_sparse_matrix,
    to_dense,
)


def _load_banded_example() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / "banded_residual_system.py"
    spec = importlib.util.spec_from_file_location("banded_residual_system", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


banded_example = _load_banded_example()


def test_assemble_banded_residual_values() -> None:
    contract = _banded_contract(size=6)

    assembly = assemble_residuals(
        contract,
        banded_example.banded_values(size=6, value=2.0),
    )

    assert [value.value for value in assembly.values] == [1.0] * 5
    assert [value.row for value in assembly.values] == list(range(5))
    assert all(value.provenance.kind == "residual" for value in assembly.values)


def test_assemble_banded_jacobian_coordinates() -> None:
    contract = _banded_contract(size=6)

    matrix = assemble_jacobian(
        contract, banded_example.banded_values(size=6, value=2.0)
    )

    assert matrix.shape == (5, 6)
    assert [(entry.row, entry.column) for entry in matrix.entries] == [
        coordinate for row in range(5) for coordinate in [(row, row), (row, row + 1)]
    ]
    for row in range(5):
        diagonal = matrix.entries[2 * row]
        upper = matrix.entries[(2 * row) + 1]
        assert diagonal.value == pytest.approx(4.0 / 3.0)
        assert upper.value == pytest.approx(-1.0 / 3.0)
        assert diagonal.provenance.variable is not None
        assert upper.provenance.variable is not None


def test_banded_jacobian_to_dense_matches_expected_matrix() -> None:
    contract = _banded_contract(size=6)

    matrix = assemble_jacobian(
        contract, banded_example.banded_values(size=6, value=2.0)
    )

    for row, expected in zip(
        to_dense(matrix),
        _expected_jacobian(size=6),
        strict=True,
    ):
        assert row == pytest.approx(expected)


def test_structural_jacobian_entry_is_kept_when_numeric_value_is_zero() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([x - x])
    contract = build_assembly_contract(problem)

    matrix = assemble_jacobian(contract, {"x": 2.0})

    assert [(entry.row, entry.column, entry.value) for entry in matrix.entries] == [
        (0, 0, 0.0)
    ]


def test_missing_variable_error_includes_residual_provenance() -> None:
    contract = _banded_contract(size=6)
    values = banded_example.banded_values(size=6, value=2.0)
    del values["x0"]

    with pytest.raises(AssemblyError, match="residual row 0.*x0"):
        assemble_residuals(contract, values)


def test_missing_variable_error_includes_jacobian_provenance() -> None:
    contract = _banded_contract(size=6)
    values = banded_example.banded_values(size=6, value=2.0)
    del values["x0"]

    with pytest.raises(AssemblyError, match="jacobian row 0 column 0.*x0"):
        assemble_jacobian(contract, values)


def test_division_by_zero_error_includes_row_and_column_provenance() -> None:
    contract = _banded_contract(size=6)
    values = banded_example.banded_values(size=6, value=2.0)
    values["x1"] = -1.0

    with pytest.raises(AssemblyError, match="residual row 0.*division by zero"):
        assemble_residuals(contract, values)

    with pytest.raises(
        AssemblyError, match="jacobian row 0 column 0.*division by zero"
    ):
        assemble_jacobian(contract, values)


def test_repeated_assembly_does_not_mutate_contract_structure() -> None:
    contract = _banded_contract(size=6)
    before = _contract_signature(contract)

    first = assemble_residuals(
        contract, banded_example.banded_values(size=6, value=2.0)
    )
    second = assemble_residuals(
        contract, banded_example.banded_values(size=6, value=3.0)
    )
    matrix = assemble_jacobian(
        contract, banded_example.banded_values(size=6, value=2.0)
    )

    assert [value.value for value in first.values] == [1.0] * 5
    assert [value.value for value in second.values] == [2.0] * 5
    assert len(matrix.entries) == 10
    assert _contract_signature(contract) == before


def test_assembly_reports_are_deterministic_and_address_free() -> None:
    contract = _banded_contract(size=6)
    values = banded_example.banded_values(size=6, value=2.0)

    residual_report = format_residual_assembly(assemble_residuals(contract, values))
    matrix_report = format_sparse_matrix(assemble_jacobian(contract, values))

    assert residual_report == format_residual_assembly(
        assemble_residuals(contract, values)
    )
    assert matrix_report == format_sparse_matrix(assemble_jacobian(contract, values))
    assert "object at" not in residual_report
    assert "object at" not in matrix_report
    assert "kind=residual row=0" in residual_report
    assert "kind=jacobian row=0 source=" in matrix_report


def _banded_contract(size: int):
    residuals = banded_example.banded_residual_expressions(size=size)
    return build_assembly_contract(Problem.from_residuals(residuals, name="banded"))


def _contract_signature(
    contract,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...]]:
    return (
        contract.sparsity.shape,
        tuple((term.row, term.column) for term in contract.jacobian_terms),
    )


def _expected_jacobian(size: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in range(size - 1):
        values = [0.0] * size
        values[row] = 4.0 / 3.0
        values[row + 1] = -1.0 / 3.0
        rows.append(values)
    return rows
