import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.autodiff import format_derivative_trace, jacobian, verify_jacobian
from tinynlp.backends import build_kernel_plan, evaluate, format_kernel_plan
from tinynlp.nlp import SparsityEntry, format_sparsity, jacobian_sparsity


def _load_banded_example() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_path / "banded_residual_system.py"
    spec = importlib.util.spec_from_file_location("banded_residual_system", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


banded_example = _load_banded_example()


def test_banded_residual_values_match_expected_uniform_point() -> None:
    assert banded_example.evaluate_banded_residuals(size=6, value=2.0) == [1.0] * 5


def test_banded_dense_jacobian_matches_expected_uniform_point() -> None:
    result = banded_example.evaluate_banded_jacobian(size=6, value=2.0)

    assert len(result) == 5
    for row, expected_row in zip(result, _expected_jacobian(size=6), strict=True):
        assert row == pytest.approx(expected_row)


def test_banded_jacobian_verification_passes() -> None:
    outputs = banded_example.banded_residual_expressions(size=6)

    result = verify_jacobian(outputs, banded_example.banded_values(size=6, value=2.0))

    assert result.passed is True


def test_banded_sparsity_has_expected_shape_and_row_major_entries() -> None:
    pattern = banded_example.banded_sparsity(size=6)

    assert pattern.shape == (5, 6)
    assert [
        (variable.name, int(variable.node_id)) for variable in pattern.variables
    ] == [(f"x{index}", index) for index in range(6)]
    assert [(entry.row, entry.column) for entry in pattern.entries] == [
        coordinate for row in range(5) for coordinate in [(row, row), (row, row + 1)]
    ]
    assert all(isinstance(entry, SparsityEntry) for entry in pattern.entries)


def test_banded_reports_have_no_object_addresses() -> None:
    outputs = banded_example.banded_residual_expressions(size=6)
    derivative_report = format_derivative_trace(jacobian(outputs).traces[0])
    kernel_report = format_kernel_plan(build_kernel_plan(outputs[0]))
    sparsity_report = format_sparsity(jacobian_sparsity(outputs))

    assert "object at" not in derivative_report
    assert "object at" not in kernel_report
    assert "object at" not in sparsity_report


def test_banded_symbolic_work_does_not_change_residual_values() -> None:
    outputs = banded_example.banded_residual_expressions(size=6)
    values = banded_example.banded_values(size=6, value=2.0)

    before = [evaluate(expr, values) for expr in outputs]
    _jacobian = jacobian(outputs)
    _sparsity = jacobian_sparsity(outputs)
    after = [evaluate(expr, values) for expr in outputs]

    assert before == [1.0] * 5
    assert after == before


def test_banded_size_must_be_at_least_two() -> None:
    with pytest.raises(ValueError, match="size must be at least 2"):
        banded_example.banded_residual_expressions(size=1)

    with pytest.raises(ValueError, match="size must be at least 2"):
        banded_example.banded_values(size=1)


def _expected_jacobian(size: int) -> list[list[float]]:
    rows: list[list[float]] = []
    for row in range(size - 1):
        values = [0.0] * size
        values[row] = 4.0 / 3.0
        values[row + 1] = -1.0 / 3.0
        rows.append(values)
    return rows
