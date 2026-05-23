"""Benchmarks for scheduled residual and Jacobian evaluation backends."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from tinynlp.nlp import assemble_jacobian, assemble_residuals, build_assembly_contract
from tinynlp.schedule import (
    prepare_scheduled_residual_jacobian_evaluator,
    validate_scheduled_residual_jacobian_evaluator,
)

HORIZON = 100
TOLERANCE = 1e-12


def _load_flagship_example() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
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


def test_reference_scheduled_residual_jacobian_benchmark(benchmark: Any) -> None:
    """Benchmark the reference residual and Jacobian assembly baseline."""

    case = flagship_example.flagship_chain_case(horizon=HORIZON)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)
    validation = validate_scheduled_residual_jacobian_evaluator(
        evaluator,
        case.values,
        tolerance=TOLERANCE,
    )
    assert validation.passed

    result = benchmark.pedantic(
        _reference_residual_jacobian,
        args=(contract, case.values),
        rounds=5,
        iterations=10,
    )

    assert tuple(value.value for value in result[0].values) == (
        validation.reference_residual_values
    )
    assert _entry_values(result[1]) == validation.reference_jacobian_entries


def test_prepared_scheduled_residual_jacobian_benchmark(benchmark: Any) -> None:
    """Benchmark prepared scheduler-backed residual and Jacobian evaluation."""

    case = flagship_example.flagship_chain_case(horizon=HORIZON)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)
    validation = validate_scheduled_residual_jacobian_evaluator(
        evaluator,
        case.values,
        tolerance=TOLERANCE,
    )
    assert validation.passed

    result = benchmark.pedantic(
        evaluator.evaluate,
        args=(case.values,),
        rounds=5,
        iterations=10,
    )

    assert tuple(value.value for value in result.residuals.values) == (
        validation.reference_residual_values
    )
    assert _entry_values(result.jacobian) == validation.reference_jacobian_entries


def _reference_residual_jacobian(contract, values):
    return assemble_residuals(contract, values), assemble_jacobian(contract, values)


def _entry_values(matrix) -> tuple[tuple[int, int, float], ...]:
    return tuple((entry.row, entry.column, entry.value) for entry in matrix.entries)
