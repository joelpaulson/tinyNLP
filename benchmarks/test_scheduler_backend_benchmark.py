"""Benchmarks for scheduled residual evaluation backends."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from tinynlp.nlp import assemble_residuals, build_assembly_contract
from tinynlp.schedule import (
    prepare_scheduled_residual_evaluator,
    validate_scheduled_residual_evaluator,
)

HORIZON = 100
TOLERANCE = 1e-12


def _load_chain_example() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "chain_dynamics_problem.py"
    spec = importlib.util.spec_from_file_location("chain_dynamics_problem", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chain_example = _load_chain_example()


def test_reference_scheduled_residual_evaluation_benchmark(benchmark: Any) -> None:
    """Benchmark the reference residual assembly baseline."""

    case = chain_example.chain_dynamics_case(horizon=HORIZON)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(contract)
    validation = validate_scheduled_residual_evaluator(
        evaluator,
        case.values,
        tolerance=TOLERANCE,
    )
    assert validation.passed
    expected = validation.reference_values

    result = benchmark.pedantic(
        assemble_residuals,
        args=(contract, case.values),
        rounds=5,
        iterations=10,
    )

    assert tuple(value.value for value in result.values) == expected


def test_prepared_scheduled_residual_evaluation_benchmark(benchmark: Any) -> None:
    """Benchmark prepared scheduler-backed residual evaluation."""

    case = chain_example.chain_dynamics_case(horizon=HORIZON)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(contract)
    validation = validate_scheduled_residual_evaluator(
        evaluator,
        case.values,
        tolerance=TOLERANCE,
    )
    assert validation.passed
    expected = validation.reference_values

    result = benchmark.pedantic(
        evaluator.evaluate,
        args=(case.values,),
        rounds=5,
        iterations=10,
    )

    assert tuple(value.value for value in result.values) == expected
