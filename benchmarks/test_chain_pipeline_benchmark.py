"""Benchmark smoke cases for the chain dynamics pipeline."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tinynlp.nlp import (
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
)
from tinynlp.solvers import build_kkt_system


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


@pytest.mark.parametrize("horizon", [10, 100])
def test_chain_residual_assembly_benchmark(benchmark: Any, horizon: int) -> None:
    """Benchmark residual assembly after correctness checks."""

    case = chain_example.chain_dynamics_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    expected_rows = horizon
    _assert_residual_assembly(assemble_residuals(contract, case.values), expected_rows)

    result = benchmark.pedantic(
        assemble_residuals,
        args=(contract, case.values),
        rounds=1,
        iterations=1,
    )

    _assert_residual_assembly(result, expected_rows)


@pytest.mark.parametrize("horizon", [10, 100])
def test_chain_sparse_jacobian_assembly_benchmark(
    benchmark: Any,
    horizon: int,
) -> None:
    """Benchmark sparse Jacobian assembly after correctness checks."""

    case = chain_example.chain_dynamics_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    expected_shape = (horizon, (2 * horizon) + 1)
    expected_entries = 3 * horizon
    _assert_jacobian_assembly(
        assemble_jacobian(contract, case.values),
        expected_shape,
        expected_entries,
    )

    result = benchmark.pedantic(
        assemble_jacobian,
        args=(contract, case.values),
        rounds=1,
        iterations=1,
    )

    _assert_jacobian_assembly(result, expected_shape, expected_entries)


@pytest.mark.parametrize("horizon", [10, 100])
def test_chain_kkt_construction_benchmark(benchmark: Any, horizon: int) -> None:
    """Benchmark KKT construction after correctness checks."""

    case = chain_example.chain_dynamics_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    jacobian = assemble_jacobian(contract, case.values)
    expected_shape = ((3 * horizon) + 1, (3 * horizon) + 1)
    expected_entries = (8 * horizon) + 1
    _assert_kkt_system(build_kkt_system(jacobian), expected_shape, expected_entries)

    result = benchmark.pedantic(
        build_kkt_system,
        args=(jacobian,),
        rounds=1,
        iterations=1,
    )

    _assert_kkt_system(result, expected_shape, expected_entries)


def _assert_residual_assembly(assembly, expected_rows: int) -> None:
    assert len(assembly.values) == expected_rows
    assert [value.row for value in assembly.values] == list(range(expected_rows))


def _assert_jacobian_assembly(
    matrix,
    expected_shape: tuple[int, int],
    expected_entries: int,
) -> None:
    assert matrix.shape == expected_shape
    assert len(matrix.entries) == expected_entries


def _assert_kkt_system(
    system,
    expected_shape: tuple[int, int],
    expected_entries: int,
) -> None:
    assert system.shape == expected_shape
    assert len(system.entries) == expected_entries
