"""Benchmark scaffolding for reference expression evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

from tinynlp.backends import Backend, KernelPlan, build_kernel_plan, get_backend

EXPECTED_RESIDUALS = [2.0, 1.0]
RESIDUAL_VALUES = {"x": 1.0, "y": 2.0}


def test_reference_residual_evaluation_benchmark(benchmark: Any) -> None:
    """Benchmark protocol-backed residual evaluation after correctness checks."""

    plans = _build_residual_plans()
    backend = get_backend("python")
    assert _execute_residual_plans(plans, backend) == EXPECTED_RESIDUALS

    result = benchmark(_execute_residual_plans, plans, backend)

    assert result == EXPECTED_RESIDUALS


def _build_residual_plans() -> list[KernelPlan]:
    examples = _load_canonical_examples()
    return [build_kernel_plan(expr) for expr in examples.residual_expressions()]


def _execute_residual_plans(plans: list[KernelPlan], backend: Backend) -> list[float]:
    return [backend.execute(plan, RESIDUAL_VALUES) for plan in plans]


def _load_canonical_examples() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "canonical_expressions.py"
    spec = importlib.util.spec_from_file_location("canonical_expressions", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
