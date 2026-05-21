"""Benchmark scaffolding for reference expression evaluation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

EXPECTED_RESIDUALS = [2.0, 1.0]
RESIDUAL_VALUES = {"x": 1.0, "y": 2.0}


def test_reference_residual_evaluation_benchmark(benchmark: Any) -> None:
    """Benchmark residual evaluation only after correctness is checked."""

    assert _evaluate_residuals() == EXPECTED_RESIDUALS

    result = benchmark(_evaluate_residuals)

    assert result == EXPECTED_RESIDUALS


def _evaluate_residuals() -> list[float]:
    examples = _load_canonical_examples()
    return examples.evaluate_residuals(RESIDUAL_VALUES)


def _load_canonical_examples() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "canonical_expressions.py"
    spec = importlib.util.spec_from_file_location("canonical_expressions", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
