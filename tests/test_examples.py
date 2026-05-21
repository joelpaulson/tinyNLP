import importlib.util
from pathlib import Path
from types import ModuleType

from tinynlp.profiling import format_trace, trace_expression


def _load_canonical_examples() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_path / "canonical_expressions.py"
    spec = importlib.util.spec_from_file_location("canonical_expressions", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


canonical_expressions = _load_canonical_examples()


def test_canonical_examples_evaluate_expected_values() -> None:
    assert canonical_expressions.evaluate_affine({"x": 2, "y": 4}) == 5.0
    assert canonical_expressions.evaluate_quadratic({"x": 3}) == 16.0
    assert canonical_expressions.evaluate_residuals({"x": 1, "y": 2}) == [2.0, 1.0]


def test_canonical_examples_can_be_traced() -> None:
    expr = canonical_expressions.affine_expression()

    formatted = format_trace(trace_expression(expr))

    assert "variable" in formatted
    assert "constant" in formatted
    assert "object at" not in formatted


def test_examples_do_not_import_future_stage_modules() -> None:
    imported_names = set(vars(canonical_expressions))
    assert "autodiff" not in imported_names
    assert "nlp" not in imported_names
    assert "solvers" not in imported_names
    assert "bridges" not in imported_names
