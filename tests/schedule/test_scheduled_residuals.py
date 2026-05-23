import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.nlp import AssemblyError, assemble_residuals, build_assembly_contract
from tinynlp.schedule import (
    ValidationStatus,
    format_schedule_report,
    format_scheduled_residual_validation,
    prepare_scheduled_residual_evaluator,
    validate_scheduled_residual_evaluator,
)


def _load_chain_example() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / "chain_dynamics_problem.py"
    spec = importlib.util.spec_from_file_location("chain_dynamics_problem", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


chain_example = _load_chain_example()


@pytest.mark.parametrize("horizon", [3, 10, 100])
def test_scheduled_residual_evaluator_matches_reference(horizon: int) -> None:
    case = chain_example.chain_dynamics_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(contract)

    reference = assemble_residuals(contract, case.values)
    optimized = evaluator.evaluate(case.values)

    assert [value.value for value in optimized.values] == pytest.approx(
        [value.value for value in reference.values]
    )
    assert [value.provenance for value in optimized.values] == [
        value.provenance for value in reference.values
    ]


def test_scheduled_residual_validation_updates_schedule_metadata() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(
        contract,
        name="chain_prepared_residuals",
    )

    validation = validate_scheduled_residual_evaluator(evaluator, case.values)
    report = format_schedule_report(validation.schedule)

    assert validation.passed is True
    assert validation.max_abs_error == 0.0
    assert validation.schedule.name == "chain_prepared_residuals"
    assert validation.schedule.tasks[0].backend_name == "prepared-python"
    assert validation.schedule.tasks[0].validation_status is (
        ValidationStatus.REFERENCE_VALIDATED
    )
    assert "stage=evaluate_residuals" in report
    assert "backend=prepared-python" in report
    assert "kind=prepared_kernels name=prepared_residual_kernels" in report
    assert "validation_status=reference_validated" in report
    assert "passed=True max_abs_error=0 tolerance=1e-12" in report
    assert "object at" not in report


def test_scheduled_residual_validation_format_is_deterministic() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(contract)

    validation = validate_scheduled_residual_evaluator(evaluator, case.values)
    formatted = format_scheduled_residual_validation(validation)

    assert formatted == format_scheduled_residual_validation(validation)
    assert "ScheduledResidualValidation" in formatted
    assert "passed=True" in formatted
    assert "max_abs_error=0" in formatted
    assert "reference_values=[0.115, 0.11595, 0.1168]" in formatted
    assert "optimized_values=[0.115, 0.11595, 0.1168]" in formatted
    assert "object at" not in formatted


def test_prepared_residual_schedule_report_example_is_deterministic() -> None:
    example = _load_example("prepared_residual_schedule_report")

    report = example.prepared_residual_schedule_report()

    assert report == example.prepared_residual_schedule_report()
    assert "ScheduledResidualValidation" in report
    assert "passed=True" in report
    assert "max_abs_error=0" in report
    assert "stage=evaluate_residuals" in report
    assert "backend=prepared-python" in report
    assert "prepared_residual_kernels" in report
    assert "validation_status=reference_validated" in report
    assert "object at" not in report


def test_scheduled_residual_evaluator_reports_missing_values() -> None:
    case = chain_example.chain_dynamics_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(contract)
    values = dict(case.values)
    values.pop("x0")

    with pytest.raises(AssemblyError, match="scheduled residual row 0"):
        evaluator.evaluate(values)


def _load_example(name: str) -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
