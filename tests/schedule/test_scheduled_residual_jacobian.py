import importlib.util
import sys
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
)
from tinynlp.schedule import (
    ValidationStatus,
    format_schedule_report,
    format_scheduled_residual_jacobian_validation,
    prepare_scheduled_residual_jacobian_evaluator,
    validate_scheduled_residual_jacobian_evaluator,
)


def _load_flagship_example() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / "flagship_chain_modeling.py"
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


flagship_example = _load_flagship_example()


@pytest.mark.parametrize("horizon", [3, 10, 100])
def test_scheduled_residual_jacobian_matches_reference(horizon: int) -> None:
    case = flagship_example.flagship_chain_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)

    reference_residuals = assemble_residuals(contract, case.values)
    reference_jacobian = assemble_jacobian(contract, case.values)
    optimized = evaluator.evaluate(case.values)

    assert [value.value for value in optimized.residuals.values] == pytest.approx(
        [value.value for value in reference_residuals.values]
    )
    assert [value.provenance for value in optimized.residuals.values] == [
        value.provenance for value in reference_residuals.values
    ]
    assert optimized.jacobian.shape == reference_jacobian.shape
    assert [(entry.row, entry.column) for entry in optimized.jacobian.entries] == [
        (entry.row, entry.column) for entry in reference_jacobian.entries
    ]
    assert [entry.value for entry in optimized.jacobian.entries] == pytest.approx(
        [entry.value for entry in reference_jacobian.entries]
    )
    assert [entry.provenance for entry in optimized.jacobian.entries] == [
        entry.provenance for entry in reference_jacobian.entries
    ]


def test_scheduled_residual_jacobian_validation_updates_schedule_metadata() -> None:
    case = flagship_example.flagship_chain_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(
        contract,
        name="flagship_prepared_residual_jacobian",
    )

    validation = validate_scheduled_residual_jacobian_evaluator(
        evaluator,
        case.values,
    )
    report = format_schedule_report(validation.schedule)

    assert validation.passed is True
    assert validation.residual_max_abs_error == 0.0
    assert validation.jacobian_max_abs_error == 0.0
    assert validation.jacobian_coordinates_match is True
    assert validation.schedule.name == "flagship_prepared_residual_jacobian"
    assert [task.validation_status for task in validation.schedule.tasks] == [
        ValidationStatus.REFERENCE_VALIDATED,
        ValidationStatus.REFERENCE_VALIDATED,
        ValidationStatus.REFERENCE_VALIDATED,
    ]
    assert "stage=evaluate_residuals" in report
    assert "stage=evaluate_jacobian" in report
    assert "stage=assemble_sparse_coordinate_jacobian" in report
    assert "backend=prepared-python" in report
    assert "prepared_residual_kernels" in report
    assert "prepared_jacobian_kernels" in report
    assert "validation_status=reference_validated" in report
    assert "reference_residual_validation" in report
    assert "reference_jacobian_validation" in report
    assert "reference_sparse_coordinate_validation" in report
    assert "values=[0.115, 0.11595, 0.1168]" in report
    assert "(0,0)=-0.99" in report
    assert "object at" not in report


def test_scheduled_residual_jacobian_validation_format_is_deterministic() -> None:
    case = flagship_example.flagship_chain_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)

    validation = validate_scheduled_residual_jacobian_evaluator(
        evaluator,
        case.values,
    )
    formatted = format_scheduled_residual_jacobian_validation(validation)

    assert formatted == format_scheduled_residual_jacobian_validation(validation)
    assert "ScheduledResidualJacobianValidation" in formatted
    assert "passed=True" in formatted
    assert "residual_max_abs_error=0" in formatted
    assert "jacobian_max_abs_error=0" in formatted
    assert "jacobian_coordinates_match=True" in formatted
    assert "reference_residual_values=[0.115, 0.11595, 0.1168]" in formatted
    assert "optimized_residual_values=[0.115, 0.11595, 0.1168]" in formatted
    assert "reference_jacobian_entries=[(0,0)=-0.99" in formatted
    assert "optimized_jacobian_entries=[(0,0)=-0.99" in formatted
    assert "object at" not in formatted


def test_prepared_residual_jacobian_schedule_report_example_is_deterministic() -> None:
    example = _load_example("prepared_residual_jacobian_schedule_report")

    report = example.prepared_residual_jacobian_schedule_report()

    assert report == example.prepared_residual_jacobian_schedule_report()
    assert "ScheduledResidualJacobianValidation" in report
    assert "passed=True" in report
    assert "stage=evaluate_residuals" in report
    assert "stage=evaluate_jacobian" in report
    assert "stage=assemble_sparse_coordinate_jacobian" in report
    assert "backend=prepared-python" in report
    assert "prepared_residual_kernels" in report
    assert "prepared_jacobian_kernels" in report
    assert "validation_status=reference_validated" in report
    assert "object at" not in report


def test_scheduled_residual_jacobian_reports_missing_values() -> None:
    case = flagship_example.flagship_chain_case(horizon=3)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)
    values = dict(case.values)
    values.pop("x0")

    with pytest.raises(AssemblyError, match="scheduled residual row 0"):
        evaluator.evaluate(values)


def test_scheduled_residual_jacobian_reports_division_by_zero() -> None:
    graph = Graph()
    x = graph.variable("x")
    problem = Problem.from_residuals([1.0 / x])
    contract = build_assembly_contract(problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)

    with pytest.raises(AssemblyError, match="scheduled residual row 0"):
        evaluator.evaluate({"x": 0.0})
