import importlib.util
from pathlib import Path
from types import ModuleType

from tinynlp.ir import Graph
from tinynlp.nlp import Problem, assemble_jacobian, build_assembly_contract
from tinynlp.schedule import (
    build_kkt_assembly_schedule,
    build_problem_assembly_schedule,
    build_sensitivity_schedule,
    format_pipeline_report,
    format_schedule_report,
)


def _load_example(name: str) -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scheduled_pipeline_report = _load_example("scheduled_pipeline_report")
sensitivity_schedule_report = _load_example("sensitivity_schedule_report")
parameter_sensitivity_problem = _load_example("parameter_sensitivity_problem")


def test_single_schedule_report_is_deterministic_and_complete() -> None:
    report = format_schedule_report(build_problem_assembly_schedule(_tiny_problem()))

    assert report == format_schedule_report(
        build_problem_assembly_schedule(_tiny_problem())
    )
    assert "ScheduleReport name=tiny_assembly tasks=3" in report
    assert "task_order: t000 -> t001 -> t002" in report
    assert "dependency_edges:" in report
    assert "t001 -> t002" in report
    assert "stage=evaluate_residuals" in report
    assert "stage=evaluate_jacobian" in report
    assert "stage=assemble_sparse_coordinate_jacobian" in report
    assert "backend=python" in report
    assert "validation_status=not_run" in report
    assert "inputs:" in report
    assert "outputs:" in report
    assert "cached:" in report
    assert "materialized:" in report
    assert "provenance:" in report
    assert "object at" not in report


def test_pipeline_report_combines_assembly_and_kkt_schedules() -> None:
    problem = _tiny_problem()
    contract = build_assembly_contract(problem)
    jacobian = assemble_jacobian(contract, {"x": 1.0, "y": 2.0})
    assembly_schedule = build_problem_assembly_schedule(problem)
    kkt_schedule = build_kkt_assembly_schedule(jacobian, dependencies=("t002",))

    report = format_pipeline_report(
        (assembly_schedule, kkt_schedule), title="TinyPipeline"
    )

    assert report == format_pipeline_report(
        (assembly_schedule, kkt_schedule),
        title="TinyPipeline",
    )
    assert "TinyPipeline schedules=2" in report
    assert "pipeline_schedule index=0 name=tiny_assembly tasks=3" in report
    assert "pipeline_schedule index=1 name=kkt_assembly tasks=1" in report
    assert "stage=evaluate_residuals" in report
    assert "stage=evaluate_jacobian" in report
    assert "stage=assemble_sparse_coordinate_jacobian" in report
    assert "stage=assemble_kkt_system" in report
    assert "dependencies=[t002]" in report
    assert "object at" not in report


def test_chain_audit_example_report_includes_materialized_pipeline_values() -> None:
    report = scheduled_pipeline_report.chain_pipeline_report()

    assert report == scheduled_pipeline_report.chain_pipeline_report()
    assert "ChainDynamicsScheduledPipelineReport schedules=2" in report
    assert "name=chain_dynamics_assembly" in report
    assert "name=chain_dynamics_kkt" in report
    assert "stage=evaluate_residuals" in report
    assert "stage=evaluate_jacobian" in report
    assert "stage=assemble_sparse_coordinate_jacobian" in report
    assert "stage=assemble_kkt_system" in report
    assert "validation_status=reference_validated" in report
    assert "name=residual_values detail=values=[0.115, 0.11595, 0.1168]" in report
    assert "name=jacobian_values detail=entries=9 values=[" in report
    assert "name=kkt_entries detail=shape=(10, 10) entries=25 values=[" in report
    assert "object at" not in report


def test_sensitivity_schedule_report_exposes_rhs_and_solve_metadata() -> None:
    result = parameter_sensitivity_problem.evaluate_parameter_sensitivity()
    schedule = build_sensitivity_schedule(result, name="parameter_sensitivity_schedule")
    report = format_schedule_report(schedule)

    assert report == format_schedule_report(
        build_sensitivity_schedule(result, name="parameter_sensitivity_schedule")
    )
    assert "ScheduleReport name=parameter_sensitivity_schedule tasks=2" in report
    assert "task_order: t000 -> t001" in report
    assert "stage=build_sensitivity_rhs" in report
    assert "stage=solve_sensitivity_system" in report
    assert "backend=reference-python" in report
    assert "backend=dense-reference" in report
    assert "validation_status=reference_validated" in report
    assert "kind=parameter name=p detail=name=p node=" in report
    assert "kind=solve_variables name=solve_variables detail=names=[x, y]" in report
    assert "kind=kkt_system name=reduced_kkt detail=shape=(4, 4) entries=6" in report
    assert "kind=rhs name=sensitivity_rhs detail=length=4 entries=2" in report
    assert "name=sensitivity_rhs detail=values=[0, 0, -1, 2]" in report
    assert "kind=sensitivity_values name=sensitivities detail=x=-1, y=2" in report
    assert "object at" not in report


def test_sensitivity_audit_example_report_is_deterministic() -> None:
    report = sensitivity_schedule_report.parameter_sensitivity_schedule_report()

    assert report == sensitivity_schedule_report.parameter_sensitivity_schedule_report()
    assert "ScheduleReport name=parameter_sensitivity_schedule tasks=2" in report
    assert "stage=build_sensitivity_rhs" in report
    assert "stage=solve_sensitivity_system" in report
    assert "parameter=p" in report
    assert "object at" not in report


def _tiny_problem() -> Problem:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    return Problem.from_residuals([x + y - 1, (x * x) + y - 2], name="tiny")
