import tinynlp
import tinynlp.schedule as schedule
from tinynlp.ir import Graph
from tinynlp.nlp import Problem, assemble_jacobian, build_assembly_contract
from tinynlp.schedule import (
    ExecutionStage,
    ValidationStatus,
    build_expression_schedule,
    build_kkt_assembly_schedule,
    build_problem_assembly_schedule,
    format_execution_schedule,
)


def test_expression_schedule_summarizes_kernel_plan_deterministically() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = (x + 1) * y

    first = build_expression_schedule(expr)
    second = build_expression_schedule(expr)

    assert first == second
    assert first.name == "expression"
    assert len(first.tasks) == 1
    task = first.tasks[0]
    assert task.task_id == "t000"
    assert task.stage is ExecutionStage.EVALUATE_EXPRESSION
    assert task.backend_name == "python"
    assert task.validation_status is ValidationStatus.NOT_RUN
    assert task.dependencies == ()
    assert task.outputs[0].detail == f"output_node={expr.id}"
    assert task.cached[0].kind == "kernel_plan"
    assert f"output={expr.id}" in task.cached[0].detail
    assert "variable:2" in task.cached[0].detail
    assert "constant:1" in task.cached[0].detail
    assert "add:1" in task.cached[0].detail
    assert "mul:1" in task.cached[0].detail
    assert "temporaries=2" in task.cached[0].detail


def test_problem_assembly_schedule_has_expected_order_and_metadata() -> None:
    problem = _tiny_problem()

    schedule_result = build_problem_assembly_schedule(problem)

    assert schedule_result.name == "tiny_assembly"
    assert [task.task_id for task in schedule_result.tasks] == ["t000", "t001", "t002"]
    assert [task.stage for task in schedule_result.tasks] == [
        ExecutionStage.EVALUATE_RESIDUALS,
        ExecutionStage.EVALUATE_JACOBIAN,
        ExecutionStage.ASSEMBLE_SPARSE_COORDINATE_JACOBIAN,
    ]
    assert schedule_result.tasks[2].dependencies == ("t001",)
    assert all(task.backend_name == "python" for task in schedule_result.tasks)
    assert all(
        task.validation_status is ValidationStatus.NOT_RUN
        for task in schedule_result.tasks
    )

    residual_task, jacobian_task, sparse_task = schedule_result.tasks
    assert _item_names(residual_task.cached) == [
        "tiny",
        "residual_kernel_plans",
    ]
    assert _item_names(jacobian_task.cached) == [
        "tiny",
        "jacobian_sparsity",
        "jacobian_kernel_plans",
        "jacobian_derivative_traces",
    ]
    assert "rows=2" in residual_task.outputs[0].detail
    assert "entries=4" in jacobian_task.outputs[0].detail
    assert "shape=(2, 2)" in sparse_task.outputs[0].detail
    assert "entries=4" in sparse_task.outputs[0].detail


def test_kkt_assembly_schedule_preserves_shape_counts_and_dependencies() -> None:
    problem = _tiny_problem()
    contract = build_assembly_contract(problem)
    jacobian = assemble_jacobian(contract, {"x": 1.0, "y": 2.0})

    schedule_result = build_kkt_assembly_schedule(
        jacobian,
        dependencies=("t002",),
    )

    assert schedule_result.name == "kkt_assembly"
    assert len(schedule_result.tasks) == 1
    task = schedule_result.tasks[0]
    assert task.task_id == "t000"
    assert task.stage is ExecutionStage.ASSEMBLE_KKT_SYSTEM
    assert task.backend_name == "reference-python"
    assert task.dependencies == ("t002",)
    assert "shape=(2, 2) entries=4" in task.inputs[0].detail
    assert "shape=(4, 4) expected_entries=10" in task.outputs[0].detail
    assert "primal_size=2 residual_size=2 blocks=4" in task.cached[0].detail


def test_execution_schedule_report_is_stable_and_address_free() -> None:
    report = format_execution_schedule(build_problem_assembly_schedule(_tiny_problem()))

    assert report == format_execution_schedule(
        build_problem_assembly_schedule(_tiny_problem())
    )
    assert "ExecutionSchedule name=tiny_assembly tasks=3" in report
    assert "task id=t000 stage=evaluate_residuals" in report
    assert "task id=t001 stage=evaluate_jacobian" in report
    assert "task id=t002 stage=assemble_sparse_coordinate_jacobian" in report
    assert "backend=python" in report
    assert "validation=not_run" in report
    assert "dependencies=[t001]" in report
    assert "inputs:" in report
    assert "outputs:" in report
    assert "cached:" in report
    assert "materialized:" in report
    assert "provenance:" in report
    assert "object at" not in report


def test_schedule_public_exports_and_top_level_package_boundary() -> None:
    assert hasattr(schedule, "ExecutionSchedule")
    assert hasattr(schedule, "ExecutionTask")
    assert hasattr(schedule, "ExecutionStage")
    assert hasattr(schedule, "ValidationStatus")
    assert hasattr(schedule, "build_expression_schedule")
    assert hasattr(schedule, "build_problem_assembly_schedule")
    assert hasattr(schedule, "build_kkt_assembly_schedule")
    assert hasattr(schedule, "format_execution_schedule")
    assert tinynlp.__all__ == ["__version__"]


def _tiny_problem() -> Problem:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    return Problem.from_residuals([x + y - 1, (x * x) + y - 2], name="tiny")


def _item_names(items) -> list[str]:
    return [item.name for item in items]
