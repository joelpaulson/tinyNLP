"""Execution schedule metadata for visible tinyNLP pipeline stages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from tinynlp.backends import KernelPlan, build_kernel_plan
from tinynlp.ir import Expr, OpKind
from tinynlp.nlp import (
    AssemblyContract,
    Problem,
    SparseMatrixAssembly,
    build_assembly_contract,
)


class ExecutionStage(StrEnum):
    """Supported scheduled pipeline stage names."""

    EVALUATE_EXPRESSION = "evaluate_expression"
    EVALUATE_RESIDUALS = "evaluate_residuals"
    EVALUATE_JACOBIAN = "evaluate_jacobian"
    ASSEMBLE_SPARSE_COORDINATE_JACOBIAN = "assemble_sparse_coordinate_jacobian"
    ASSEMBLE_KKT_SYSTEM = "assemble_kkt_system"
    SOLVE_REFERENCE_LINEAR_SYSTEM = "solve_reference_linear_system"
    SOLVER_ITERATION_STEP = "solver_iteration_step"
    BUILD_SENSITIVITY_RHS = "build_sensitivity_rhs"
    SOLVE_SENSITIVITY_SYSTEM = "solve_sensitivity_system"


class ValidationStatus(StrEnum):
    """Validation state recorded for a scheduled task."""

    NOT_RUN = "not_run"
    REFERENCE_VALIDATED = "reference_validated"


@dataclass(frozen=True)
class ScheduleItem:
    """Stable schedule item summary."""

    kind: str
    name: str
    detail: str


@dataclass(frozen=True)
class ScheduleProvenance:
    """Stable provenance summary for a scheduled task."""

    kind: str
    detail: str


@dataclass(frozen=True)
class ExecutionTask:
    """One scheduled pipeline task."""

    task_id: str
    stage: ExecutionStage
    label: str
    inputs: tuple[ScheduleItem, ...]
    outputs: tuple[ScheduleItem, ...]
    dependencies: tuple[str, ...]
    backend_name: str
    cached: tuple[ScheduleItem, ...]
    materialized: tuple[ScheduleItem, ...]
    provenance: tuple[ScheduleProvenance, ...]
    validation_status: ValidationStatus

    def __post_init__(self) -> None:
        if not self.task_id:
            msg = "execution task id must be non-empty"
            raise ValueError(msg)
        if not self.label:
            msg = "execution task label must be non-empty"
            raise ValueError(msg)
        if not self.backend_name:
            msg = "execution task backend name must be non-empty"
            raise ValueError(msg)


@dataclass(frozen=True)
class ExecutionSchedule:
    """Ordered schedule for visible pipeline tasks."""

    name: str
    tasks: tuple[ExecutionTask, ...]

    def __post_init__(self) -> None:
        if not self.name:
            msg = "execution schedule name must be non-empty"
            raise ValueError(msg)
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            msg = "execution schedule task ids must be unique"
            raise ValueError(msg)


def build_expression_schedule(
    expr: Expr,
    *,
    name: str = "expression",
    backend_name: str = "python",
) -> ExecutionSchedule:
    """Build a schedule for evaluating one expression."""

    plan = build_kernel_plan(expr)
    task = ExecutionTask(
        task_id="t000",
        stage=ExecutionStage.EVALUATE_EXPRESSION,
        label=f"evaluate expression node {expr.id}",
        inputs=_kernel_plan_inputs(plan),
        outputs=(
            ScheduleItem(
                kind="value",
                name="expression_value",
                detail=f"output_node={plan.output}",
            ),
        ),
        dependencies=(),
        backend_name=backend_name,
        cached=(_kernel_plan_item("kernel_plan", plan),),
        materialized=(
            ScheduleItem(
                kind="materialized_value",
                name="expression_value",
                detail="status=not_run",
            ),
        ),
        provenance=(
            ScheduleProvenance(
                kind="expression",
                detail=f"output_node={expr.id}",
            ),
        ),
        validation_status=ValidationStatus.NOT_RUN,
    )
    return ExecutionSchedule(name=name, tasks=(task,))


def build_problem_assembly_schedule(
    problem: Problem,
    *,
    name: str | None = None,
    backend_name: str = "python",
) -> ExecutionSchedule:
    """Build a metadata schedule for residual/Jacobian assembly."""

    contract = build_assembly_contract(problem)
    schedule_name = name or f"{problem.name}_assembly"
    tasks = (
        _residual_task(contract, backend_name),
        _jacobian_task(contract, backend_name),
        _sparse_coordinate_task(contract, backend_name),
    )
    return ExecutionSchedule(name=schedule_name, tasks=tasks)


def build_kkt_assembly_schedule(
    jacobian_assembly: SparseMatrixAssembly,
    *,
    name: str = "kkt_assembly",
    dependencies: Sequence[str] = (),
    backend_name: str = "reference-python",
) -> ExecutionSchedule:
    """Build a metadata schedule for KKT assembly from a sparse Jacobian."""

    residual_size, primal_size = jacobian_assembly.shape
    total_size = residual_size + primal_size
    expected_entries = primal_size + (2 * len(jacobian_assembly.entries))
    task = ExecutionTask(
        task_id="t000",
        stage=ExecutionStage.ASSEMBLE_KKT_SYSTEM,
        label="assemble KKT system",
        inputs=(
            ScheduleItem(
                kind="sparse_matrix",
                name="jacobian",
                detail=(
                    f"shape={_format_shape(jacobian_assembly.shape)} "
                    f"entries={len(jacobian_assembly.entries)}"
                ),
            ),
        ),
        outputs=(
            ScheduleItem(
                kind="kkt_system",
                name="kkt",
                detail=(
                    f"shape={_format_shape((total_size, total_size))} "
                    f"expected_entries={expected_entries}"
                ),
            ),
        ),
        dependencies=tuple(dependencies),
        backend_name=backend_name,
        cached=(
            ScheduleItem(
                kind="structure",
                name="kkt_blocks",
                detail=(
                    f"primal_size={primal_size} residual_size={residual_size} blocks=4"
                ),
            ),
        ),
        materialized=(
            ScheduleItem(
                kind="materialized_value",
                name="kkt_entries",
                detail="status=not_run",
            ),
        ),
        provenance=(
            ScheduleProvenance(
                kind="sparse_jacobian",
                detail=(
                    f"shape={_format_shape(jacobian_assembly.shape)} "
                    f"entries={len(jacobian_assembly.entries)}"
                ),
            ),
        ),
        validation_status=ValidationStatus.NOT_RUN,
    )
    return ExecutionSchedule(name=name, tasks=(task,))


def format_execution_schedule(schedule: ExecutionSchedule) -> str:
    """Format an execution schedule deterministically."""

    lines = [f"ExecutionSchedule name={schedule.name} tasks={len(schedule.tasks)}"]
    for task in schedule.tasks:
        dependencies = ", ".join(task.dependencies)
        if not dependencies:
            dependencies = "<none>"
        lines.append(
            "task "
            f"id={task.task_id} "
            f"stage={task.stage.value} "
            f"label={task.label} "
            f"backend={task.backend_name} "
            f"validation={task.validation_status.value} "
            f"dependencies=[{dependencies}]"
        )
        _append_items(lines, "inputs", task.inputs)
        _append_items(lines, "outputs", task.outputs)
        _append_items(lines, "cached", task.cached)
        _append_items(lines, "materialized", task.materialized)
        _append_provenance(lines, task.provenance)
    return "\n".join(lines)


def _residual_task(
    contract: AssemblyContract,
    backend_name: str,
) -> ExecutionTask:
    return ExecutionTask(
        task_id="t000",
        stage=ExecutionStage.EVALUATE_RESIDUALS,
        label=f"evaluate residuals for {contract.problem.name}",
        inputs=_problem_inputs(contract),
        outputs=(
            ScheduleItem(
                kind="residual_values",
                name="residual_vector",
                detail=f"rows={len(contract.residual_terms)}",
            ),
        ),
        dependencies=(),
        backend_name=backend_name,
        cached=(
            _assembly_contract_item(contract),
            ScheduleItem(
                kind="kernel_plans",
                name="residual_kernel_plans",
                detail=f"count={len(contract.residual_terms)}",
            ),
        ),
        materialized=(
            ScheduleItem(
                kind="materialized_value",
                name="residual_values",
                detail="status=not_run",
            ),
        ),
        provenance=(
            ScheduleProvenance(
                kind="problem",
                detail=(
                    f"name={contract.problem.name} "
                    f"residuals={len(contract.residual_terms)}"
                ),
            ),
        ),
        validation_status=ValidationStatus.NOT_RUN,
    )


def _jacobian_task(
    contract: AssemblyContract,
    backend_name: str,
) -> ExecutionTask:
    return ExecutionTask(
        task_id="t001",
        stage=ExecutionStage.EVALUATE_JACOBIAN,
        label=f"evaluate Jacobian entries for {contract.problem.name}",
        inputs=_problem_inputs(contract),
        outputs=(
            ScheduleItem(
                kind="jacobian_values",
                name="jacobian_entries",
                detail=f"entries={len(contract.jacobian_terms)}",
            ),
        ),
        dependencies=(),
        backend_name=backend_name,
        cached=(
            _assembly_contract_item(contract),
            _sparsity_item(contract),
            ScheduleItem(
                kind="kernel_plans",
                name="jacobian_kernel_plans",
                detail=f"count={len(contract.jacobian_terms)}",
            ),
            ScheduleItem(
                kind="derivative_traces",
                name="jacobian_derivative_traces",
                detail=f"rows={len(contract.residual_terms)}",
            ),
        ),
        materialized=(
            ScheduleItem(
                kind="materialized_value",
                name="jacobian_values",
                detail="status=not_run",
            ),
        ),
        provenance=(
            ScheduleProvenance(
                kind="sparsity",
                detail=(
                    f"shape={_format_shape(contract.sparsity.shape)} "
                    f"entries={len(contract.sparsity.entries)}"
                ),
            ),
        ),
        validation_status=ValidationStatus.NOT_RUN,
    )


def _sparse_coordinate_task(
    contract: AssemblyContract,
    backend_name: str,
) -> ExecutionTask:
    return ExecutionTask(
        task_id="t002",
        stage=ExecutionStage.ASSEMBLE_SPARSE_COORDINATE_JACOBIAN,
        label=f"assemble sparse coordinate Jacobian for {contract.problem.name}",
        inputs=(
            ScheduleItem(
                kind="jacobian_values",
                name="jacobian_entries",
                detail=f"entries={len(contract.jacobian_terms)}",
            ),
            _sparsity_item(contract),
        ),
        outputs=(
            ScheduleItem(
                kind="sparse_matrix",
                name="jacobian",
                detail=(
                    f"shape={_format_shape(contract.sparsity.shape)} "
                    f"entries={len(contract.jacobian_terms)}"
                ),
            ),
        ),
        dependencies=("t001",),
        backend_name=backend_name,
        cached=(_sparsity_item(contract),),
        materialized=(
            ScheduleItem(
                kind="materialized_value",
                name="sparse_coordinate_jacobian",
                detail="status=not_run",
            ),
        ),
        provenance=(
            ScheduleProvenance(
                kind="assembly_contract",
                detail=(
                    f"problem={contract.problem.name} "
                    f"jacobian_terms={len(contract.jacobian_terms)}"
                ),
            ),
        ),
        validation_status=ValidationStatus.NOT_RUN,
    )


def _kernel_plan_inputs(plan: KernelPlan) -> tuple[ScheduleItem, ...]:
    variables = ScheduleItem(
        kind="variables",
        name="runtime_values",
        detail="names=[" + ", ".join(name for _, name in plan.variables) + "]",
    )
    constants = ScheduleItem(
        kind="constants",
        name="literal_values",
        detail=f"count={len(plan.constants)}",
    )
    return (variables, constants)


def _problem_inputs(contract: AssemblyContract) -> tuple[ScheduleItem, ...]:
    return (
        ScheduleItem(
            kind="problem",
            name=contract.problem.name,
            detail=(
                f"variables={len(contract.variables)} "
                f"residuals={len(contract.residual_terms)}"
            ),
        ),
        ScheduleItem(
            kind="variables",
            name="runtime_values",
            detail=(
                "names=["
                + ", ".join(variable.name for variable in contract.variables)
                + "]"
            ),
        ),
    )


def _assembly_contract_item(contract: AssemblyContract) -> ScheduleItem:
    return ScheduleItem(
        kind="assembly_contract",
        name=contract.problem.name,
        detail=(
            f"variables={len(contract.variables)} "
            f"residual_terms={len(contract.residual_terms)} "
            f"jacobian_terms={len(contract.jacobian_terms)}"
        ),
    )


def _sparsity_item(contract: AssemblyContract) -> ScheduleItem:
    return ScheduleItem(
        kind="sparsity",
        name="jacobian_sparsity",
        detail=(
            f"shape={_format_shape(contract.sparsity.shape)} "
            f"entries={len(contract.sparsity.entries)}"
        ),
    )


def _kernel_plan_item(name: str, plan: KernelPlan) -> ScheduleItem:
    return ScheduleItem(
        kind="kernel_plan",
        name=name,
        detail=(
            f"output={plan.output} "
            f"ops=[{_format_operation_counts(plan)}] "
            f"temporaries={plan.stats.temporary_count}"
        ),
    )


def _format_operation_counts(plan: KernelPlan) -> str:
    counts = dict(plan.stats.operation_counts)
    return ", ".join(
        f"{op.value}:{counts[op]}" for op in OpKind if counts.get(op, 0) > 0
    )


def _format_shape(shape: tuple[int, int]) -> str:
    return f"({shape[0]}, {shape[1]})"


def _append_items(
    lines: list[str],
    title: str,
    items: tuple[ScheduleItem, ...],
) -> None:
    lines.append(f"  {title}:")
    if not items:
        lines.append("    <none>")
        return
    lines.extend(
        f"    - kind={item.kind} name={item.name} detail={item.detail}"
        for item in items
    )


def _append_provenance(
    lines: list[str],
    provenance: tuple[ScheduleProvenance, ...],
) -> None:
    lines.append("  provenance:")
    if not provenance:
        lines.append("    <none>")
        return
    lines.extend(f"    - kind={item.kind} detail={item.detail}" for item in provenance)
