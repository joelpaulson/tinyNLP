"""Scheduler-backed residual and Jacobian evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from tinynlp.backends import (
    EvaluationError,
    PreparedKernel,
    PreparedKernelBackend,
    prepare_kernel,
)
from tinynlp.nlp import (
    AssemblyContract,
    AssemblyError,
    AssemblyProvenance,
    CoordinateEntry,
    ResidualAssembly,
    ResidualValue,
    SparseMatrixAssembly,
    assemble_jacobian,
    assemble_residuals,
)
from tinynlp.schedule.core import (
    ExecutionSchedule,
    ExecutionStage,
    ExecutionTask,
    ScheduleItem,
    ScheduleProvenance,
    ValidationStatus,
)


@dataclass(frozen=True)
class ScheduledResidualJacobianEvaluation:
    """Prepared residual and Jacobian assembly outputs."""

    residuals: ResidualAssembly
    jacobian: SparseMatrixAssembly


@dataclass(frozen=True)
class ScheduledResidualJacobianEvaluator:
    """Prepared evaluator for scheduled residual and Jacobian tasks."""

    name: str
    contract: AssemblyContract
    backend: PreparedKernelBackend
    residual_kernels: tuple[PreparedKernel, ...]
    jacobian_kernels: tuple[PreparedKernel, ...]
    schedule: ExecutionSchedule

    def evaluate(
        self,
        values: Mapping[str, float],
    ) -> ScheduledResidualJacobianEvaluation:
        """Evaluate scheduled residuals and Jacobian coordinates."""

        residual_values: list[ResidualValue] = []
        for term, kernel in zip(
            self.contract.residual_terms,
            self.residual_kernels,
            strict=True,
        ):
            provenance = AssemblyProvenance(
                kind="residual",
                row=term.row,
                column=None,
                source_node_id=term.expr.id,
            )
            try:
                value = self.backend.execute_prepared(kernel, values)
            except EvaluationError as exc:
                raise _scheduled_assembly_error(provenance, exc) from exc
            residual_values.append(
                ResidualValue(
                    row=term.row,
                    value=value,
                    provenance=provenance,
                )
            )

        jacobian_entries: list[CoordinateEntry] = []
        for term, kernel in zip(
            self.contract.jacobian_terms,
            self.jacobian_kernels,
            strict=True,
        ):
            provenance = AssemblyProvenance(
                kind="jacobian",
                row=term.row,
                column=term.column,
                source_node_id=term.source_residual.id,
                derivative_node_id=term.derivative.id,
                variable=term.variable,
            )
            try:
                value = self.backend.execute_prepared(kernel, values)
            except EvaluationError as exc:
                raise _scheduled_assembly_error(provenance, exc) from exc
            jacobian_entries.append(
                CoordinateEntry(
                    row=term.row,
                    column=term.column,
                    value=value,
                    provenance=provenance,
                )
            )

        return ScheduledResidualJacobianEvaluation(
            residuals=ResidualAssembly(values=tuple(residual_values)),
            jacobian=SparseMatrixAssembly(
                shape=self.contract.sparsity.shape,
                entries=tuple(jacobian_entries),
            ),
        )


@dataclass(frozen=True)
class ScheduledResidualJacobianValidation:
    """Reference validation for scheduled residual and Jacobian execution."""

    schedule: ExecutionSchedule
    passed: bool
    tolerance: float
    residual_max_abs_error: float
    jacobian_max_abs_error: float
    jacobian_coordinates_match: bool
    reference_residual_values: tuple[float, ...]
    optimized_residual_values: tuple[float, ...]
    reference_jacobian_entries: tuple[tuple[int, int, float], ...]
    optimized_jacobian_entries: tuple[tuple[int, int, float], ...]


def prepare_scheduled_residual_jacobian_evaluator(
    contract: AssemblyContract,
    *,
    name: str | None = None,
) -> ScheduledResidualJacobianEvaluator:
    """Prepare residual and Jacobian KernelPlans for scheduled execution."""

    backend = PreparedKernelBackend()
    residual_kernels = tuple(
        prepare_kernel(term.kernel_plan) for term in contract.residual_terms
    )
    jacobian_kernels = tuple(
        prepare_kernel(term.kernel_plan) for term in contract.jacobian_terms
    )
    evaluator_name = name or f"{contract.problem.name}_scheduled_residual_jacobian"
    schedule = _build_prepared_residual_jacobian_schedule(
        contract=contract,
        name=evaluator_name,
        backend_name=backend.name,
        residual_kernels=residual_kernels,
        jacobian_kernels=jacobian_kernels,
        validation_status=ValidationStatus.NOT_RUN,
    )
    return ScheduledResidualJacobianEvaluator(
        name=evaluator_name,
        contract=contract,
        backend=backend,
        residual_kernels=residual_kernels,
        jacobian_kernels=jacobian_kernels,
        schedule=schedule,
    )


def validate_scheduled_residual_jacobian_evaluator(
    evaluator: ScheduledResidualJacobianEvaluator,
    values: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> ScheduledResidualJacobianValidation:
    """Validate prepared execution against reference residual/Jacobian assembly."""

    reference_residuals = assemble_residuals(evaluator.contract, values)
    reference_jacobian = assemble_jacobian(evaluator.contract, values)
    optimized = evaluator.evaluate(values)

    reference_residual_values = tuple(
        value.value for value in reference_residuals.values
    )
    optimized_residual_values = tuple(
        value.value for value in optimized.residuals.values
    )
    reference_jacobian_entries = _entry_values(reference_jacobian)
    optimized_jacobian_entries = _entry_values(optimized.jacobian)
    residual_error = _max_abs_error(
        reference_residual_values,
        optimized_residual_values,
    )
    coordinates_match = _coordinates(reference_jacobian_entries) == _coordinates(
        optimized_jacobian_entries
    )
    jacobian_error = _entry_max_abs_error(
        reference_jacobian_entries,
        optimized_jacobian_entries,
        coordinates_match,
    )
    residual_passed = residual_error <= tolerance
    jacobian_passed = coordinates_match and jacobian_error <= tolerance
    passed = residual_passed and jacobian_passed

    schedule = _validated_schedule(
        evaluator=evaluator,
        optimized_residual_values=optimized_residual_values,
        optimized_jacobian_entries=optimized_jacobian_entries,
        residual_error=residual_error,
        jacobian_error=jacobian_error,
        residual_passed=residual_passed,
        jacobian_passed=jacobian_passed,
        tolerance=tolerance,
    )
    return ScheduledResidualJacobianValidation(
        schedule=schedule,
        passed=passed,
        tolerance=float(tolerance),
        residual_max_abs_error=residual_error,
        jacobian_max_abs_error=jacobian_error,
        jacobian_coordinates_match=coordinates_match,
        reference_residual_values=reference_residual_values,
        optimized_residual_values=optimized_residual_values,
        reference_jacobian_entries=reference_jacobian_entries,
        optimized_jacobian_entries=optimized_jacobian_entries,
    )


def format_scheduled_residual_jacobian_validation(
    validation: ScheduledResidualJacobianValidation,
) -> str:
    """Format scheduled residual/Jacobian validation deterministically."""

    return "\n".join(
        [
            "ScheduledResidualJacobianValidation",
            f"  passed={validation.passed}",
            f"  tolerance={validation.tolerance:g}",
            f"  residual_max_abs_error={validation.residual_max_abs_error:g}",
            f"  jacobian_max_abs_error={validation.jacobian_max_abs_error:g}",
            f"  jacobian_coordinates_match={validation.jacobian_coordinates_match}",
            "  reference_residual_values="
            + _format_values(validation.reference_residual_values),
            "  optimized_residual_values="
            + _format_values(validation.optimized_residual_values),
            "  reference_jacobian_entries="
            + _format_entries(validation.reference_jacobian_entries),
            "  optimized_jacobian_entries="
            + _format_entries(validation.optimized_jacobian_entries),
        ]
    )


def _build_prepared_residual_jacobian_schedule(
    *,
    contract: AssemblyContract,
    name: str,
    backend_name: str,
    residual_kernels: tuple[PreparedKernel, ...],
    jacobian_kernels: tuple[PreparedKernel, ...],
    validation_status: ValidationStatus,
) -> ExecutionSchedule:
    residual_slot_count = sum(kernel.slot_count for kernel in residual_kernels)
    jacobian_slot_count = sum(kernel.slot_count for kernel in jacobian_kernels)
    residual_task = ExecutionTask(
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
                kind="prepared_kernels",
                name="prepared_residual_kernels",
                detail=(
                    f"count={len(residual_kernels)} total_slots={residual_slot_count}"
                ),
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
        validation_status=validation_status,
    )
    jacobian_task = ExecutionTask(
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
                kind="prepared_kernels",
                name="prepared_jacobian_kernels",
                detail=(
                    f"count={len(jacobian_kernels)} total_slots={jacobian_slot_count}"
                ),
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
        validation_status=validation_status,
    )
    sparse_coordinate_task = ExecutionTask(
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
        validation_status=validation_status,
    )
    return ExecutionSchedule(
        name=name,
        tasks=(residual_task, jacobian_task, sparse_coordinate_task),
    )


def _validated_schedule(
    *,
    evaluator: ScheduledResidualJacobianEvaluator,
    optimized_residual_values: tuple[float, ...],
    optimized_jacobian_entries: tuple[tuple[int, int, float], ...],
    residual_error: float,
    jacobian_error: float,
    residual_passed: bool,
    jacobian_passed: bool,
    tolerance: float,
) -> ExecutionSchedule:
    residual_task, jacobian_task, sparse_coordinate_task = evaluator.schedule.tasks
    return replace(
        evaluator.schedule,
        tasks=(
            replace(
                residual_task,
                materialized=(
                    ScheduleItem(
                        kind="materialized_value",
                        name="residual_values",
                        detail="values=" + _format_values(optimized_residual_values),
                    ),
                    ScheduleItem(
                        kind="validation",
                        name="reference_residual_validation",
                        detail=(
                            f"passed={residual_passed} "
                            f"max_abs_error={residual_error:g} "
                            f"tolerance={tolerance:g}"
                        ),
                    ),
                ),
                validation_status=_validation_status(residual_passed),
            ),
            replace(
                jacobian_task,
                materialized=(
                    ScheduleItem(
                        kind="materialized_value",
                        name="jacobian_values",
                        detail="values=" + _format_entries(optimized_jacobian_entries),
                    ),
                    ScheduleItem(
                        kind="validation",
                        name="reference_jacobian_validation",
                        detail=(
                            f"passed={jacobian_passed} "
                            f"max_abs_error={jacobian_error:g} "
                            f"tolerance={tolerance:g}"
                        ),
                    ),
                ),
                validation_status=_validation_status(jacobian_passed),
            ),
            replace(
                sparse_coordinate_task,
                materialized=(
                    ScheduleItem(
                        kind="materialized_value",
                        name="sparse_coordinate_jacobian",
                        detail=(
                            f"shape={_format_shape(evaluator.contract.sparsity.shape)} "
                            f"entries={len(optimized_jacobian_entries)}"
                        ),
                    ),
                    ScheduleItem(
                        kind="validation",
                        name="reference_sparse_coordinate_validation",
                        detail=(
                            f"passed={jacobian_passed} "
                            f"max_abs_error={jacobian_error:g} "
                            f"tolerance={tolerance:g}"
                        ),
                    ),
                ),
                validation_status=_validation_status(jacobian_passed),
            ),
        ),
    )


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


def _scheduled_assembly_error(
    provenance: AssemblyProvenance,
    error: EvaluationError,
) -> AssemblyError:
    return AssemblyError(
        f"failed to assemble scheduled {_format_error_context(provenance)}: {error}"
    )


def _format_error_context(provenance: AssemblyProvenance) -> str:
    if provenance.column is None:
        return f"residual row {provenance.row} source node {provenance.source_node_id}"
    variable = "<none>"
    if provenance.variable is not None:
        variable = f"{provenance.variable.name}@{provenance.variable.node_id}"
    return (
        f"jacobian row {provenance.row} column {provenance.column} "
        f"variable {variable} source node {provenance.source_node_id} "
        f"derivative node {provenance.derivative_node_id}"
    )


def _entry_values(
    matrix: SparseMatrixAssembly,
) -> tuple[tuple[int, int, float], ...]:
    return tuple((entry.row, entry.column, entry.value) for entry in matrix.entries)


def _coordinates(
    entries: tuple[tuple[int, int, float], ...],
) -> tuple[tuple[int, int], ...]:
    return tuple((row, column) for row, column, _value in entries)


def _max_abs_error(
    reference_values: tuple[float, ...],
    optimized_values: tuple[float, ...],
) -> float:
    if len(reference_values) != len(optimized_values):
        return float("inf")
    if not reference_values:
        return 0.0
    return max(
        abs(reference - optimized)
        for reference, optimized in zip(reference_values, optimized_values, strict=True)
    )


def _entry_max_abs_error(
    reference_entries: tuple[tuple[int, int, float], ...],
    optimized_entries: tuple[tuple[int, int, float], ...],
    coordinates_match: bool,
) -> float:
    if not coordinates_match:
        return float("inf")
    return _max_abs_error(
        tuple(value for _row, _column, value in reference_entries),
        tuple(value for _row, _column, value in optimized_entries),
    )


def _validation_status(passed: bool) -> ValidationStatus:
    if passed:
        return ValidationStatus.REFERENCE_VALIDATED
    return ValidationStatus.NOT_RUN


def _format_shape(shape: tuple[int, int]) -> str:
    return f"({shape[0]}, {shape[1]})"


def _format_values(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def _format_entries(entries: tuple[tuple[int, int, float], ...]) -> str:
    return (
        "["
        + ", ".join(f"({row},{column})={value:g}" for row, column, value in entries)
        + "]"
    )
