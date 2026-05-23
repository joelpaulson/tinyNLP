"""Scheduler-backed residual evaluation helpers."""

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
    ResidualAssembly,
    ResidualValue,
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
class ScheduledResidualEvaluator:
    """Prepared evaluator for one scheduled residual-evaluation task."""

    name: str
    contract: AssemblyContract
    backend: PreparedKernelBackend
    prepared_kernels: tuple[PreparedKernel, ...]
    schedule: ExecutionSchedule

    def evaluate(self, values: Mapping[str, float]) -> ResidualAssembly:
        """Evaluate scheduled residuals with the prepared backend."""

        assembled: list[ResidualValue] = []
        for term, kernel in zip(
            self.contract.residual_terms,
            self.prepared_kernels,
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
            assembled.append(
                ResidualValue(
                    row=term.row,
                    value=value,
                    provenance=provenance,
                )
            )
        return ResidualAssembly(values=tuple(assembled))


@dataclass(frozen=True)
class ScheduledResidualValidation:
    """Reference validation result for a scheduled residual evaluator."""

    schedule: ExecutionSchedule
    passed: bool
    tolerance: float
    max_abs_error: float
    reference_values: tuple[float, ...]
    optimized_values: tuple[float, ...]


def prepare_scheduled_residual_evaluator(
    contract: AssemblyContract,
    *,
    name: str | None = None,
) -> ScheduledResidualEvaluator:
    """Prepare residual KernelPlans for scheduled repeated execution."""

    backend = PreparedKernelBackend()
    prepared_kernels = tuple(
        prepare_kernel(term.kernel_plan) for term in contract.residual_terms
    )
    evaluator_name = name or f"{contract.problem.name}_scheduled_residuals"
    schedule = _build_prepared_residual_schedule(
        contract=contract,
        name=evaluator_name,
        backend_name=backend.name,
        prepared_kernels=prepared_kernels,
        validation_status=ValidationStatus.NOT_RUN,
    )
    return ScheduledResidualEvaluator(
        name=evaluator_name,
        contract=contract,
        backend=backend,
        prepared_kernels=prepared_kernels,
        schedule=schedule,
    )


def validate_scheduled_residual_evaluator(
    evaluator: ScheduledResidualEvaluator,
    values: Mapping[str, float],
    *,
    tolerance: float = 1e-12,
) -> ScheduledResidualValidation:
    """Validate scheduled residual execution against reference assembly."""

    reference = assemble_residuals(evaluator.contract, values)
    optimized = evaluator.evaluate(values)
    reference_values = tuple(value.value for value in reference.values)
    optimized_values = tuple(value.value for value in optimized.values)
    max_abs_error = _max_abs_error(reference_values, optimized_values)
    passed = max_abs_error <= tolerance
    schedule = _validated_schedule(
        evaluator=evaluator,
        optimized_values=optimized_values,
        max_abs_error=max_abs_error,
        tolerance=tolerance,
        passed=passed,
    )
    return ScheduledResidualValidation(
        schedule=schedule,
        passed=passed,
        tolerance=float(tolerance),
        max_abs_error=max_abs_error,
        reference_values=reference_values,
        optimized_values=optimized_values,
    )


def format_scheduled_residual_validation(
    validation: ScheduledResidualValidation,
) -> str:
    """Format scheduled residual validation deterministically."""

    return "\n".join(
        [
            "ScheduledResidualValidation",
            f"  passed={validation.passed}",
            f"  tolerance={validation.tolerance:g}",
            f"  max_abs_error={validation.max_abs_error:g}",
            "  reference_values=" + _format_values(validation.reference_values),
            "  optimized_values=" + _format_values(validation.optimized_values),
        ]
    )


def _build_prepared_residual_schedule(
    *,
    contract: AssemblyContract,
    name: str,
    backend_name: str,
    prepared_kernels: tuple[PreparedKernel, ...],
    validation_status: ValidationStatus,
) -> ExecutionSchedule:
    slot_count = sum(kernel.slot_count for kernel in prepared_kernels)
    task = ExecutionTask(
        task_id="t000",
        stage=ExecutionStage.EVALUATE_RESIDUALS,
        label=f"evaluate residuals for {contract.problem.name}",
        inputs=(
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
        ),
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
            ScheduleItem(
                kind="assembly_contract",
                name=contract.problem.name,
                detail=(
                    f"variables={len(contract.variables)} "
                    f"residual_terms={len(contract.residual_terms)} "
                    f"jacobian_terms={len(contract.jacobian_terms)}"
                ),
            ),
            ScheduleItem(
                kind="prepared_kernels",
                name="prepared_residual_kernels",
                detail=(f"count={len(prepared_kernels)} total_slots={slot_count}"),
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
    return ExecutionSchedule(name=name, tasks=(task,))


def _validated_schedule(
    *,
    evaluator: ScheduledResidualEvaluator,
    optimized_values: tuple[float, ...],
    max_abs_error: float,
    tolerance: float,
    passed: bool,
) -> ExecutionSchedule:
    task = evaluator.schedule.tasks[0]
    return replace(
        evaluator.schedule,
        tasks=(
            replace(
                task,
                materialized=(
                    ScheduleItem(
                        kind="materialized_value",
                        name="residual_values",
                        detail="values=" + _format_values(optimized_values),
                    ),
                    ScheduleItem(
                        kind="validation",
                        name="reference_residual_validation",
                        detail=(
                            f"passed={passed} "
                            f"max_abs_error={max_abs_error:g} "
                            f"tolerance={tolerance:g}"
                        ),
                    ),
                ),
                validation_status=(
                    ValidationStatus.REFERENCE_VALIDATED
                    if passed
                    else ValidationStatus.NOT_RUN
                ),
            ),
        ),
    )


def _scheduled_assembly_error(
    provenance: AssemblyProvenance,
    error: EvaluationError,
) -> AssemblyError:
    return AssemblyError(
        "failed to assemble scheduled residual "
        f"row {provenance.row} source node {provenance.source_node_id}: {error}"
    )


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


def _format_values(values: tuple[float, ...]) -> str:
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"
