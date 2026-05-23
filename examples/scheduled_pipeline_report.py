"""Scheduled pipeline report example for chain dynamics."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from chain_dynamics_problem import (
    assemble_chain_jacobian,
    assemble_chain_residuals,
    build_chain_kkt,
    chain_dynamics_case,
)

from tinynlp.nlp import ResidualAssembly, SparseMatrixAssembly
from tinynlp.schedule import (
    ExecutionSchedule,
    ExecutionStage,
    ScheduleItem,
    ValidationStatus,
    build_kkt_assembly_schedule,
    build_problem_assembly_schedule,
    format_pipeline_report,
)
from tinynlp.solvers import KKTSystem


def chain_pipeline_schedules(horizon: int = 3) -> tuple[ExecutionSchedule, ...]:
    """Build annotated schedules for the chain assembly and KKT path."""

    case = chain_dynamics_case(horizon=horizon)
    residuals = assemble_chain_residuals(case)
    jacobian = assemble_chain_jacobian(case)
    kkt = build_chain_kkt(case)

    assembly_schedule = _annotate_assembly_schedule(
        build_problem_assembly_schedule(
            case.problem,
            name="chain_dynamics_assembly",
        ),
        residuals,
        jacobian,
    )
    kkt_schedule = _annotate_kkt_schedule(
        build_kkt_assembly_schedule(
            jacobian,
            name="chain_dynamics_kkt",
            dependencies=("t002",),
        ),
        kkt,
    )
    return (assembly_schedule, kkt_schedule)


def chain_pipeline_report(horizon: int = 3) -> str:
    """Return an audit report for the chain assembly and KKT path."""

    return format_pipeline_report(
        chain_pipeline_schedules(horizon=horizon),
        title="ChainDynamicsScheduledPipelineReport",
    )


def _annotate_assembly_schedule(
    schedule: ExecutionSchedule,
    residuals: ResidualAssembly,
    jacobian: SparseMatrixAssembly,
) -> ExecutionSchedule:
    tasks = []
    for task in schedule.tasks:
        if task.stage is ExecutionStage.EVALUATE_RESIDUALS:
            tasks.append(
                replace(
                    task,
                    materialized=(
                        ScheduleItem(
                            kind="materialized_value",
                            name="residual_values",
                            detail=_format_residual_values(residuals),
                        ),
                    ),
                    validation_status=ValidationStatus.REFERENCE_VALIDATED,
                )
            )
        elif task.stage is ExecutionStage.EVALUATE_JACOBIAN:
            tasks.append(
                replace(
                    task,
                    materialized=(
                        ScheduleItem(
                            kind="materialized_value",
                            name="jacobian_values",
                            detail=_format_coordinate_values(jacobian),
                        ),
                    ),
                    validation_status=ValidationStatus.REFERENCE_VALIDATED,
                )
            )
        elif task.stage is ExecutionStage.ASSEMBLE_SPARSE_COORDINATE_JACOBIAN:
            tasks.append(
                replace(
                    task,
                    materialized=(
                        ScheduleItem(
                            kind="materialized_value",
                            name="sparse_coordinate_jacobian",
                            detail=(
                                f"shape={_format_shape(jacobian.shape)} "
                                f"entries={len(jacobian.entries)}"
                            ),
                        ),
                    ),
                    validation_status=ValidationStatus.REFERENCE_VALIDATED,
                )
            )
        else:
            tasks.append(task)
    return replace(schedule, tasks=tuple(tasks))


def _annotate_kkt_schedule(
    schedule: ExecutionSchedule,
    kkt: KKTSystem,
) -> ExecutionSchedule:
    task = schedule.tasks[0]
    return replace(
        schedule,
        tasks=(
            replace(
                task,
                materialized=(
                    ScheduleItem(
                        kind="materialized_value",
                        name="kkt_entries",
                        detail=(
                            f"shape={_format_shape(kkt.shape)} "
                            f"entries={len(kkt.entries)} "
                            f"values={_format_kkt_values(kkt)}"
                        ),
                    ),
                ),
                validation_status=ValidationStatus.REFERENCE_VALIDATED,
            ),
        ),
    )


def _format_residual_values(residuals: ResidualAssembly) -> str:
    return (
        "values=[" + ", ".join(f"{value.value:g}" for value in residuals.values) + "]"
    )


def _format_coordinate_values(matrix: SparseMatrixAssembly) -> str:
    values = ", ".join(
        f"({entry.row},{entry.column})={entry.value:g}" for entry in matrix.entries
    )
    return f"entries={len(matrix.entries)} values=[{values}]"


def _format_kkt_values(kkt: KKTSystem) -> str:
    return (
        "["
        + ", ".join(
            f"({entry.row},{entry.column})={entry.value:g}" for entry in kkt.entries
        )
        + "]"
    )


def _format_shape(shape: tuple[int, int]) -> str:
    return f"({shape[0]}, {shape[1]})"


def _main() -> None:
    print(chain_pipeline_report())


if __name__ == "__main__":
    _main()
