"""Start-here flagship walkthrough for the current tinyNLP pipeline."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from casadi_correctness_report import casadi_correctness_report
from flagship_chain_modeling import (
    assemble_flagship_jacobian,
    assemble_flagship_residuals,
    evaluate_flagship_objective,
    flagship_chain_case,
)

from tinynlp.nlp import build_assembly_contract
from tinynlp.schedule import (
    prepare_scheduled_residual_jacobian_evaluator,
    validate_scheduled_residual_jacobian_evaluator,
)
from tinynlp.solvers import solve_least_squares


def flagship_start_here_report(horizon: int = 3) -> str:
    """Return a compact end-to-end report for the flagship chain workflow."""

    case = flagship_chain_case(horizon=horizon)
    residuals = assemble_flagship_residuals(case)
    jacobian = assemble_flagship_jacobian(case)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)
    validation = validate_scheduled_residual_jacobian_evaluator(
        evaluator,
        case.values,
    )
    least_squares = solve_least_squares(case.problem, case.values, max_iterations=8)
    casadi_report = casadi_correctness_report(horizon=horizon)

    lines = [
        "FlagshipStartHere",
        "model:",
        f"  problem={case.problem.name}",
        f"  horizon={case.horizon}",
        (
            "  dimensions="
            f"variables={case.problem.variable_dimension} "
            f"residuals={case.problem.residual_dimension}"
        ),
        (
            "  residual_blocks=["
            + ", ".join(block.name for block in case.problem.residual_blocks)
            + "]"
        ),
        f"  tracked_objective_value={evaluate_flagship_objective(case):g}",
        "assembly:",
        "  residual_values="
        + _format_values(value.value for value in residuals.values),
        f"  jacobian_shape={jacobian.shape}",
        f"  jacobian_entries={len(jacobian.entries)}",
        "scheduled_execution:",
        "  backend=prepared-python",
        "  stage_group=residual_plus_jacobian",
        f"  validation_passed={validation.passed}",
        f"  residual_max_abs_error={validation.residual_max_abs_error:g}",
        f"  jacobian_max_abs_error={validation.jacobian_max_abs_error:g}",
        f"  jacobian_coordinates_match={validation.jacobian_coordinates_match}",
        "  schedule_tasks=["
        + ", ".join(
            f"{task.task_id}:{task.stage}" for task in validation.schedule.tasks
        )
        + "]",
        "least_squares_reference:",
        f"  status={least_squares.status}",
        f"  success={least_squares.success}",
        f"  iterations={len(least_squares.trace)}",
        f"  initial_residual_norm={_initial_residual_norm(least_squares):g}",
        f"  final_residual_norm={_final_residual_norm(least_squares):g}",
        "casadi_correctness:",
        *("  " + line for line in casadi_report.splitlines()),
        "benchmark_evidence:",
        (
            "  scheduled_residual_summary="
            "benchmarks/results/scheduled_residual_evaluation.md"
        ),
        (
            "  flagship_residual_jacobian_summary="
            "benchmarks/results/flagship_residual_jacobian_evaluation.md"
        ),
        (
            "  residual_jacobian_command="
            "uv run pytest benchmarks/test_scheduler_residual_jacobian_benchmark.py "
            "--benchmark-json /private/tmp/tinynlp-flagship-rj-benchmark.json"
        ),
        "boundaries:",
        (
            "  narrow_claims=scheduled residual and scheduled residual+Jacobian "
            "stages only"
        ),
        (
            "  not_claimed=solver speed, CasADi speed, KKT speed, sensitivity "
            "speed, package-wide speed, production optimization behavior, GPU support"
        ),
        "next_step:",
        "  write the next roadmap before adding new runtime milestones",
    ]
    return "\n".join(lines)


def _format_values(values) -> str:
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def _initial_residual_norm(result) -> float:
    if not result.trace or result.trace[0].previous_residual_norm is None:
        return 0.0
    return result.trace[0].previous_residual_norm


def _final_residual_norm(result) -> float:
    if not result.trace:
        return 0.0
    return result.trace[-1].residual_norm


def _main() -> None:
    print(flagship_start_here_report())


if __name__ == "__main__":
    _main()
