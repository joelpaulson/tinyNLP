"""Flagship chain residual least-squares trace example."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from flagship_chain_modeling import flagship_chain_case

from tinynlp.solvers import (
    LeastSquaresResult,
    format_least_squares_trace,
    format_normal_equations,
    solve_least_squares,
)


def flagship_least_squares_result() -> LeastSquaresResult:
    """Run the transparent least-squares prototype on the flagship chain case."""

    case = flagship_chain_case(horizon=3)
    return solve_least_squares(case.problem, case.values, max_iterations=8)


def flagship_least_squares_report() -> str:
    """Return a deterministic human-facing least-squares trace report."""

    case = flagship_chain_case(horizon=3)
    result = solve_least_squares(case.problem, case.values, max_iterations=8)
    lines = [
        "flagship_least_squares",
        f"status={result.status}",
        f"success={result.success}",
        f"message={result.message}",
        (
            "dimensions="
            f"variables={case.problem.variable_dimension} "
            f"residuals={case.problem.residual_dimension}"
        ),
        (
            f"initial_residual_norm={result.trace[0].previous_residual_norm:g}"
            if result.trace and result.trace[0].previous_residual_norm is not None
            else "initial_residual_norm=<none>"
        ),
        (
            f"final_residual_norm={result.trace[-1].residual_norm:g}"
            if result.trace
            else "final_residual_norm=<none>"
        ),
        format_least_squares_trace(result.trace),
    ]
    last_normal_equations = _last_normal_equations(result)
    if last_normal_equations is not None:
        lines.append(format_normal_equations(last_normal_equations))
    return "\n".join(lines)


def _last_normal_equations(result: LeastSquaresResult):
    for record in reversed(result.trace):
        if record.normal_equations is not None:
            return record.normal_equations
    return None


def _main() -> None:
    print(flagship_least_squares_report())


if __name__ == "__main__":
    _main()
