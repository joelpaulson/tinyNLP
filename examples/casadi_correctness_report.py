"""Optional CasADi correctness report example."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from chain_dynamics_problem import chain_dynamics_case

from tinynlp.bridges import (
    casadi_available,
    compare_problem_assembly,
    format_casadi_comparison,
)


def casadi_correctness_report(horizon: int = 3) -> str:
    """Return a skip-safe CasADi correctness report for chain dynamics."""

    lines = [
        "CasadiCorrectnessReport",
        "  purpose=correctness_only",
        "  comparison=problem_residual_jacobian_assembly",
        f"  problem=chain_dynamics horizon={horizon}",
    ]
    if not casadi_available():
        lines.extend(
            [
                "  available=False",
                "  status=skipped",
                "  install=uv sync --extra casadi",
            ]
        )
        return "\n".join(lines)

    case = chain_dynamics_case(horizon=horizon)
    comparison = compare_problem_assembly(case.problem, case.values)
    lines.extend(
        [
            "  available=True",
            format_casadi_comparison(comparison),
        ]
    )
    return "\n".join(lines)


def _main() -> None:
    print(casadi_correctness_report())


if __name__ == "__main__":
    _main()
