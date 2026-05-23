"""Prepared residual schedule report example."""

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

from tinynlp.nlp import build_assembly_contract
from tinynlp.schedule import (
    format_schedule_report,
    format_scheduled_residual_validation,
    prepare_scheduled_residual_evaluator,
    validate_scheduled_residual_evaluator,
)


def prepared_residual_schedule_report(horizon: int = 3) -> str:
    """Return a report for prepared scheduled chain residual evaluation."""

    case = chain_dynamics_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_evaluator(contract)
    validation = validate_scheduled_residual_evaluator(evaluator, case.values)
    return "\n\n".join(
        [
            format_scheduled_residual_validation(validation),
            format_schedule_report(validation.schedule),
        ]
    )


def _main() -> None:
    print(prepared_residual_schedule_report())


if __name__ == "__main__":
    _main()
