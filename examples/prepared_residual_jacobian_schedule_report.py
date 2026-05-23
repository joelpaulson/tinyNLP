"""Prepared residual and Jacobian schedule report example."""

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

from tinynlp.nlp import build_assembly_contract
from tinynlp.schedule import (
    format_schedule_report,
    format_scheduled_residual_jacobian_validation,
    prepare_scheduled_residual_jacobian_evaluator,
    validate_scheduled_residual_jacobian_evaluator,
)


def prepared_residual_jacobian_schedule_report(horizon: int = 3) -> str:
    """Return a report for prepared scheduled residual and Jacobian execution."""

    case = flagship_chain_case(horizon=horizon)
    contract = build_assembly_contract(case.problem)
    evaluator = prepare_scheduled_residual_jacobian_evaluator(contract)
    validation = validate_scheduled_residual_jacobian_evaluator(
        evaluator,
        case.values,
    )
    return "\n\n".join(
        [
            format_scheduled_residual_jacobian_validation(validation),
            format_schedule_report(validation.schedule),
        ]
    )


def _main() -> None:
    print(prepared_residual_jacobian_schedule_report())


if __name__ == "__main__":
    _main()
