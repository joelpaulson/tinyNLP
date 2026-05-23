"""Scheduled sensitivity report example."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from parameter_sensitivity_problem import evaluate_parameter_sensitivity

from tinynlp.schedule import (
    ExecutionSchedule,
    build_sensitivity_schedule,
    format_schedule_report,
)


def parameter_sensitivity_schedule(
    parameter_value: float = 1.0,
) -> ExecutionSchedule:
    """Build a sensitivity schedule for the parameter sensitivity example."""

    result = evaluate_parameter_sensitivity(parameter_value=parameter_value)
    return build_sensitivity_schedule(
        result,
        name="parameter_sensitivity_schedule",
    )


def parameter_sensitivity_schedule_report(
    parameter_value: float = 1.0,
) -> str:
    """Return an audit report for the parameter sensitivity path."""

    return format_schedule_report(
        parameter_sensitivity_schedule(parameter_value=parameter_value)
    )


def _main() -> None:
    print(parameter_sensitivity_schedule_report())


if __name__ == "__main__":
    _main()
