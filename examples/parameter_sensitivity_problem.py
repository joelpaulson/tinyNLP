"""Tiny scalar-parameter sensitivity example."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_examples_dir = Path(__file__).resolve().parent
if str(_examples_dir) not in sys.path:
    sys.path.insert(0, str(_examples_dir))

from _bootstrap import ensure_repo_src_on_path

ensure_repo_src_on_path()

from tinynlp.ir import Graph
from tinynlp.nlp import Problem
from tinynlp.solvers import SensitivityResult, format_sensitivity, implicit_sensitivity


@dataclass(frozen=True)
class ParameterSensitivityCase:
    """Minimal parameterized residual system with hand-checkable sensitivity."""

    problem: Problem
    values: dict[str, float]
    parameter: str
    solve_variables: tuple[str, ...]
    expected_sensitivities: dict[str, float]


def parameter_sensitivity_case(
    parameter_value: float = 1.0,
) -> ParameterSensitivityCase:
    """Build ``x + p - 3 = 0`` and ``y - 2*p = 0``.

    At ``p = 1``, the solution is ``x = 2`` and ``y = 2`` with sensitivities
    ``dx/dp = -1`` and ``dy/dp = 2``.
    """

    p_value = float(parameter_value)
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    p = graph.variable("p")
    problem = Problem.from_residuals(
        [
            x + p - 3,
            y - (2 * p),
        ],
        name="parameter_sensitivity",
    )
    return ParameterSensitivityCase(
        problem=problem,
        values={
            "x": 3.0 - p_value,
            "y": 2.0 * p_value,
            "p": p_value,
        },
        parameter="p",
        solve_variables=("x", "y"),
        expected_sensitivities={
            "x": -1.0,
            "y": 2.0,
        },
    )


def solve_only_problem(
    parameter_value: float,
) -> tuple[Problem, dict[str, float]]:
    """Build the same residuals with the parameter fixed as a constant."""

    p_value = float(parameter_value)
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    problem = Problem.from_residuals(
        [
            x + p_value - 3,
            y - (2.0 * p_value),
        ],
        name="parameter_sensitivity_solve_only",
    )
    return problem, {
        "x": 3.0 - p_value,
        "y": 2.0 * p_value,
    }


def evaluate_parameter_sensitivity(
    parameter_value: float = 1.0,
) -> SensitivityResult:
    """Evaluate the example sensitivity through the current KKT path."""

    case = parameter_sensitivity_case(parameter_value)
    return implicit_sensitivity(
        case.problem,
        case.values,
        parameter=case.parameter,
        solve_variables=case.solve_variables,
    )


def _main() -> None:
    case = parameter_sensitivity_case()
    result = evaluate_parameter_sensitivity()
    print(
        f"parameter_sensitivity parameter={case.parameter} "
        f"variables={case.solve_variables}"
    )
    print(format_sensitivity(result))


if __name__ == "__main__":
    _main()
