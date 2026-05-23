"""Flagship chain workflow built with ergonomic modeling helpers."""

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

from tinynlp.backends import evaluate
from tinynlp.ir import Graph
from tinynlp.nlp import (
    Problem,
    ResidualAssembly,
    SparseMatrixAssembly,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
    merge_value_maps,
    residual_block,
    sum_expressions,
    variable_array,
)


@dataclass(frozen=True)
class FlagshipChainCase:
    """Helper-built flagship chain case for the visible pipeline."""

    problem: Problem
    values: dict[str, float]
    references: tuple[float, ...]
    horizon: int
    dt: float
    a: float
    b: float
    c: float
    rho: float


def flagship_chain_case(
    horizon: int = 3,
    *,
    dt: float = 0.1,
    a: float = -0.2,
    b: float = 1.5,
    c: float = 0.05,
    rho: float = 0.1,
    references: tuple[float, ...] | None = None,
) -> FlagshipChainCase:
    """Build the flagship chain problem with small modeling helpers."""

    _validate_horizon(horizon)
    reference_values = _references(horizon, references)
    graph = Graph()
    states = variable_array(graph, "x", horizon + 1)
    controls = variable_array(graph, "u", horizon)

    residuals = [
        states[index + 1]
        - states[index]
        - dt
        * (
            (a * states[index])
            + (b * controls[index])
            + (c * states[index] * states[index])
        )
        for index in range(horizon)
    ]
    objective = sum_expressions(
        [
            (states[index] - reference_values[index])
            * (states[index] - reference_values[index])
            for index in range(horizon + 1)
        ]
        + [rho * control * control for control in controls]
    )
    values = merge_value_maps(
        states.value_map(reference_values),
        controls.value_map([0.0] * horizon),
    )
    return FlagshipChainCase(
        problem=Problem.from_blocks(
            (residual_block("chain_dynamics", residuals),),
            name="flagship_chain",
            objective=objective,
        ),
        values=values,
        references=reference_values,
        horizon=horizon,
        dt=float(dt),
        a=float(a),
        b=float(b),
        c=float(c),
        rho=float(rho),
    )


def evaluate_flagship_objective(case: FlagshipChainCase) -> float:
    """Evaluate the tracked objective metric for the flagship chain case."""

    if case.problem.objective is None:
        msg = "flagship chain case is missing an objective expression"
        raise ValueError(msg)
    return evaluate(case.problem.objective, case.values)


def assemble_flagship_residuals(case: FlagshipChainCase) -> ResidualAssembly:
    """Assemble flagship residual values through the current pipeline."""

    contract = build_assembly_contract(case.problem)
    return assemble_residuals(contract, case.values)


def assemble_flagship_jacobian(case: FlagshipChainCase) -> SparseMatrixAssembly:
    """Assemble flagship Jacobian coordinates through the current pipeline."""

    contract = build_assembly_contract(case.problem)
    return assemble_jacobian(contract, case.values)


def _references(
    horizon: int,
    references: tuple[float, ...] | None,
) -> tuple[float, ...]:
    if references is None:
        return tuple(1.0 + (0.1 * index) for index in range(horizon + 1))
    if len(references) != horizon + 1:
        msg = (
            "references must contain horizon + 1 values; "
            f"got {len(references)} for horizon {horizon}"
        )
        raise ValueError(msg)
    return tuple(float(reference) for reference in references)


def _validate_horizon(horizon: int) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        msg = "flagship chain horizon must be a positive integer"
        raise ValueError(msg)


def _main() -> None:
    case = flagship_chain_case()
    residuals = assemble_flagship_residuals(case)
    jacobian = assemble_flagship_jacobian(case)

    print(f"flagship_chain horizon={case.horizon}")
    print(
        f"variables={case.problem.variable_dimension} "
        f"residuals={case.problem.residual_dimension}"
    )
    print(
        "blocks=["
        + ", ".join(block.name for block in case.problem.residual_blocks)
        + "]"
    )
    print(f"objective={evaluate_flagship_objective(case):g}")
    print(
        "residuals=["
        + ", ".join(f"{value.value:g}" for value in residuals.values)
        + "]"
    )
    print(f"jacobian_shape={jacobian.shape} jacobian_entries={len(jacobian.entries)}")


if __name__ == "__main__":
    _main()
