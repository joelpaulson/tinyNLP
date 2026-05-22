"""Chain dynamics problem for pre-solver pipeline smoke tests."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


def _ensure_repo_src_on_path() -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))


_ensure_repo_src_on_path()

from tinynlp.backends import evaluate
from tinynlp.ir import Expr, Graph
from tinynlp.nlp import (
    Problem,
    ResidualAssembly,
    SparseMatrixAssembly,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
)
from tinynlp.solvers import KKTSystem, build_kkt_system


@dataclass(frozen=True)
class ChainDynamicsCase:
    """Canonical chain dynamics case for assembly and KKT smoke tests."""

    problem: Problem
    values: dict[str, float]
    references: tuple[float, ...]
    horizon: int
    dt: float
    a: float
    b: float
    c: float
    rho: float


def chain_dynamics_case(
    horizon: int = 3,
    *,
    dt: float = 0.1,
    a: float = -0.2,
    b: float = 1.5,
    c: float = 0.05,
    rho: float = 0.1,
    references: tuple[float, ...] | None = None,
) -> ChainDynamicsCase:
    """Build the canonical chain dynamics problem.

    The residuals are:

        x_{k+1} - x_k - dt*(a*x_k + b*u_k + c*x_k*x_k)

    The objective is tracked as a metric for the current pipeline:

        sum_k (x_k - r_k)^2 + sum_k rho*u_k^2
    """

    _validate_horizon(horizon)
    reference_values = _references(horizon, references)
    graph = Graph()
    states = [graph.variable(f"x{index}") for index in range(horizon + 1)]
    controls = [graph.variable(f"u{index}") for index in range(horizon)]

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
    objective = _sum_expressions(
        [
            (states[index] - reference_values[index])
            * (states[index] - reference_values[index])
            for index in range(horizon + 1)
        ]
        + [rho * control * control for control in controls]
    )
    values = {
        **{f"x{index}": reference_values[index] for index in range(horizon + 1)},
        **{f"u{index}": 0.0 for index in range(horizon)},
    }
    return ChainDynamicsCase(
        problem=Problem.from_residuals(
            residuals,
            name="chain_dynamics",
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


def evaluate_chain_objective(case: ChainDynamicsCase) -> float:
    """Evaluate the tracked objective metric for a chain dynamics case."""

    if case.problem.objective is None:
        msg = "chain dynamics case is missing an objective expression"
        raise ValueError(msg)
    return evaluate(case.problem.objective, case.values)


def assemble_chain_residuals(case: ChainDynamicsCase) -> ResidualAssembly:
    """Assemble chain residual values through the current assembly pipeline."""

    contract = build_assembly_contract(case.problem)
    return assemble_residuals(contract, case.values)


def assemble_chain_jacobian(case: ChainDynamicsCase) -> SparseMatrixAssembly:
    """Assemble chain Jacobian coordinates through the current pipeline."""

    contract = build_assembly_contract(case.problem)
    return assemble_jacobian(contract, case.values)


def build_chain_kkt(case: ChainDynamicsCase) -> KKTSystem:
    """Build the reference KKT system for the chain Jacobian."""

    return build_kkt_system(assemble_chain_jacobian(case))


def _sum_expressions(expressions: list[Expr]) -> Expr:
    result = expressions[0]
    for expression in expressions[1:]:
        result = result + expression
    return result


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
        msg = "chain dynamics horizon must be a positive integer"
        raise ValueError(msg)


def _main() -> None:
    case = chain_dynamics_case()
    residuals = assemble_chain_residuals(case)
    jacobian = assemble_chain_jacobian(case)
    kkt = build_chain_kkt(case)

    print(f"chain_dynamics horizon={case.horizon}")
    print(
        f"variables={case.problem.variable_dimension} "
        f"residuals={case.problem.residual_dimension}"
    )
    print(f"objective={evaluate_chain_objective(case):g}")
    print(
        "residuals=["
        + ", ".join(f"{value.value:g}" for value in residuals.values)
        + "]"
    )
    print(f"jacobian_shape={jacobian.shape} jacobian_entries={len(jacobian.entries)}")
    print(f"kkt_shape={kkt.shape} kkt_entries={len(kkt.entries)}")


if __name__ == "__main__":
    _main()
