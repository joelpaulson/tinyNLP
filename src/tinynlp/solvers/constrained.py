"""Simple constrained residual-reduction prototype."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt

from tinynlp.backends import EvaluationError, evaluate
from tinynlp.nlp import (
    AssemblyContract,
    AssemblyError,
    Problem,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
)
from tinynlp.solvers.kkt import KKTError, build_kkt_system
from tinynlp.solvers.linear import (
    DenseReferenceLinearSolver,
    LinearSolveError,
    LinearSolver,
)

DEFAULT_DAMPING_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625)


class SolverError(ValueError):
    """Raised when the prototype constrained solver cannot run."""


class SolverStatus(StrEnum):
    """Termination states for the constrained solver prototype."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    STEP_TOLERANCE = "step_tolerance"
    LINE_SEARCH_FAILED = "line_search_failed"


@dataclass(frozen=True)
class VariableValue:
    """One variable value in deterministic problem-variable order."""

    name: str
    value: float


@dataclass(frozen=True)
class IterationRecord:
    """One visible solver-iteration record."""

    iteration: int
    residual_norm: float
    step_norm: float | None
    kkt_solve_residual_norm: float | None
    objective_value: float | None
    accepted_step_length: float | None
    variables: tuple[VariableValue, ...]


@dataclass(frozen=True)
class SolverResult:
    """Result from the simple constrained solver prototype."""

    status: SolverStatus
    message: str
    final_values: dict[str, float]
    trace: tuple[IterationRecord, ...]

    @property
    def success(self) -> bool:
        """Whether the solver reached the residual tolerance."""

        return self.status is SolverStatus.CONVERGED


def solve_constraints(
    problem: Problem,
    values: Mapping[str, float],
    *,
    max_iterations: int = 10,
    residual_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    damping_steps: Sequence[float] = DEFAULT_DAMPING_STEPS,
    linear_solver: LinearSolver | None = None,
) -> SolverResult:
    """Reduce supported equality residuals with transparent KKT corrections.

    The prototype step solves ``[I J^T; J 0] [dx; lambda] = [0; -r]`` using the
    current identity/reference primal block. This is a residual-correction
    prototype, not a Hessian-backed NLP method.
    """

    _validate_options(
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance,
        step_tolerance=step_tolerance,
        damping_steps=damping_steps,
    )
    contract = _build_contract(problem)
    solver = linear_solver or DenseReferenceLinearSolver()
    current_values = _initial_values(contract, values)
    records: list[IterationRecord] = []

    for iteration in range(max_iterations):
        state = _evaluate_state(contract, current_values, iteration)
        if state.residual_norm <= residual_tolerance:
            records.append(state.record)
            return SolverResult(
                status=SolverStatus.CONVERGED,
                message=(
                    "converged: residual norm "
                    f"{state.residual_norm:g} <= {residual_tolerance:g}"
                ),
                final_values=current_values,
                trace=tuple(records),
            )

        step, kkt_residual_norm = _solve_step(contract, current_values, state, solver)
        step_norm = _norm(step)
        if step_norm <= step_tolerance:
            records.append(
                _record(
                    contract=contract,
                    values=current_values,
                    iteration=iteration,
                    residual_norm=state.residual_norm,
                    step_norm=step_norm,
                    kkt_solve_residual_norm=kkt_residual_norm,
                    objective_value=state.objective_value,
                    accepted_step_length=None,
                )
            )
            return SolverResult(
                status=SolverStatus.STEP_TOLERANCE,
                message=f"stopped: step norm {step_norm:g} <= {step_tolerance:g}",
                final_values=current_values,
                trace=tuple(records),
            )

        accepted = _accept_step(
            contract=contract,
            values=current_values,
            step=step,
            current_norm=state.residual_norm,
            damping_steps=damping_steps,
        )
        if accepted is None:
            records.append(
                _record(
                    contract=contract,
                    values=current_values,
                    iteration=iteration,
                    residual_norm=state.residual_norm,
                    step_norm=step_norm,
                    kkt_solve_residual_norm=kkt_residual_norm,
                    objective_value=state.objective_value,
                    accepted_step_length=None,
                )
            )
            return SolverResult(
                status=SolverStatus.LINE_SEARCH_FAILED,
                message="stopped: no damping step reduced the residual norm",
                final_values=current_values,
                trace=tuple(records),
            )

        current_values = accepted.values
        records.append(
            _record(
                contract=contract,
                values=current_values,
                iteration=iteration,
                residual_norm=accepted.residual_norm,
                step_norm=step_norm * accepted.step_length,
                kkt_solve_residual_norm=kkt_residual_norm,
                objective_value=_objective_value(contract.problem, current_values),
                accepted_step_length=accepted.step_length,
            )
        )
        if accepted.residual_norm <= residual_tolerance:
            return SolverResult(
                status=SolverStatus.CONVERGED,
                message=(
                    "converged: residual norm "
                    f"{accepted.residual_norm:g} <= {residual_tolerance:g}"
                ),
                final_values=current_values,
                trace=tuple(records),
            )

    return SolverResult(
        status=SolverStatus.MAX_ITERATIONS,
        message=f"stopped: reached max_iterations={max_iterations}",
        final_values=current_values,
        trace=tuple(records),
    )


def format_solver_trace(records: Iterable[IterationRecord]) -> str:
    """Format solver trace records deterministically."""

    lines = ["SolverTrace"]
    for record in records:
        parts = [
            f"iteration={record.iteration}",
            f"residual_norm={record.residual_norm:g}",
            f"step_norm={_format_optional(record.step_norm)}",
            f"kkt_solve_residual_norm={_format_optional(record.kkt_solve_residual_norm)}",
            f"objective_value={_format_optional(record.objective_value)}",
            f"accepted_step_length={_format_optional(record.accepted_step_length)}",
            "variables=["
            + ", ".join(f"{value.name}={value.value:g}" for value in record.variables)
            + "]",
        ]
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


@dataclass(frozen=True)
class _IterationState:
    residual_values: tuple[float, ...]
    residual_norm: float
    objective_value: float | None
    record: IterationRecord


@dataclass(frozen=True)
class _AcceptedStep:
    values: dict[str, float]
    residual_norm: float
    step_length: float


def _validate_options(
    *,
    max_iterations: int,
    residual_tolerance: float,
    step_tolerance: float,
    damping_steps: Sequence[float],
) -> None:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        msg = "max_iterations must be a positive integer"
        raise SolverError(msg)
    if max_iterations < 1:
        msg = "max_iterations must be a positive integer"
        raise SolverError(msg)
    try:
        residual_tolerance_value = float(residual_tolerance)
        step_tolerance_value = float(step_tolerance)
        damping_step_values = tuple(float(step) for step in damping_steps)
    except (TypeError, ValueError) as exc:
        msg = "solver tolerances and damping_steps must be numeric"
        raise SolverError(msg) from exc

    if residual_tolerance_value <= 0.0:
        msg = "residual_tolerance must be positive"
        raise SolverError(msg)
    if step_tolerance_value <= 0.0:
        msg = "step_tolerance must be positive"
        raise SolverError(msg)
    if not damping_step_values:
        msg = "damping_steps must contain at least one positive step"
        raise SolverError(msg)
    for step in damping_step_values:
        if step <= 0.0 or step > 1.0:
            msg = "damping_steps must be in the interval (0, 1]"
            raise SolverError(msg)


def _build_contract(problem: Problem) -> AssemblyContract:
    try:
        return build_assembly_contract(problem)
    except ValueError as exc:
        raise SolverError(f"failed to build assembly contract: {exc}") from exc


def _initial_values(
    contract: AssemblyContract,
    values: Mapping[str, float],
) -> dict[str, float]:
    try:
        current_values = {name: float(value) for name, value in values.items()}
    except (TypeError, ValueError) as exc:
        msg = "values must map variable names to numeric values"
        raise SolverError(msg) from exc
    missing = [
        variable.name
        for variable in contract.variables
        if variable.name not in current_values
    ]
    if missing:
        msg = "missing values for problem variables: " + ", ".join(missing)
        raise SolverError(msg)
    return current_values


def _evaluate_state(
    contract: AssemblyContract,
    values: dict[str, float],
    iteration: int,
) -> _IterationState:
    residual_values = _residual_values(contract, values)
    residual_norm = _norm(residual_values)
    objective_value = _objective_value(contract.problem, values)
    return _IterationState(
        residual_values=residual_values,
        residual_norm=residual_norm,
        objective_value=objective_value,
        record=_record(
            contract=contract,
            values=values,
            iteration=iteration,
            residual_norm=residual_norm,
            step_norm=None,
            kkt_solve_residual_norm=None,
            objective_value=objective_value,
            accepted_step_length=None,
        ),
    )


def _solve_step(
    contract: AssemblyContract,
    values: dict[str, float],
    state: _IterationState,
    linear_solver: LinearSolver,
) -> tuple[tuple[float, ...], float]:
    try:
        jacobian = assemble_jacobian(contract, values)
        system = build_kkt_system(jacobian)
        rhs = (0.0,) * contract.problem.variable_dimension + tuple(
            -value for value in state.residual_values
        )
        result = linear_solver.solve(system, rhs)
    except (AssemblyError, KKTError, LinearSolveError) as exc:
        raise SolverError(f"failed to compute constrained solver step: {exc}") from exc

    step = tuple(result.solution[: contract.problem.variable_dimension])
    return step, _norm(result.residual)


def _accept_step(
    *,
    contract: AssemblyContract,
    values: dict[str, float],
    step: tuple[float, ...],
    current_norm: float,
    damping_steps: Sequence[float],
) -> _AcceptedStep | None:
    for step_length in damping_steps:
        candidate_values = _apply_step(contract, values, step, step_length)
        candidate_norm = _norm(_residual_values(contract, candidate_values))
        if candidate_norm < current_norm:
            return _AcceptedStep(
                values=candidate_values,
                residual_norm=candidate_norm,
                step_length=float(step_length),
            )
    return None


def _apply_step(
    contract: AssemblyContract,
    values: dict[str, float],
    step: tuple[float, ...],
    step_length: float,
) -> dict[str, float]:
    updated = dict(values)
    for index, variable in enumerate(contract.variables):
        updated[variable.name] = values[variable.name] + (step_length * step[index])
    return updated


def _residual_values(
    contract: AssemblyContract,
    values: dict[str, float],
) -> tuple[float, ...]:
    try:
        return tuple(
            value.value for value in assemble_residuals(contract, values).values
        )
    except AssemblyError as exc:
        raise SolverError(f"failed to assemble residuals: {exc}") from exc


def _objective_value(problem: Problem, values: dict[str, float]) -> float | None:
    if problem.objective is None:
        return None
    try:
        return evaluate(problem.objective, values)
    except EvaluationError as exc:
        raise SolverError(f"failed to evaluate objective metric: {exc}") from exc


def _record(
    *,
    contract: AssemblyContract,
    values: dict[str, float],
    iteration: int,
    residual_norm: float,
    step_norm: float | None,
    kkt_solve_residual_norm: float | None,
    objective_value: float | None,
    accepted_step_length: float | None,
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        residual_norm=float(residual_norm),
        step_norm=step_norm,
        kkt_solve_residual_norm=kkt_solve_residual_norm,
        objective_value=objective_value,
        accepted_step_length=accepted_step_length,
        variables=tuple(
            VariableValue(name=variable.name, value=values[variable.name])
            for variable in contract.variables
        ),
    )


def _norm(values: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _format_optional(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:g}"
