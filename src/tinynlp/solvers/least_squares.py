"""Transparent residual least-squares reference prototype."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from numbers import Real

from tinynlp.backends import EvaluationError, evaluate
from tinynlp.ir import VariableRef
from tinynlp.nlp import (
    AssemblyContract,
    AssemblyError,
    Problem,
    SparseMatrixAssembly,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
    to_dense,
)
from tinynlp.solvers.constrained import VariableValue

DEFAULT_LEAST_SQUARES_DAMPING_STEPS = (1.0, 0.5, 0.25, 0.125, 0.0625)


class LeastSquaresError(ValueError):
    """Raised when the reference least-squares prototype cannot run."""


class LeastSquaresStatus(StrEnum):
    """Termination states for the residual least-squares prototype."""

    CONVERGED = "converged"
    MAX_ITERATIONS = "max_iterations"
    STEP_TOLERANCE = "step_tolerance"
    GRADIENT_TOLERANCE = "gradient_tolerance"
    LINE_SEARCH_FAILED = "line_search_failed"


@dataclass(frozen=True)
class NormalEquationSystem:
    """Dense reference normal equations with source metadata."""

    shape: tuple[int, int]
    coefficients: tuple[tuple[float, ...], ...]
    rhs: tuple[float, ...]
    regularization: float
    source_jacobian_shape: tuple[int, int]
    residual_dimension: int
    variables: tuple[VariableRef, ...]
    gradient: tuple[float, ...]
    linear_solve_residual: tuple[float, ...] | None = None


@dataclass(frozen=True)
class LeastSquaresIterationRecord:
    """One visible residual least-squares iteration record."""

    iteration: int
    previous_residual_norm: float | None
    residual_norm: float
    least_squares_value: float
    gradient_norm: float
    step_norm: float | None
    accepted_step_length: float | None
    linear_solve_residual_norm: float | None
    tracked_objective_value: float | None
    regularization: float
    jacobian_shape: tuple[int, int]
    normal_equation_shape: tuple[int, int]
    variables: tuple[VariableValue, ...]
    normal_equations: NormalEquationSystem | None


@dataclass(frozen=True)
class LeastSquaresResult:
    """Result from the transparent residual least-squares prototype."""

    status: LeastSquaresStatus
    message: str
    final_values: dict[str, object]
    trace: tuple[LeastSquaresIterationRecord, ...]

    @property
    def success(self) -> bool:
        """Whether the solve reached the residual tolerance."""

        return self.status is LeastSquaresStatus.CONVERGED


@dataclass(frozen=True)
class _LeastSquaresOptions:
    max_iterations: int
    residual_tolerance: float
    step_tolerance: float
    gradient_tolerance: float
    regularization: float
    damping_steps: tuple[float, ...]


@dataclass(frozen=True)
class _IterationState:
    residual_values: tuple[float, ...]
    residual_norm: float
    least_squares_value: float
    tracked_objective_value: float | None
    jacobian: SparseMatrixAssembly
    normal_equations: NormalEquationSystem
    gradient_norm: float


@dataclass(frozen=True)
class _LinearStep:
    step: tuple[float, ...]
    residual: tuple[float, ...]
    residual_norm: float
    normal_equations: NormalEquationSystem


@dataclass(frozen=True)
class _AcceptedStep:
    values: dict[str, object]
    residual_norm: float
    least_squares_value: float
    step_length: float


def solve_least_squares(
    problem: Problem,
    values: Mapping[str, object],
    *,
    max_iterations: int = 10,
    residual_tolerance: float = 1e-8,
    step_tolerance: float = 1e-12,
    gradient_tolerance: float = 1e-10,
    regularization: float = 1e-6,
    damping_steps: Sequence[float] = DEFAULT_LEAST_SQUARES_DAMPING_STEPS,
) -> LeastSquaresResult:
    """Reduce supported residual least-squares with visible normal equations.

    The prototype solves ``(J^T J + regularization*I) dx = -J^T r`` and accepts
    damped steps that reduce ``0.5 * ||r||^2``. The problem objective, when
    present, is reported as a tracked metric only.
    """

    options = _validate_options(
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance,
        step_tolerance=step_tolerance,
        gradient_tolerance=gradient_tolerance,
        regularization=regularization,
        damping_steps=damping_steps,
    )
    contract = _build_contract(problem)
    current_values = _initial_values(contract, values)
    records: list[LeastSquaresIterationRecord] = []

    for iteration in range(options.max_iterations):
        state = _evaluate_state(contract, current_values, iteration, options)
        if state.residual_norm <= options.residual_tolerance:
            records.append(
                _record(
                    contract=contract,
                    values=current_values,
                    iteration=iteration,
                    previous_residual_norm=None,
                    residual_norm=state.residual_norm,
                    least_squares_value=state.least_squares_value,
                    gradient_norm=state.gradient_norm,
                    step_norm=None,
                    accepted_step_length=None,
                    linear_solve_residual_norm=None,
                    tracked_objective_value=state.tracked_objective_value,
                    normal_equations=state.normal_equations,
                )
            )
            return LeastSquaresResult(
                status=LeastSquaresStatus.CONVERGED,
                message=(
                    "converged: residual norm "
                    f"{state.residual_norm:g} <= {options.residual_tolerance:g}"
                ),
                final_values=dict(current_values),
                trace=tuple(records),
            )

        if state.gradient_norm <= options.gradient_tolerance:
            records.append(
                _record(
                    contract=contract,
                    values=current_values,
                    iteration=iteration,
                    previous_residual_norm=None,
                    residual_norm=state.residual_norm,
                    least_squares_value=state.least_squares_value,
                    gradient_norm=state.gradient_norm,
                    step_norm=None,
                    accepted_step_length=None,
                    linear_solve_residual_norm=None,
                    tracked_objective_value=state.tracked_objective_value,
                    normal_equations=state.normal_equations,
                )
            )
            return LeastSquaresResult(
                status=LeastSquaresStatus.GRADIENT_TOLERANCE,
                message=(
                    "stopped: gradient norm "
                    f"{state.gradient_norm:g} <= {options.gradient_tolerance:g}"
                ),
                final_values=dict(current_values),
                trace=tuple(records),
            )

        linear_step = _solve_normal_equations(state.normal_equations)
        step_norm = _norm(linear_step.step)
        normal_equations = linear_step.normal_equations
        if step_norm <= options.step_tolerance:
            records.append(
                _record(
                    contract=contract,
                    values=current_values,
                    iteration=iteration,
                    previous_residual_norm=None,
                    residual_norm=state.residual_norm,
                    least_squares_value=state.least_squares_value,
                    gradient_norm=state.gradient_norm,
                    step_norm=step_norm,
                    accepted_step_length=None,
                    linear_solve_residual_norm=linear_step.residual_norm,
                    tracked_objective_value=state.tracked_objective_value,
                    normal_equations=normal_equations,
                )
            )
            return LeastSquaresResult(
                status=LeastSquaresStatus.STEP_TOLERANCE,
                message=(
                    f"stopped: step norm {step_norm:g} <= {options.step_tolerance:g}"
                ),
                final_values=dict(current_values),
                trace=tuple(records),
            )

        accepted = _accept_step(
            contract=contract,
            values=current_values,
            step=linear_step.step,
            current_value=state.least_squares_value,
            damping_steps=options.damping_steps,
        )
        if accepted is None:
            records.append(
                _record(
                    contract=contract,
                    values=current_values,
                    iteration=iteration,
                    previous_residual_norm=None,
                    residual_norm=state.residual_norm,
                    least_squares_value=state.least_squares_value,
                    gradient_norm=state.gradient_norm,
                    step_norm=step_norm,
                    accepted_step_length=None,
                    linear_solve_residual_norm=linear_step.residual_norm,
                    tracked_objective_value=state.tracked_objective_value,
                    normal_equations=normal_equations,
                )
            )
            return LeastSquaresResult(
                status=LeastSquaresStatus.LINE_SEARCH_FAILED,
                message="stopped: no damping step reduced the least-squares value",
                final_values=dict(current_values),
                trace=tuple(records),
            )

        current_values = accepted.values
        records.append(
            _record(
                contract=contract,
                values=current_values,
                iteration=iteration,
                previous_residual_norm=state.residual_norm,
                residual_norm=accepted.residual_norm,
                least_squares_value=accepted.least_squares_value,
                gradient_norm=state.gradient_norm,
                step_norm=step_norm * accepted.step_length,
                accepted_step_length=accepted.step_length,
                linear_solve_residual_norm=linear_step.residual_norm,
                tracked_objective_value=_objective_value(
                    contract.problem,
                    current_values,
                ),
                normal_equations=normal_equations,
            )
        )
        if accepted.residual_norm <= options.residual_tolerance:
            return LeastSquaresResult(
                status=LeastSquaresStatus.CONVERGED,
                message=(
                    "converged: residual norm "
                    f"{accepted.residual_norm:g} <= {options.residual_tolerance:g}"
                ),
                final_values=dict(current_values),
                trace=tuple(records),
            )

    return LeastSquaresResult(
        status=LeastSquaresStatus.MAX_ITERATIONS,
        message=f"stopped: reached max_iterations={options.max_iterations}",
        final_values=dict(current_values),
        trace=tuple(records),
    )


def format_least_squares_trace(
    records: Iterable[LeastSquaresIterationRecord],
) -> str:
    """Format least-squares trace records deterministically."""

    lines = ["LeastSquaresTrace"]
    for record in records:
        parts = [
            f"iteration={record.iteration}",
            f"previous_residual_norm={_format_optional(record.previous_residual_norm)}",
            f"residual_norm={record.residual_norm:g}",
            f"least_squares_value={record.least_squares_value:g}",
            f"gradient_norm={record.gradient_norm:g}",
            f"step_norm={_format_optional(record.step_norm)}",
            f"accepted_step_length={_format_optional(record.accepted_step_length)}",
            "linear_solve_residual_norm="
            f"{_format_optional(record.linear_solve_residual_norm)}",
            f"tracked_objective_value={_format_optional(record.tracked_objective_value)}",
            f"regularization={record.regularization:g}",
            f"jacobian_shape={record.jacobian_shape}",
            f"normal_equation_shape={record.normal_equation_shape}",
            "variables=["
            + ", ".join(f"{value.name}={value.value:g}" for value in record.variables)
            + "]",
        ]
        lines.append("  " + " ".join(parts))
    return "\n".join(lines)


def format_normal_equations(system: NormalEquationSystem) -> str:
    """Format normal-equation metadata and coefficients deterministically."""

    lines = [
        (
            "NormalEquationSystem "
            f"shape={system.shape} "
            f"source_jacobian_shape={system.source_jacobian_shape} "
            f"residual_dimension={system.residual_dimension} "
            f"regularization={system.regularization:g}"
        ),
        "variables:",
    ]
    lines.extend(
        f"  col={column} node={variable.node_id} name={variable.name}"
        for column, variable in enumerate(system.variables)
    )
    lines.append("gradient:")
    lines.extend(
        f"  col={column} value={value:g}"
        for column, value in enumerate(system.gradient)
    )
    lines.append("rhs:")
    lines.extend(f"  row={row} value={value:g}" for row, value in enumerate(system.rhs))
    lines.append("coefficients:")
    lines.extend(
        f"  row={row} values=[" + ", ".join(f"{value:g}" for value in row_values) + "]"
        for row, row_values in enumerate(system.coefficients)
    )
    lines.append(
        "linear_solve_residual="
        + (
            "<none>"
            if system.linear_solve_residual is None
            else "["
            + ", ".join(f"{value:g}" for value in system.linear_solve_residual)
            + "]"
        )
    )
    return "\n".join(lines)


def _validate_options(
    *,
    max_iterations: int,
    residual_tolerance: float,
    step_tolerance: float,
    gradient_tolerance: float,
    regularization: float,
    damping_steps: Sequence[float],
) -> _LeastSquaresOptions:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        msg = "max_iterations must be a positive integer"
        raise LeastSquaresError(msg)
    if max_iterations < 1:
        msg = "max_iterations must be a positive integer"
        raise LeastSquaresError(msg)
    if not _is_real_number(residual_tolerance):
        msg = "residual_tolerance must be a positive real number"
        raise LeastSquaresError(msg)
    if not _is_real_number(step_tolerance):
        msg = "step_tolerance must be a positive real number"
        raise LeastSquaresError(msg)
    if not _is_real_number(gradient_tolerance):
        msg = "gradient_tolerance must be a positive real number"
        raise LeastSquaresError(msg)
    if not _is_real_number(regularization):
        msg = "regularization must be a non-negative real number"
        raise LeastSquaresError(msg)
    try:
        damping_step_items = tuple(damping_steps)
    except TypeError as exc:
        msg = "damping_steps must be a sequence of numeric values"
        raise LeastSquaresError(msg) from exc
    if any(not _is_real_number(step) for step in damping_step_items):
        msg = "solver tolerances and damping_steps must be numeric"
        raise LeastSquaresError(msg)

    residual_tolerance_value = float(residual_tolerance)
    step_tolerance_value = float(step_tolerance)
    gradient_tolerance_value = float(gradient_tolerance)
    regularization_value = float(regularization)
    damping_step_values = tuple(float(step) for step in damping_step_items)

    if residual_tolerance_value <= 0.0:
        msg = "residual_tolerance must be positive"
        raise LeastSquaresError(msg)
    if step_tolerance_value <= 0.0:
        msg = "step_tolerance must be positive"
        raise LeastSquaresError(msg)
    if gradient_tolerance_value <= 0.0:
        msg = "gradient_tolerance must be positive"
        raise LeastSquaresError(msg)
    if regularization_value < 0.0:
        msg = "regularization must be non-negative"
        raise LeastSquaresError(msg)
    if not damping_step_values:
        msg = "damping_steps must contain at least one positive step"
        raise LeastSquaresError(msg)
    for step in damping_step_values:
        if step <= 0.0 or step > 1.0:
            msg = "damping_steps must be in the interval (0, 1]"
            raise LeastSquaresError(msg)
    return _LeastSquaresOptions(
        max_iterations=max_iterations,
        residual_tolerance=residual_tolerance_value,
        step_tolerance=step_tolerance_value,
        gradient_tolerance=gradient_tolerance_value,
        regularization=regularization_value,
        damping_steps=damping_step_values,
    )


def _build_contract(problem: Problem) -> AssemblyContract:
    try:
        return build_assembly_contract(problem)
    except ValueError as exc:
        raise LeastSquaresError(f"failed to build assembly contract: {exc}") from exc


def _initial_values(
    contract: AssemblyContract,
    values: Mapping[str, object],
) -> dict[str, object]:
    current_values = dict(values)
    missing = [
        variable.name
        for variable in contract.variables
        if variable.name not in current_values
    ]
    if missing:
        msg = "missing values for problem variables: " + ", ".join(missing)
        raise LeastSquaresError(msg)
    for variable in contract.variables:
        value = current_values[variable.name]
        if not _is_real_number(value):
            msg = "values must map problem variable names to numeric values"
            raise LeastSquaresError(msg)
        current_values[variable.name] = float(value)
    return current_values


def _evaluate_state(
    contract: AssemblyContract,
    values: dict[str, object],
    _iteration: int,
    options: _LeastSquaresOptions,
) -> _IterationState:
    try:
        numeric_values = _problem_numeric_values(contract, values)
        residual_assembly = assemble_residuals(contract, numeric_values)
        residual_values = tuple(value.value for value in residual_assembly.values)
        jacobian = assemble_jacobian(contract, numeric_values)
    except AssemblyError as exc:
        msg = f"failed to assemble least-squares state: {exc}"
        raise LeastSquaresError(msg) from exc

    normal_equations = _build_normal_equations(
        contract=contract,
        jacobian=jacobian,
        residual_values=residual_values,
        regularization=options.regularization,
    )
    return _IterationState(
        residual_values=residual_values,
        residual_norm=_norm(residual_values),
        least_squares_value=_least_squares_value(residual_values),
        tracked_objective_value=_objective_value(contract.problem, values),
        jacobian=jacobian,
        normal_equations=normal_equations,
        gradient_norm=_norm(normal_equations.gradient),
    )


def _build_normal_equations(
    *,
    contract: AssemblyContract,
    jacobian: SparseMatrixAssembly,
    residual_values: tuple[float, ...],
    regularization: float,
) -> NormalEquationSystem:
    dense_jacobian = to_dense(jacobian)
    rows, columns = jacobian.shape
    gradient = tuple(
        sum(dense_jacobian[row][column] * residual_values[row] for row in range(rows))
        for column in range(columns)
    )
    coefficients = tuple(
        tuple(
            sum(
                dense_jacobian[row][left] * dense_jacobian[row][right]
                for row in range(rows)
            )
            + (regularization if left == right else 0.0)
            for right in range(columns)
        )
        for left in range(columns)
    )
    return NormalEquationSystem(
        shape=(columns, columns),
        coefficients=coefficients,
        rhs=tuple(-value for value in gradient),
        regularization=regularization,
        source_jacobian_shape=jacobian.shape,
        residual_dimension=len(residual_values),
        variables=contract.variables,
        gradient=gradient,
    )


def _solve_normal_equations(system: NormalEquationSystem) -> _LinearStep:
    try:
        solution = tuple(_solve_dense_system(system.coefficients, system.rhs))
    except LeastSquaresError:
        raise
    except ValueError as exc:
        raise LeastSquaresError(f"failed to solve normal equations: {exc}") from exc
    residual = _linear_residual(system.coefficients, solution, system.rhs)
    normal_equations = NormalEquationSystem(
        shape=system.shape,
        coefficients=system.coefficients,
        rhs=system.rhs,
        regularization=system.regularization,
        source_jacobian_shape=system.source_jacobian_shape,
        residual_dimension=system.residual_dimension,
        variables=system.variables,
        gradient=system.gradient,
        linear_solve_residual=residual,
    )
    return _LinearStep(
        step=solution,
        residual=residual,
        residual_norm=_norm(residual),
        normal_equations=normal_equations,
    )


def _solve_dense_system(
    coefficients: tuple[tuple[float, ...], ...],
    rhs: tuple[float, ...],
) -> list[float]:
    size = len(rhs)
    if len(coefficients) != size or any(len(row) != size for row in coefficients):
        msg = "normal-equation coefficients must be square"
        raise LeastSquaresError(msg)
    matrix = [list(row) for row in coefficients]
    values = list(rhs)
    tolerance = 1e-12

    for pivot_column in range(size):
        pivot_row = max(
            range(pivot_column, size),
            key=lambda row: abs(matrix[row][pivot_column]),
        )
        pivot_value = matrix[pivot_row][pivot_column]
        if abs(pivot_value) <= tolerance:
            msg = f"singular normal-equation system at pivot column {pivot_column}"
            raise LeastSquaresError(msg)
        if pivot_row != pivot_column:
            matrix[pivot_column], matrix[pivot_row] = (
                matrix[pivot_row],
                matrix[pivot_column],
            )
            values[pivot_column], values[pivot_row] = (
                values[pivot_row],
                values[pivot_column],
            )

        pivot_value = matrix[pivot_column][pivot_column]
        for row in range(pivot_column + 1, size):
            factor = matrix[row][pivot_column] / pivot_value
            if factor == 0.0:
                continue
            for column in range(pivot_column, size):
                matrix[row][column] -= factor * matrix[pivot_column][column]
            values[row] -= factor * values[pivot_column]

    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        diagonal = matrix[row][row]
        if abs(diagonal) <= tolerance:
            msg = f"singular normal-equation system at back-substitution row {row}"
            raise LeastSquaresError(msg)
        trailing = sum(
            matrix[row][column] * solution[column] for column in range(row + 1, size)
        )
        solution[row] = (values[row] - trailing) / diagonal
    return solution


def _linear_residual(
    coefficients: tuple[tuple[float, ...], ...],
    solution: tuple[float, ...],
    rhs: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        sum(value * solution[column] for column, value in enumerate(row_values))
        - rhs[row]
        for row, row_values in enumerate(coefficients)
    )


def _accept_step(
    *,
    contract: AssemblyContract,
    values: dict[str, object],
    step: tuple[float, ...],
    current_value: float,
    damping_steps: Sequence[float],
) -> _AcceptedStep | None:
    for step_length in damping_steps:
        candidate_values = _apply_step(contract, values, step, step_length)
        residual_values = _residual_values(contract, candidate_values)
        candidate_value = _least_squares_value(residual_values)
        if candidate_value < current_value:
            return _AcceptedStep(
                values=candidate_values,
                residual_norm=_norm(residual_values),
                least_squares_value=candidate_value,
                step_length=float(step_length),
            )
    return None


def _apply_step(
    contract: AssemblyContract,
    values: dict[str, object],
    step: tuple[float, ...],
    step_length: float,
) -> dict[str, object]:
    updated = dict(values)
    for index, variable in enumerate(contract.variables):
        updated[variable.name] = _variable_float(values, variable.name) + (
            step_length * step[index]
        )
    return updated


def _residual_values(
    contract: AssemblyContract,
    values: dict[str, object],
) -> tuple[float, ...]:
    try:
        numeric_values = _problem_numeric_values(contract, values)
        return tuple(
            value.value for value in assemble_residuals(contract, numeric_values).values
        )
    except AssemblyError as exc:
        raise LeastSquaresError(f"failed to assemble residuals: {exc}") from exc


def _objective_value(problem: Problem, values: dict[str, object]) -> float | None:
    if problem.objective is None:
        return None
    try:
        return evaluate(
            problem.objective,
            _problem_numeric_values_for_variables(problem.variables, values),
        )
    except EvaluationError as exc:
        raise LeastSquaresError(f"failed to evaluate tracked objective: {exc}") from exc


def _record(
    *,
    contract: AssemblyContract,
    values: dict[str, object],
    iteration: int,
    previous_residual_norm: float | None,
    residual_norm: float,
    least_squares_value: float,
    gradient_norm: float,
    step_norm: float | None,
    accepted_step_length: float | None,
    linear_solve_residual_norm: float | None,
    tracked_objective_value: float | None,
    normal_equations: NormalEquationSystem | None,
) -> LeastSquaresIterationRecord:
    return LeastSquaresIterationRecord(
        iteration=iteration,
        previous_residual_norm=previous_residual_norm,
        residual_norm=float(residual_norm),
        least_squares_value=float(least_squares_value),
        gradient_norm=float(gradient_norm),
        step_norm=step_norm,
        accepted_step_length=accepted_step_length,
        linear_solve_residual_norm=linear_solve_residual_norm,
        tracked_objective_value=tracked_objective_value,
        regularization=normal_equations.regularization if normal_equations else 0.0,
        jacobian_shape=normal_equations.source_jacobian_shape
        if normal_equations
        else (0, 0),
        normal_equation_shape=normal_equations.shape if normal_equations else (0, 0),
        variables=tuple(
            VariableValue(
                name=variable.name,
                value=_variable_float(values, variable.name),
            )
            for variable in contract.variables
        ),
        normal_equations=normal_equations,
    )


def _problem_numeric_values(
    contract: AssemblyContract,
    values: dict[str, object],
) -> dict[str, float]:
    return _problem_numeric_values_for_variables(contract.variables, values)


def _problem_numeric_values_for_variables(
    variables: tuple[VariableRef, ...],
    values: dict[str, object],
) -> dict[str, float]:
    return {
        variable.name: _variable_float(values, variable.name) for variable in variables
    }


def _variable_float(values: dict[str, object], name: str) -> float:
    value = values[name]
    if not _is_real_number(value):
        msg = f"value for problem variable {name!r} must be numeric"
        raise LeastSquaresError(msg)
    return float(value)


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _norm(values: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _least_squares_value(values: Sequence[float]) -> float:
    return 0.5 * sum(value * value for value in values)


def _format_optional(value: float | None) -> str:
    if value is None:
        return "<none>"
    return f"{value:g}"
