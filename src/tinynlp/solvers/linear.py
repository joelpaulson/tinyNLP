"""Reference linear-solve interface for explicit KKT systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from tinynlp.solvers.kkt import KKTSystem, kkt_to_dense


class LinearSolveError(ValueError):
    """Raised when the reference linear solver cannot solve a system."""


class LinearSolver(Protocol):
    """Minimal linear-solver protocol."""

    name: str

    def solve(
        self,
        system: KKTSystem,
        rhs: Sequence[float],
    ) -> LinearSolveResult:
        """Solve a KKT system for the supplied right-hand side."""


@dataclass(frozen=True)
class LinearSolveResult:
    """Result from a linear solve."""

    solver_name: str
    solution: tuple[float, ...]
    residual: tuple[float, ...]


class DenseReferenceLinearSolver:
    """Pure-Python dense reference solver for tiny deterministic systems."""

    name = "dense-reference"

    def solve(
        self,
        system: KKTSystem,
        rhs: Sequence[float],
    ) -> LinearSolveResult:
        """Solve a KKT system with Gaussian elimination and partial pivoting."""

        matrix = kkt_to_dense(system)
        rhs_tuple = _validate_rhs(system, rhs)
        solution = _gaussian_elimination(matrix, rhs_tuple)
        residual = _residual(matrix, solution, rhs_tuple)
        return LinearSolveResult(
            solver_name=self.name,
            solution=tuple(solution),
            residual=tuple(residual),
        )


def _validate_rhs(system: KKTSystem, rhs: Sequence[float]) -> tuple[float, ...]:
    rhs_tuple = tuple(float(value) for value in rhs)
    if len(rhs_tuple) != system.shape[0]:
        msg = (
            "right-hand side length must match KKT dimension; "
            f"got {len(rhs_tuple)} for shape {system.shape}"
        )
        raise LinearSolveError(msg)
    return rhs_tuple


def _gaussian_elimination(
    matrix: list[list[float]],
    rhs: tuple[float, ...],
) -> list[float]:
    size = len(rhs)
    coefficients = [row[:] for row in matrix]
    values = list(rhs)
    tolerance = 1e-12

    for pivot_column in range(size):
        pivot_row = max(
            range(pivot_column, size),
            key=lambda row: abs(coefficients[row][pivot_column]),
        )
        pivot_value = coefficients[pivot_row][pivot_column]
        if abs(pivot_value) <= tolerance:
            msg = f"singular KKT system at pivot column {pivot_column}"
            raise LinearSolveError(msg)
        if pivot_row != pivot_column:
            coefficients[pivot_column], coefficients[pivot_row] = (
                coefficients[pivot_row],
                coefficients[pivot_column],
            )
            values[pivot_column], values[pivot_row] = (
                values[pivot_row],
                values[pivot_column],
            )

        pivot_value = coefficients[pivot_column][pivot_column]
        for row in range(pivot_column + 1, size):
            factor = coefficients[row][pivot_column] / pivot_value
            if factor == 0.0:
                continue
            for column in range(pivot_column, size):
                coefficients[row][column] -= factor * coefficients[pivot_column][column]
            values[row] -= factor * values[pivot_column]

    solution = [0.0 for _ in range(size)]
    for row in range(size - 1, -1, -1):
        diagonal = coefficients[row][row]
        if abs(diagonal) <= tolerance:
            msg = f"singular KKT system at back-substitution row {row}"
            raise LinearSolveError(msg)
        trailing = sum(
            coefficients[row][column] * solution[column]
            for column in range(row + 1, size)
        )
        solution[row] = (values[row] - trailing) / diagonal
    return solution


def _residual(
    matrix: list[list[float]],
    solution: list[float],
    rhs: tuple[float, ...],
) -> list[float]:
    return [
        sum(value * solution[column] for column, value in enumerate(row_values))
        - rhs[row]
        for row, row_values in enumerate(matrix)
    ]
