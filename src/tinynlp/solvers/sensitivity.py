"""Reference implicit sensitivity workflow for supported constrained problems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import sqrt
from numbers import Real

from tinynlp.ir import VariableRef
from tinynlp.nlp import (
    AssemblyContract,
    AssemblyError,
    AssemblyProvenance,
    CoordinateEntry,
    Problem,
    SparseMatrixAssembly,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
)
from tinynlp.solvers.constrained import SolverResult
from tinynlp.solvers.kkt import KKTError, KKTSystem, build_kkt_system
from tinynlp.solvers.linear import (
    DenseReferenceLinearSolver,
    LinearSolveError,
    LinearSolver,
    LinearSolveResult,
)


class SensitivityError(ValueError):
    """Raised when the reference sensitivity workflow cannot run."""


@dataclass(frozen=True)
class SensitivityEntry:
    """One scalar sensitivity for a solve variable."""

    variable: VariableRef
    value: float


@dataclass(frozen=True)
class SensitivityRhsEntry:
    """One residual-row contribution to the sensitivity right-hand side."""

    row: int
    derivative: float
    rhs_value: float
    provenance: AssemblyProvenance | None


@dataclass(frozen=True)
class SensitivityTrace:
    """Inspectable metadata for one implicit sensitivity calculation."""

    parameter: VariableRef
    solve_variables: tuple[VariableRef, ...]
    residual_norm: float
    residual_values: tuple[float, ...]
    kkt_system: KKTSystem
    rhs: tuple[float, ...]
    rhs_entries: tuple[SensitivityRhsEntry, ...]
    kkt_solve_residual_norm: float


@dataclass(frozen=True)
class SensitivityResult:
    """Result from the scalar-parameter implicit sensitivity prototype."""

    parameter: VariableRef
    entries: tuple[SensitivityEntry, ...]
    trace: SensitivityTrace

    @property
    def sensitivities(self) -> dict[str, float]:
        """Return sensitivities keyed by solve-variable name."""

        return {entry.variable.name: entry.value for entry in self.entries}


def implicit_sensitivity(
    problem: Problem,
    solution: SolverResult | Mapping[str, object],
    *,
    parameter: str,
    solve_variables: Sequence[str] | None = None,
    residual_tolerance: float = 1e-8,
    linear_solver: LinearSolver | None = None,
) -> SensitivityResult:
    """Compute first-order sensitivities for one scalar problem parameter.

    The reference equation is:

    ``[I J_z^T; J_z 0] [dz/dp; lambda] = [0; -J_p]``

    where ``J_z`` contains columns for the selected solve variables and ``J_p``
    is the residual derivative column for the scalar parameter. The primal block
    is the same identity/reference block used by the M12 residual-correction
    prototype; this is not a Hessian-backed sensitivity method.
    """

    tolerance = _validate_residual_tolerance(residual_tolerance)
    values = _solution_values(solution)
    contract = _build_contract(problem)
    variable_by_name = _variables_by_name(problem.variables)
    parameter_ref = _parameter_ref(parameter, variable_by_name)
    solve_refs = _solve_variable_refs(
        solve_variables=solve_variables,
        parameter=parameter_ref,
        problem_variables=problem.variables,
        variable_by_name=variable_by_name,
    )
    numeric_values = _numeric_values(problem.variables, values)
    residual_values = _residual_values(contract, numeric_values)
    residual_norm = _norm(residual_values)
    if residual_norm > tolerance:
        msg = (
            "solution residual norm exceeds tolerance; "
            f"residual_norm={residual_norm:g} tolerance={tolerance:g}"
        )
        raise SensitivityError(msg)

    jacobian = _assemble_jacobian(contract, numeric_values)
    reduced_jacobian, rhs_entries = _partition_jacobian(
        jacobian=jacobian,
        problem_variables=problem.variables,
        solve_variables=solve_refs,
        parameter=parameter_ref,
    )
    kkt_system = _build_kkt_system(reduced_jacobian)
    rhs = (0.0,) * len(solve_refs) + tuple(entry.rhs_value for entry in rhs_entries)
    solve_result = _solve_sensitivity_system(kkt_system, rhs, linear_solver)
    entries = tuple(
        SensitivityEntry(variable=variable, value=float(solve_result.solution[index]))
        for index, variable in enumerate(solve_refs)
    )
    trace = SensitivityTrace(
        parameter=parameter_ref,
        solve_variables=solve_refs,
        residual_norm=residual_norm,
        residual_values=residual_values,
        kkt_system=kkt_system,
        rhs=tuple(rhs),
        rhs_entries=rhs_entries,
        kkt_solve_residual_norm=_norm(solve_result.residual),
    )
    return SensitivityResult(
        parameter=parameter_ref,
        entries=entries,
        trace=trace,
    )


def format_sensitivity(result: SensitivityResult) -> str:
    """Format a sensitivity result deterministically."""

    trace = result.trace
    lines = [
        (
            "SensitivityResult "
            f"parameter={_format_variable(result.parameter)} "
            f"residual_norm={trace.residual_norm:g} "
            f"kkt_shape={trace.kkt_system.shape} "
            f"kkt_solve_residual_norm={trace.kkt_solve_residual_norm:g}"
        ),
        "solve_variables:",
    ]
    lines.extend(
        f"  variable={_format_variable(entry.variable)} sensitivity={entry.value:g}"
        for entry in result.entries
    )
    lines.append("rhs_entries:")
    lines.extend(_format_rhs_entry(entry) for entry in trace.rhs_entries)
    lines.append("rhs=[" + ", ".join(f"{value:g}" for value in trace.rhs) + "]")
    return "\n".join(lines)


def _validate_residual_tolerance(residual_tolerance: float) -> float:
    if not _is_real_number(residual_tolerance):
        msg = "residual_tolerance must be a positive real number"
        raise SensitivityError(msg)
    tolerance = float(residual_tolerance)
    if tolerance <= 0.0:
        msg = "residual_tolerance must be positive"
        raise SensitivityError(msg)
    return tolerance


def _solution_values(
    solution: SolverResult | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(solution, SolverResult):
        if not solution.success:
            msg = "implicit sensitivity requires a successful SolverResult"
            raise SensitivityError(msg)
        return dict(solution.final_values)
    if isinstance(solution, Mapping):
        return dict(solution)
    msg = "solution must be a SolverResult or a value mapping"
    raise SensitivityError(msg)


def _build_contract(problem: Problem) -> AssemblyContract:
    try:
        return build_assembly_contract(problem)
    except ValueError as exc:
        raise SensitivityError(f"failed to build assembly contract: {exc}") from exc


def _variables_by_name(
    variables: tuple[VariableRef, ...],
) -> dict[str, VariableRef]:
    return {variable.name: variable for variable in variables}


def _parameter_ref(
    parameter: str,
    variable_by_name: Mapping[str, VariableRef],
) -> VariableRef:
    if not isinstance(parameter, str) or not parameter:
        msg = "parameter must be a non-empty variable name"
        raise SensitivityError(msg)
    if parameter not in variable_by_name:
        msg = f"unknown sensitivity parameter {parameter!r}"
        raise SensitivityError(msg)
    return variable_by_name[parameter]


def _solve_variable_refs(
    *,
    solve_variables: Sequence[str] | None,
    parameter: VariableRef,
    problem_variables: tuple[VariableRef, ...],
    variable_by_name: Mapping[str, VariableRef],
) -> tuple[VariableRef, ...]:
    if solve_variables is None:
        refs = tuple(
            variable
            for variable in problem_variables
            if variable.name != parameter.name
        )
        if not refs:
            msg = "solve_variables must contain at least one variable"
            raise SensitivityError(msg)
        return refs
    if isinstance(solve_variables, str):
        msg = "solve_variables must be a sequence of variable names"
        raise SensitivityError(msg)

    names = tuple(solve_variables)
    if not names:
        msg = "solve_variables must contain at least one variable"
        raise SensitivityError(msg)
    duplicates = _duplicates(names)
    if duplicates:
        msg = "solve_variables contains duplicate names: " + ", ".join(duplicates)
        raise SensitivityError(msg)
    if parameter.name in names:
        msg = "solve_variables must not include the sensitivity parameter"
        raise SensitivityError(msg)

    unknown = [name for name in names if name not in variable_by_name]
    if unknown:
        msg = "unknown solve_variables: " + ", ".join(repr(name) for name in unknown)
        raise SensitivityError(msg)
    return tuple(variable_by_name[name] for name in names)


def _duplicates(names: Iterable[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    return tuple(duplicates)


def _numeric_values(
    variables: tuple[VariableRef, ...],
    values: Mapping[str, object],
) -> dict[str, float]:
    numeric_values: dict[str, float] = {}
    for variable in variables:
        if variable.name not in values:
            msg = f"missing value for problem variable {variable.name!r}"
            raise SensitivityError(msg)
        value = values[variable.name]
        if not _is_real_number(value):
            msg = f"value for problem variable {variable.name!r} must be numeric"
            raise SensitivityError(msg)
        numeric_values[variable.name] = float(value)
    return numeric_values


def _residual_values(
    contract: AssemblyContract,
    values: Mapping[str, float],
) -> tuple[float, ...]:
    try:
        residuals = assemble_residuals(contract, values)
    except AssemblyError as exc:
        raise SensitivityError(f"failed to assemble residuals: {exc}") from exc
    return tuple(value.value for value in residuals.values)


def _assemble_jacobian(
    contract: AssemblyContract,
    values: Mapping[str, float],
) -> SparseMatrixAssembly:
    try:
        return assemble_jacobian(contract, values)
    except AssemblyError as exc:
        raise SensitivityError(f"failed to assemble Jacobian: {exc}") from exc


def _partition_jacobian(
    *,
    jacobian: SparseMatrixAssembly,
    problem_variables: tuple[VariableRef, ...],
    solve_variables: tuple[VariableRef, ...],
    parameter: VariableRef,
) -> tuple[SparseMatrixAssembly, tuple[SensitivityRhsEntry, ...]]:
    full_column_by_node = {
        variable.node_id: column for column, variable in enumerate(problem_variables)
    }
    solve_column_by_full_column = {
        full_column_by_node[variable.node_id]: column
        for column, variable in enumerate(solve_variables)
    }
    parameter_column = full_column_by_node[parameter.node_id]
    reduced_entries: list[CoordinateEntry] = []
    parameter_entries: list[CoordinateEntry | None] = [None] * jacobian.shape[0]

    for entry in jacobian.entries:
        if entry.column in solve_column_by_full_column:
            reduced_entries.append(
                CoordinateEntry(
                    row=entry.row,
                    column=solve_column_by_full_column[entry.column],
                    value=entry.value,
                    provenance=entry.provenance,
                )
            )
        if entry.column == parameter_column:
            if parameter_entries[entry.row] is not None:
                msg = (
                    f"Jacobian contains duplicate parameter derivative row {entry.row}"
                )
                raise SensitivityError(msg)
            parameter_entries[entry.row] = entry

    rhs_entries = tuple(
        _rhs_entry(row=row, entry=entry) for row, entry in enumerate(parameter_entries)
    )
    return (
        SparseMatrixAssembly(
            shape=(jacobian.shape[0], len(solve_variables)),
            entries=tuple(
                sorted(reduced_entries, key=lambda entry: (entry.row, entry.column))
            ),
        ),
        rhs_entries,
    )


def _rhs_entry(
    *,
    row: int,
    entry: CoordinateEntry | None,
) -> SensitivityRhsEntry:
    derivative = 0.0 if entry is None else float(entry.value)
    return SensitivityRhsEntry(
        row=row,
        derivative=derivative,
        rhs_value=-derivative,
        provenance=None if entry is None else entry.provenance,
    )


def _build_kkt_system(jacobian: SparseMatrixAssembly) -> KKTSystem:
    try:
        return build_kkt_system(jacobian)
    except KKTError as exc:
        msg = f"failed to build sensitivity KKT system: {exc}"
        raise SensitivityError(msg) from exc


def _solve_sensitivity_system(
    system: KKTSystem,
    rhs: Sequence[float],
    linear_solver: LinearSolver | None,
) -> LinearSolveResult:
    solver = linear_solver or DenseReferenceLinearSolver()
    try:
        return solver.solve(system, rhs)
    except LinearSolveError as exc:
        msg = f"failed to solve sensitivity KKT system: {exc}"
        raise SensitivityError(msg) from exc


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _norm(values: Sequence[float]) -> float:
    return sqrt(sum(value * value for value in values))


def _format_rhs_entry(entry: SensitivityRhsEntry) -> str:
    provenance = (
        "structural-zero"
        if entry.provenance is None
        else _format_assembly_provenance(entry.provenance)
    )
    return (
        "  "
        f"row={entry.row} "
        f"derivative={entry.derivative:g} "
        f"rhs={entry.rhs_value:g} "
        f"provenance=[{provenance}]"
    )


def _format_assembly_provenance(provenance: AssemblyProvenance) -> str:
    parts = [
        f"kind={provenance.kind}",
        f"row={provenance.row}",
        f"source={provenance.source_node_id}",
    ]
    if provenance.column is not None:
        parts.append(f"col={provenance.column}")
    if provenance.variable is not None:
        parts.append(f"variable={_format_variable(provenance.variable)}")
    if provenance.derivative_node_id is not None:
        parts.append(f"derivative={provenance.derivative_node_id}")
    return " ".join(parts)


def _format_variable(variable: VariableRef) -> str:
    return f"{variable.name}@{variable.node_id}"
