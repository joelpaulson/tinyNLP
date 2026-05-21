"""Assembly contracts for residual and Jacobian expression evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from tinynlp.autodiff import DerivativeTraceEvent, jacobian
from tinynlp.backends import EvaluationError, KernelPlan, build_kernel_plan, get_backend
from tinynlp.ir import Expr, NodeId, VariableRef
from tinynlp.nlp.problem import Problem
from tinynlp.nlp.sparsity import SparsityEntry, SparsityPattern, jacobian_sparsity


class AssemblyError(ValueError):
    """Raised when a contract term cannot be assembled numerically."""


@dataclass(frozen=True)
class ResidualAssemblyTerm:
    """One residual row ready for numeric assembly."""

    row: int
    expr: Expr
    kernel_plan: KernelPlan


@dataclass(frozen=True)
class JacobianAssemblyTerm:
    """One structurally present Jacobian coordinate ready for assembly."""

    row: int
    column: int
    variable: VariableRef
    derivative: Expr
    kernel_plan: KernelPlan
    source_residual: Expr
    derivative_trace: tuple[DerivativeTraceEvent, ...]


@dataclass(frozen=True)
class AssemblyContract:
    """Deterministic symbolic contract for residual/Jacobian assembly."""

    problem: Problem
    variables: tuple[VariableRef, ...]
    residual_terms: tuple[ResidualAssemblyTerm, ...]
    jacobian_terms: tuple[JacobianAssemblyTerm, ...]
    sparsity: SparsityPattern


@dataclass(frozen=True)
class AssemblyProvenance:
    """Stable provenance for one assembled value."""

    kind: str
    row: int
    column: int | None
    source_node_id: NodeId
    derivative_node_id: NodeId | None = None
    variable: VariableRef | None = None


@dataclass(frozen=True)
class ResidualValue:
    """One assembled residual value."""

    row: int
    value: float
    provenance: AssemblyProvenance


@dataclass(frozen=True)
class ResidualAssembly:
    """Assembled residual vector with row provenance."""

    values: tuple[ResidualValue, ...]


@dataclass(frozen=True)
class CoordinateEntry:
    """One sparse coordinate value."""

    row: int
    column: int
    value: float
    provenance: AssemblyProvenance


@dataclass(frozen=True)
class SparseMatrixAssembly:
    """Dependency-free coordinate sparse matrix assembly."""

    shape: tuple[int, int]
    entries: tuple[CoordinateEntry, ...]


def build_assembly_contract(problem: Problem) -> AssemblyContract:
    """Build a symbolic assembly contract without binding numeric values."""

    residuals = problem.residuals
    jacobian_result = jacobian(list(residuals))
    source_sparsity = jacobian_sparsity(list(residuals))
    sparsity = _remap_sparsity_to_problem_variables(
        source_sparsity,
        problem.variables,
    )
    derivative_by_coordinate = {
        (row, entry.variable.node_id): entry.derivative
        for row, row_entries in enumerate(jacobian_result.rows)
        for entry in row_entries
    }
    trace_by_row = {row: trace for row, trace in enumerate(jacobian_result.traces)}

    residual_terms = tuple(
        ResidualAssemblyTerm(
            row=row,
            expr=expr,
            kernel_plan=build_kernel_plan(expr),
        )
        for row, expr in enumerate(residuals)
    )
    jacobian_terms = tuple(
        JacobianAssemblyTerm(
            row=entry.row,
            column=entry.column,
            variable=entry.variable,
            derivative=derivative_by_coordinate[(entry.row, entry.variable.node_id)],
            kernel_plan=build_kernel_plan(
                derivative_by_coordinate[(entry.row, entry.variable.node_id)]
            ),
            source_residual=residuals[entry.row],
            derivative_trace=trace_by_row[entry.row],
        )
        for entry in sparsity.entries
    )

    return AssemblyContract(
        problem=problem,
        variables=problem.variables,
        residual_terms=residual_terms,
        jacobian_terms=jacobian_terms,
        sparsity=sparsity,
    )


def assemble_residuals(
    contract: AssemblyContract,
    values: Mapping[str, float],
) -> ResidualAssembly:
    """Assemble residual values through the registered Python backend."""

    backend = get_backend("python")
    assembled: list[ResidualValue] = []
    for term in contract.residual_terms:
        provenance = _residual_provenance(term)
        try:
            value = backend.execute(term.kernel_plan, values)
        except EvaluationError as exc:
            raise _assembly_error(provenance, exc) from exc
        assembled.append(
            ResidualValue(
                row=term.row,
                value=value,
                provenance=provenance,
            )
        )
    return ResidualAssembly(values=tuple(assembled))


def assemble_jacobian(
    contract: AssemblyContract,
    values: Mapping[str, float],
) -> SparseMatrixAssembly:
    """Assemble structurally present Jacobian coordinates."""

    backend = get_backend("python")
    entries: list[CoordinateEntry] = []
    for term in contract.jacobian_terms:
        provenance = _jacobian_provenance(term)
        try:
            value = backend.execute(term.kernel_plan, values)
        except EvaluationError as exc:
            raise _assembly_error(provenance, exc) from exc
        entries.append(
            CoordinateEntry(
                row=term.row,
                column=term.column,
                value=value,
                provenance=provenance,
            )
        )
    return SparseMatrixAssembly(
        shape=contract.sparsity.shape,
        entries=tuple(entries),
    )


def to_dense(matrix: SparseMatrixAssembly) -> list[list[float]]:
    """Convert a coordinate assembly to a dense list for reference checks."""

    rows, columns = matrix.shape
    dense = [[0.0 for _ in range(columns)] for _ in range(rows)]
    for entry in matrix.entries:
        dense[entry.row][entry.column] = entry.value
    return dense


def format_residual_assembly(assembly: ResidualAssembly) -> str:
    """Format residual values and provenance deterministically."""

    lines = ["ResidualAssembly"]
    lines.extend(
        f"  row={value.row} value={value.value:g} "
        f"provenance=[{_format_provenance(value.provenance)}]"
        for value in assembly.values
    )
    return "\n".join(lines)


def format_sparse_matrix(matrix: SparseMatrixAssembly) -> str:
    """Format sparse coordinate entries and provenance deterministically."""

    lines = [f"SparseMatrixAssembly shape={matrix.shape}"]
    lines.extend(
        f"  row={entry.row} col={entry.column} value={entry.value:g} "
        f"provenance=[{_format_provenance(entry.provenance)}]"
        for entry in matrix.entries
    )
    return "\n".join(lines)


def format_assembly_contract(contract: AssemblyContract) -> str:
    """Format an assembly contract with stable provenance text."""

    lines = [
        f"AssemblyContract problem={contract.problem.name}",
        (
            "dimensions: "
            f"variables={len(contract.variables)} "
            f"residuals={len(contract.residual_terms)} "
            f"jacobian_terms={len(contract.jacobian_terms)}"
        ),
        "variables:",
    ]
    lines.extend(
        f"  col={column} node={variable.node_id} name={variable.name}"
        for column, variable in enumerate(contract.variables)
    )
    lines.append("residual_terms:")
    lines.extend(
        f"  row={term.row} expr={term.expr.id} plan_output={term.kernel_plan.output}"
        for term in contract.residual_terms
    )
    lines.append("jacobian_terms:")
    lines.extend(_format_jacobian_term(term) for term in contract.jacobian_terms)
    return "\n".join(lines)


def _remap_sparsity_to_problem_variables(
    sparsity: SparsityPattern,
    variables: tuple[VariableRef, ...],
) -> SparsityPattern:
    column_by_node = {
        variable.node_id: column for column, variable in enumerate(variables)
    }
    entries = tuple(
        SparsityEntry(
            row=entry.row,
            column=column_by_node[entry.variable.node_id],
            variable=entry.variable,
        )
        for entry in sparsity.entries
    )
    return SparsityPattern(
        shape=(sparsity.shape[0], len(variables)),
        variables=variables,
        entries=entries,
        trace=sparsity.trace,
    )


def _format_jacobian_term(term: JacobianAssemblyTerm) -> str:
    return (
        "  "
        f"row={term.row} col={term.column} "
        f"variable={term.variable.name}@{term.variable.node_id} "
        f"source={term.source_residual.id} "
        f"derivative={term.derivative.id} "
        f"plan_output={term.kernel_plan.output} "
        f"trace_events={len(term.derivative_trace)}"
    )


def _residual_provenance(term: ResidualAssemblyTerm) -> AssemblyProvenance:
    return AssemblyProvenance(
        kind="residual",
        row=term.row,
        column=None,
        source_node_id=term.expr.id,
    )


def _jacobian_provenance(term: JacobianAssemblyTerm) -> AssemblyProvenance:
    return AssemblyProvenance(
        kind="jacobian",
        row=term.row,
        column=term.column,
        source_node_id=term.source_residual.id,
        derivative_node_id=term.derivative.id,
        variable=term.variable,
    )


def _assembly_error(
    provenance: AssemblyProvenance,
    error: EvaluationError,
) -> AssemblyError:
    return AssemblyError(
        f"failed to assemble {_format_error_context(provenance)}: {error}"
    )


def _format_error_context(provenance: AssemblyProvenance) -> str:
    if provenance.column is None:
        return f"residual row {provenance.row} source node {provenance.source_node_id}"
    variable = _format_variable(provenance.variable)
    return (
        f"jacobian row {provenance.row} column {provenance.column} "
        f"variable {variable} source node {provenance.source_node_id} "
        f"derivative node {provenance.derivative_node_id}"
    )


def _format_provenance(provenance: AssemblyProvenance) -> str:
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


def _format_variable(variable: VariableRef | None) -> str:
    if variable is None:
        return "<none>"
    return f"{variable.name}@{variable.node_id}"
