"""Assembly contracts for residual and Jacobian expression evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from tinynlp.autodiff import DerivativeTraceEvent, jacobian
from tinynlp.backends import KernelPlan, build_kernel_plan
from tinynlp.ir import Expr, VariableRef
from tinynlp.nlp.problem import Problem
from tinynlp.nlp.sparsity import SparsityEntry, SparsityPattern, jacobian_sparsity


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
