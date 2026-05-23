"""Optional CasADi correctness bridge for supported tinyNLP expressions."""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any

from tinynlp.backends import evaluate as evaluate_tinynlp_expression
from tinynlp.ir import Expr, Node, NodeId, OpKind, VariableRef
from tinynlp.ir.analysis import (
    reachable_node_ids,
    require_non_empty_same_graph,
    variable_refs_for_expressions,
)
from tinynlp.nlp import (
    Problem,
    assemble_jacobian,
    assemble_residuals,
    build_assembly_contract,
)


class CasadiBridgeError(ValueError):
    """Raised when the optional CasADi bridge cannot compare a supported object."""


@dataclass(frozen=True)
class ValueComparison:
    """One scalar value comparison against CasADi."""

    label: str
    tiny_value: float
    casadi_value: float
    error: float


@dataclass(frozen=True)
class JacobianComparison:
    """One Jacobian coordinate comparison against CasADi."""

    row: int
    column: int
    variable: VariableRef
    tiny_value: float
    casadi_value: float
    error: float


@dataclass(frozen=True)
class ExpressionComparison:
    """Expression value comparison result."""

    name: str
    tolerance: float
    value: ValueComparison

    @property
    def max_error(self) -> float:
        """Maximum absolute error for this comparison."""

        return self.value.error

    @property
    def passed(self) -> bool:
        """Whether the comparison is within tolerance."""

        return self.max_error <= self.tolerance


@dataclass(frozen=True)
class ProblemAssemblyComparison:
    """Residual/Jacobian assembly comparison result."""

    problem_name: str
    tolerance: float
    residuals: tuple[ValueComparison, ...]
    jacobian_entries: tuple[JacobianComparison, ...]

    @property
    def max_residual_error(self) -> float:
        """Maximum residual comparison error."""

        return _max_error(self.residuals)

    @property
    def max_jacobian_error(self) -> float:
        """Maximum Jacobian comparison error."""

        return _max_error(self.jacobian_entries)

    @property
    def max_error(self) -> float:
        """Maximum absolute error across residuals and Jacobian entries."""

        return max(self.max_residual_error, self.max_jacobian_error)

    @property
    def passed(self) -> bool:
        """Whether every compared value is within tolerance."""

        return self.max_error <= self.tolerance


CasadiComparison = ExpressionComparison | ProblemAssemblyComparison


def casadi_available() -> bool:
    """Return whether CasADi can be imported."""

    return importlib.util.find_spec("casadi") is not None


def evaluate_expression(expr: Expr, values: Mapping[str, float]) -> float:
    """Evaluate a tinyNLP expression with CasADi."""

    return evaluate_expressions([expr], values)[0]


def evaluate_expressions(
    outputs: Sequence[Expr],
    values: Mapping[str, float],
) -> list[float]:
    """Evaluate tinyNLP expressions with CasADi."""

    compiled = _compile_outputs(outputs)
    return compiled.evaluate(values)


def evaluate_jacobian(
    outputs: Sequence[Expr],
    values: Mapping[str, float],
) -> list[list[float]]:
    """Evaluate a dense CasADi Jacobian for tinyNLP expressions."""

    compiled = _compile_outputs(outputs)
    return compiled.evaluate_jacobian(values)


def compare_expression(
    expr: Expr,
    values: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
) -> ExpressionComparison:
    """Compare one tinyNLP expression value against CasADi."""

    checked_tolerance = _validate_tolerance(tolerance)
    tiny_value = evaluate_tinynlp_expression(expr, values)
    casadi_value = evaluate_expression(expr, values)
    return ExpressionComparison(
        name=f"expression_node_{expr.id}",
        tolerance=checked_tolerance,
        value=ValueComparison(
            label=f"output_node={expr.id}",
            tiny_value=tiny_value,
            casadi_value=casadi_value,
            error=abs(tiny_value - casadi_value),
        ),
    )


def compare_problem_assembly(
    problem: Problem,
    values: Mapping[str, float],
    *,
    tolerance: float = 1e-9,
) -> ProblemAssemblyComparison:
    """Compare tinyNLP residual/Jacobian assembly against CasADi."""

    checked_tolerance = _validate_tolerance(tolerance)
    contract = build_assembly_contract(problem)
    tiny_residuals = assemble_residuals(contract, values)
    tiny_jacobian = assemble_jacobian(contract, values)
    compiled = _compile_outputs(problem.residuals, variables=problem.variables)
    casadi_residuals = compiled.evaluate(values)
    casadi_jacobian = compiled.evaluate_jacobian(values)

    residuals = tuple(
        ValueComparison(
            label=f"residual_row={value.row}",
            tiny_value=value.value,
            casadi_value=casadi_residuals[value.row],
            error=abs(value.value - casadi_residuals[value.row]),
        )
        for value in tiny_residuals.values
    )
    jacobian_entries = tuple(
        JacobianComparison(
            row=entry.row,
            column=entry.column,
            variable=contract.variables[entry.column],
            tiny_value=entry.value,
            casadi_value=casadi_jacobian[entry.row][entry.column],
            error=abs(entry.value - casadi_jacobian[entry.row][entry.column]),
        )
        for entry in tiny_jacobian.entries
    )
    return ProblemAssemblyComparison(
        problem_name=problem.name,
        tolerance=checked_tolerance,
        residuals=residuals,
        jacobian_entries=jacobian_entries,
    )


def format_casadi_comparison(result: CasadiComparison) -> str:
    """Format a CasADi comparison result deterministically."""

    if isinstance(result, ExpressionComparison):
        return "\n".join(
            [
                (
                    "CasadiExpressionComparison "
                    f"name={result.name} "
                    f"passed={result.passed} "
                    f"tolerance={result.tolerance:g} "
                    f"max_error={result.max_error:g}"
                ),
                (
                    "  value "
                    f"label={result.value.label} "
                    f"tiny={result.value.tiny_value:g} "
                    f"casadi={result.value.casadi_value:g} "
                    f"error={result.value.error:g}"
                ),
            ]
        )

    lines = [
        (
            "CasadiProblemAssemblyComparison "
            f"problem={result.problem_name} "
            f"passed={result.passed} "
            f"tolerance={result.tolerance:g} "
            f"max_error={result.max_error:g} "
            f"residual_max_error={result.max_residual_error:g} "
            f"jacobian_max_error={result.max_jacobian_error:g}"
        ),
        "residuals:",
    ]
    lines.extend(
        "  "
        f"label={entry.label} "
        f"tiny={entry.tiny_value:g} "
        f"casadi={entry.casadi_value:g} "
        f"error={entry.error:g}"
        for entry in result.residuals
    )
    lines.append("jacobian_entries:")
    lines.extend(
        "  "
        f"row={entry.row} col={entry.column} "
        f"variable={entry.variable.name} "
        f"tiny={entry.tiny_value:g} "
        f"casadi={entry.casadi_value:g} "
        f"error={entry.error:g}"
        for entry in result.jacobian_entries
    )
    return "\n".join(lines)


@dataclass(frozen=True)
class _CompiledOutputs:
    variable_names: tuple[str, ...]
    value_function: Any
    jacobian_function: Any

    def evaluate(self, values: Mapping[str, float]) -> list[float]:
        result = _call_function(self.value_function, self.variable_names, values)
        return _dm_vector(result)

    def evaluate_jacobian(self, values: Mapping[str, float]) -> list[list[float]]:
        result = _call_function(self.jacobian_function, self.variable_names, values)
        return _dm_matrix(result)


def _compile_outputs(
    outputs: Sequence[Expr],
    *,
    variables: tuple[VariableRef, ...] | None = None,
) -> _CompiledOutputs:
    ca = _casadi()
    output_tuple = tuple(outputs)
    graph = require_non_empty_same_graph(list(output_tuple))
    variable_refs = variables or variable_refs_for_expressions(output_tuple)
    _ensure_unique_variables(variable_refs)
    symbols_by_node = {
        variable.node_id: ca.MX.sym(variable.name) for variable in variable_refs
    }
    reachable = set().union(*(reachable_node_ids(output) for output in output_tuple))
    if variables is not None:
        reachable.update(variable.node_id for variable in variables)
    expression_by_node: dict[NodeId, Any] = {}
    for node in graph.nodes:
        if node.id not in reachable:
            continue
        expression_by_node[node.id] = _convert_node(
            ca,
            node,
            expression_by_node,
            symbols_by_node,
        )

    casadi_outputs = [expression_by_node[output.id] for output in output_tuple]
    output_vector = ca.vertcat(*casadi_outputs)
    symbol_vector = ca.vertcat(
        *(symbols_by_node[variable.node_id] for variable in variable_refs)
    )
    value_function = ca.Function("tiny_value", [symbol_vector], [output_vector])
    jacobian_function = ca.Function(
        "tiny_jacobian",
        [symbol_vector],
        [ca.jacobian(output_vector, symbol_vector)],
    )
    return _CompiledOutputs(
        variable_names=tuple(variable.name for variable in variable_refs),
        value_function=value_function,
        jacobian_function=jacobian_function,
    )


def _convert_node(
    ca: Any,
    node: Node,
    expression_by_node: Mapping[NodeId, Any],
    symbols_by_node: Mapping[NodeId, Any],
) -> Any:
    if node.op is OpKind.VARIABLE:
        try:
            return symbols_by_node[node.id]
        except KeyError as exc:
            msg = f"variable node {node.id} is not available to CasADi bridge"
            raise CasadiBridgeError(msg) from exc
    if node.op is OpKind.CONSTANT:
        if node.value is None:
            msg = f"constant node {node.id} is missing a value"
            raise CasadiBridgeError(msg)
        return ca.MX(node.value)
    if node.op is OpKind.NEG:
        return -expression_by_node[node.inputs[0]]

    left = expression_by_node[node.inputs[0]]
    right = expression_by_node[node.inputs[1]]
    if node.op is OpKind.ADD:
        return left + right
    if node.op is OpKind.SUB:
        return left - right
    if node.op is OpKind.MUL:
        return left * right
    if node.op is OpKind.DIV:
        return left / right

    msg = f"unsupported operation for CasADi bridge: {node.op}"
    raise CasadiBridgeError(msg)


def _casadi() -> Any:
    try:
        return importlib.import_module("casadi")
    except ImportError as exc:
        msg = "CasADi is not installed; install tinynlp-opt[casadi] to use this bridge"
        raise CasadiBridgeError(msg) from exc


def _call_function(function: Any, names: tuple[str, ...], values: Mapping[str, float]):
    numeric_values = [_numeric_value(name, values) for name in names]
    result = function(numeric_values)
    if isinstance(result, list | tuple):
        return result[0]
    return result


def _numeric_value(name: str, values: Mapping[str, float]) -> float:
    if name not in values:
        msg = f"missing value for variable {name!r}"
        raise CasadiBridgeError(msg)
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"value for variable {name!r} must be a real number"
        raise CasadiBridgeError(msg)
    return float(value)


def _dm_vector(value: Any) -> list[float]:
    return [float(value[index]) for index in range(value.numel())]


def _dm_matrix(value: Any) -> list[list[float]]:
    return [
        [float(value[row, column]) for column in range(value.size2())]
        for row in range(value.size1())
    ]


def _ensure_unique_variables(variables: tuple[VariableRef, ...]) -> None:
    seen: set[str] = set()
    for variable in variables:
        if variable.name in seen:
            msg = (
                "CasADi bridge requires unique variable names; "
                f"{variable.name!r} appears more than once"
            )
            raise CasadiBridgeError(msg)
        seen.add(variable.name)


def _validate_tolerance(tolerance: float) -> float:
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real):
        msg = "tolerance must be a positive real number"
        raise CasadiBridgeError(msg)
    checked = float(tolerance)
    if checked <= 0.0:
        msg = "tolerance must be positive"
        raise CasadiBridgeError(msg)
    return checked


def _max_error(entries: Sequence[ValueComparison | JacobianComparison]) -> float:
    if not entries:
        return 0.0
    return max(entry.error for entry in entries)
