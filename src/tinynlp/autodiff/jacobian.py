"""Vector-output Jacobians and derivative verification."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tinynlp.autodiff.reverse import (
    DerivativeTraceEvent,
    GradientEntry,
    gradient,
)
from tinynlp.backends import evaluate
from tinynlp.ir import Expr, VariableRef
from tinynlp.ir.analysis import (
    require_non_empty_same_graph,
    variable_refs_for_expressions,
)


@dataclass(frozen=True)
class Jacobian:
    """Dense symbolic Jacobian for a sequence of scalar outputs."""

    outputs: tuple[Expr, ...]
    variables: tuple[VariableRef, ...]
    rows: tuple[tuple[GradientEntry, ...], ...]
    traces: tuple[tuple[DerivativeTraceEvent, ...], ...]


@dataclass(frozen=True)
class DerivativeCheck:
    """One analytic-vs-finite-difference derivative check."""

    output_index: int
    variable: VariableRef
    analytic: float
    finite_difference: float
    error: float
    tolerance: float
    passed: bool


@dataclass(frozen=True)
class DerivativeVerification:
    """Deterministic derivative verification result."""

    passed: bool
    checks: tuple[DerivativeCheck, ...]


def jacobian(outputs: Sequence[Expr]) -> Jacobian:
    """Construct a dense symbolic Jacobian for list/tuple scalar outputs."""

    graph = require_non_empty_same_graph(outputs)
    output_tuple = tuple(outputs)
    variables = variable_refs_for_expressions(output_tuple)
    rows: list[tuple[GradientEntry, ...]] = []
    traces: list[tuple[DerivativeTraceEvent, ...]] = []

    for output in output_tuple:
        result = gradient(output)
        derivatives_by_node = {
            entry.variable.node_id: entry.derivative for entry in result.entries
        }
        rows.append(
            tuple(
                GradientEntry(
                    variable=variable,
                    derivative=derivatives_by_node.get(
                        variable.node_id,
                        graph.constant(0.0),
                    ),
                )
                for variable in variables
            )
        )
        traces.append(result.trace)

    return Jacobian(
        outputs=output_tuple,
        variables=variables,
        rows=tuple(rows),
        traces=tuple(traces),
    )


def evaluate_jacobian(
    result: Jacobian,
    values: Mapping[str, float],
) -> list[list[float]]:
    """Evaluate a dense symbolic Jacobian."""

    return [
        [evaluate(entry.derivative, values) for entry in row] for row in result.rows
    ]


def verify_gradient(
    expr: Expr,
    values: Mapping[str, float],
    *,
    step: float = 1e-6,
    tolerance: float = 1e-5,
) -> DerivativeVerification:
    """Verify a scalar gradient with central finite differences."""

    result = gradient(expr)
    _ensure_positive_step(step)
    _ensure_unique_variable_names(tuple(entry.variable for entry in result.entries))
    checks = tuple(
        _check_derivative(
            output=expr,
            output_index=0,
            variable=entry.variable,
            analytic=evaluate(entry.derivative, values),
            values=values,
            step=step,
            tolerance=tolerance,
        )
        for entry in result.entries
    )
    return DerivativeVerification(
        passed=all(check.passed for check in checks),
        checks=checks,
    )


def verify_jacobian(
    outputs: Sequence[Expr],
    values: Mapping[str, float],
    *,
    step: float = 1e-6,
    tolerance: float = 1e-5,
) -> DerivativeVerification:
    """Verify a vector Jacobian with central finite differences."""

    result = jacobian(outputs)
    _ensure_positive_step(step)
    _ensure_unique_variable_names(result.variables)
    checks: list[DerivativeCheck] = []
    for output_index, output in enumerate(result.outputs):
        for entry in result.rows[output_index]:
            checks.append(
                _check_derivative(
                    output=output,
                    output_index=output_index,
                    variable=entry.variable,
                    analytic=evaluate(entry.derivative, values),
                    values=values,
                    step=step,
                    tolerance=tolerance,
                )
            )
    return DerivativeVerification(
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _check_derivative(
    *,
    output: Expr,
    output_index: int,
    variable: VariableRef,
    analytic: float,
    values: Mapping[str, float],
    step: float,
    tolerance: float,
) -> DerivativeCheck:
    finite_difference = _central_difference(output, variable.name, values, step)
    error = abs(analytic - finite_difference)
    return DerivativeCheck(
        output_index=output_index,
        variable=variable,
        analytic=analytic,
        finite_difference=finite_difference,
        error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
    )


def _central_difference(
    output: Expr,
    variable_name: str,
    values: Mapping[str, float],
    step: float,
) -> float:
    if variable_name not in values:
        msg = f"missing value for variable {variable_name!r}"
        raise ValueError(msg)
    forward = dict(values)
    backward = dict(values)
    forward[variable_name] = float(values[variable_name]) + step
    backward[variable_name] = float(values[variable_name]) - step
    return (evaluate(output, forward) - evaluate(output, backward)) / (2.0 * step)


def _ensure_positive_step(step: float) -> None:
    if step <= 0.0:
        msg = "finite-difference step must be positive"
        raise ValueError(msg)


def _ensure_unique_variable_names(variables: tuple[VariableRef, ...]) -> None:
    seen: set[str] = set()
    for variable in variables:
        if variable.name in seen:
            msg = (
                "derivative verification requires unique variable names; "
                f"{variable.name!r} appears more than once"
            )
            raise ValueError(msg)
        seen.add(variable.name)
