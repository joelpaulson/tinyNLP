"""Problem-level containers for supported tinyNLP expression systems."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from tinynlp.ir import Expr, Graph, VariableRef
from tinynlp.ir.analysis import (
    require_non_empty_same_graph,
    variable_refs_for_expressions,
)


class ResidualKind(StrEnum):
    """Supported residual block kinds."""

    EQUALITY = "equality"


@dataclass(frozen=True)
class ResidualBlock:
    """A named block of residual expressions."""

    name: str
    kind: ResidualKind
    expressions: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if not self.name:
            msg = "residual block name must be non-empty"
            raise ValueError(msg)
        if not self.expressions:
            msg = "residual block must contain at least one expression"
            raise ValueError(msg)
        require_non_empty_same_graph(list(self.expressions))


@dataclass(frozen=True)
class Problem:
    """Symbolic problem definition for the current expression pipeline."""

    name: str
    graph: Graph
    variables: tuple[VariableRef, ...]
    objective: Expr | None
    residual_blocks: tuple[ResidualBlock, ...]

    @classmethod
    def from_residuals(
        cls,
        residuals: Sequence[Expr],
        *,
        name: str = "problem",
        objective: Expr | None = None,
    ) -> Problem:
        """Build a problem from supported residual expressions.

        Equality residuals are the first implemented block kind. The container
        keeps block metadata explicit so later inequality and bound blocks can
        be added without changing the current residual assembly contract.
        """

        residual_tuple = tuple(residuals)
        graph = require_non_empty_same_graph(list(residual_tuple))
        if objective is not None and objective.graph is not graph:
            msg = "objective must belong to the same graph as residuals"
            raise ValueError(msg)

        expressions_for_variables = residual_tuple
        if objective is not None:
            expressions_for_variables = (*expressions_for_variables, objective)
        variables = variable_refs_for_expressions(list(expressions_for_variables))
        _ensure_unique_variable_names(variables)

        block = ResidualBlock(
            name="residuals",
            kind=ResidualKind.EQUALITY,
            expressions=residual_tuple,
        )
        return cls(
            name=_validate_problem_name(name),
            graph=graph,
            variables=variables,
            objective=objective,
            residual_blocks=(block,),
        )

    @property
    def residuals(self) -> tuple[Expr, ...]:
        """Return all residual expressions in block order."""

        return tuple(
            expression
            for block in self.residual_blocks
            for expression in block.expressions
        )

    @property
    def variable_dimension(self) -> int:
        """Number of symbolic variables in the problem."""

        return len(self.variables)

    @property
    def residual_dimension(self) -> int:
        """Number of residual rows in the problem."""

        return len(self.residuals)


def _ensure_unique_variable_names(variables: tuple[VariableRef, ...]) -> None:
    seen: set[str] = set()
    for variable in variables:
        if variable.name in seen:
            msg = (
                "problem variables must have unique names; "
                f"{variable.name!r} appears more than once"
            )
            raise ValueError(msg)
        seen.add(variable.name)


def _validate_problem_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        msg = "problem name must be a non-empty string"
        raise ValueError(msg)
    return name
