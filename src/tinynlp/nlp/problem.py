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

        block = ResidualBlock(
            name="residuals",
            kind=ResidualKind.EQUALITY,
            expressions=tuple(residuals),
        )
        return cls.from_blocks((block,), name=name, objective=objective)

    @classmethod
    def from_blocks(
        cls,
        blocks: Sequence[ResidualBlock],
        *,
        name: str = "problem",
        objective: Expr | None = None,
    ) -> Problem:
        """Build a problem from explicit residual blocks.

        Equality residual blocks are the first implemented block kind. Keeping
        block names explicit makes the current path easier to inspect and leaves
        room for future inequality and bound block types.
        """

        block_tuple = tuple(blocks)
        if not block_tuple:
            msg = "problem must contain at least one residual block"
            raise ValueError(msg)
        residual_tuple = tuple(
            expression for block in block_tuple for expression in block.expressions
        )
        graph = require_non_empty_same_graph(list(residual_tuple))
        for block in block_tuple:
            for expression in block.expressions:
                if expression.graph is not graph:
                    msg = "all residual blocks must belong to the same graph"
                    raise ValueError(msg)
        if objective is not None and objective.graph is not graph:
            msg = "objective must belong to the same graph as residuals"
            raise ValueError(msg)

        expressions_for_variables = residual_tuple
        if objective is not None:
            expressions_for_variables = (*expressions_for_variables, objective)
        variables = variable_refs_for_expressions(list(expressions_for_variables))
        _ensure_unique_variable_names(variables)

        return cls(
            name=_validate_problem_name(name),
            graph=graph,
            variables=variables,
            objective=objective,
            residual_blocks=block_tuple,
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
