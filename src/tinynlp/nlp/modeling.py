"""Small ergonomic modeling helpers for tinyNLP examples."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real

from tinynlp.ir import Expr, Graph, OpKind
from tinynlp.ir.analysis import require_non_empty_same_graph
from tinynlp.nlp.problem import ResidualBlock, ResidualKind


@dataclass(frozen=True)
class VariableArray:
    """Deterministic sequence of named scalar variables."""

    prefix: str
    variables: tuple[Expr, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.prefix, str) or not self.prefix:
            msg = "variable array prefix must be a non-empty string"
            raise ValueError(msg)
        if not self.variables:
            msg = "variable array must contain at least one variable"
            raise ValueError(msg)
        require_non_empty_same_graph(list(self.variables))
        for variable in self.variables:
            if variable.node.op is not OpKind.VARIABLE or variable.node.name is None:
                msg = "variable array entries must be named variables"
                raise ValueError(msg)

    @property
    def names(self) -> tuple[str, ...]:
        """Variable names in deterministic array order."""

        names = []
        for variable in self.variables:
            name = variable.node.name
            if name is None:
                msg = f"variable node {variable.id} is missing a name"
                raise ValueError(msg)
            names.append(name)
        return tuple(names)

    def value_map(self, values: Sequence[float | int]) -> dict[str, float]:
        """Return explicit numeric values keyed by variable name."""

        value_tuple = tuple(values)
        if len(value_tuple) != len(self.variables):
            msg = (
                "value count must match variable array size; "
                f"got {len(value_tuple)} for {len(self.variables)} variables"
            )
            raise ValueError(msg)
        return {
            name: _as_float(value, label=f"value for {name!r}")
            for name, value in zip(self.names, value_tuple, strict=True)
        }

    def __len__(self) -> int:
        return len(self.variables)

    def __iter__(self) -> Iterator[Expr]:
        return iter(self.variables)

    def __getitem__(self, index: int) -> Expr:
        return self.variables[index]


def variable_array(graph: Graph, prefix: str, size: int) -> VariableArray:
    """Create a deterministic named variable array on a graph."""

    if not isinstance(graph, Graph):
        msg = "graph must be a Graph"
        raise TypeError(msg)
    if not isinstance(prefix, str) or not prefix:
        msg = "variable array prefix must be a non-empty string"
        raise ValueError(msg)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        msg = "variable array size must be a positive integer"
        raise ValueError(msg)
    return VariableArray(
        prefix=prefix,
        variables=tuple(graph.variable(f"{prefix}{index}") for index in range(size)),
    )


def merge_value_maps(*maps: Mapping[str, float | int]) -> dict[str, float]:
    """Merge explicit value maps and reject duplicate names."""

    merged: dict[str, float] = {}
    for value_map in maps:
        for name, value in value_map.items():
            if not isinstance(name, str) or not name:
                msg = "value map keys must be non-empty strings"
                raise ValueError(msg)
            if name in merged:
                msg = f"duplicate value for variable {name!r}"
                raise ValueError(msg)
            merged[name] = _as_float(value, label=f"value for {name!r}")
    return merged


def sum_expressions(expressions: Sequence[Expr]) -> Expr:
    """Sum a non-empty sequence of expressions from the same graph."""

    expression_tuple = tuple(expressions)
    require_non_empty_same_graph(list(expression_tuple))
    result = expression_tuple[0]
    for expression in expression_tuple[1:]:
        result = result + expression
    return result


def residual_block(
    name: str,
    expressions: Sequence[Expr],
    *,
    kind: ResidualKind = ResidualKind.EQUALITY,
) -> ResidualBlock:
    """Create a named residual block for the current problem API."""

    if isinstance(kind, str):
        kind = ResidualKind(kind)
    if kind is not ResidualKind.EQUALITY:
        msg = f"unsupported residual block kind: {kind}"
        raise ValueError(msg)
    return ResidualBlock(name=name, kind=kind, expressions=tuple(expressions))


def _as_float(value: float | int, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        msg = f"{label} must be a real number"
        raise TypeError(msg)
    return float(value)
