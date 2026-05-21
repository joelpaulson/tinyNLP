"""Small symbolic expression graph IR for tinyNLP."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final


class OpKind(StrEnum):
    """Supported symbolic operation kinds."""

    VARIABLE = "variable"
    CONSTANT = "constant"
    ADD = "add"
    SUB = "sub"
    NEG = "neg"
    MUL = "mul"
    DIV = "div"


@dataclass(frozen=True, order=True)
class NodeId:
    """Deterministic node identifier assigned by insertion order."""

    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            msg = "node id must be non-negative"
            raise ValueError(msg)

    def __int__(self) -> int:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


@dataclass(frozen=True)
class Node:
    """Symbolic IR node.

    Variable runtime values are intentionally not stored here. Constants are
    literal structure and can be inspected without binding variable values.
    """

    id: NodeId
    op: OpKind
    inputs: tuple[NodeId, ...] = ()
    name: str | None = None
    value: float | None = None


@dataclass(frozen=True)
class Expr:
    """Expression handle into a graph."""

    graph: Graph
    id: NodeId

    @property
    def node(self) -> Node:
        return self.graph.node(self.id)

    def __add__(self, other: Expr | float | int) -> Expr:
        return self.graph.binary(OpKind.ADD, self, self.graph.as_expr(other))

    def __radd__(self, other: float | int) -> Expr:
        return self.graph.binary(OpKind.ADD, self.graph.as_expr(other), self)

    def __sub__(self, other: Expr | float | int) -> Expr:
        return self.graph.binary(OpKind.SUB, self, self.graph.as_expr(other))

    def __rsub__(self, other: float | int) -> Expr:
        return self.graph.binary(OpKind.SUB, self.graph.as_expr(other), self)

    def __neg__(self) -> Expr:
        return self.graph.unary(OpKind.NEG, self)

    def __mul__(self, other: Expr | float | int) -> Expr:
        return self.graph.binary(OpKind.MUL, self, self.graph.as_expr(other))

    def __rmul__(self, other: float | int) -> Expr:
        return self.graph.binary(OpKind.MUL, self.graph.as_expr(other), self)

    def __truediv__(self, other: Expr | float | int) -> Expr:
        return self.graph.binary(OpKind.DIV, self, self.graph.as_expr(other))

    def __rtruediv__(self, other: float | int) -> Expr:
        return self.graph.binary(OpKind.DIV, self.graph.as_expr(other), self)


class Graph:
    """Append-only symbolic expression graph."""

    def __init__(self) -> None:
        self._nodes: list[Node] = []

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes)

    def variable(self, name: str) -> Expr:
        if not isinstance(name, str) or not name:
            msg = "variable name must be a non-empty string"
            raise ValueError(msg)
        return self._add_node(OpKind.VARIABLE, name=name)

    def constant(self, value: float | int) -> Expr:
        if isinstance(value, bool) or not isinstance(value, int | float):
            msg = "constant value must be an int or float"
            raise TypeError(msg)
        return self._add_node(OpKind.CONSTANT, value=float(value))

    def as_expr(self, value: Expr | float | int) -> Expr:
        if isinstance(value, Expr):
            self._check_same_graph(value)
            return value
        return self.constant(value)

    def node(self, node_id: NodeId) -> Node:
        return self._nodes[node_id.value]

    def unary(self, op: OpKind, operand: Expr) -> Expr:
        if op is not OpKind.NEG:
            msg = f"unsupported unary operation: {op}"
            raise ValueError(msg)
        self._check_same_graph(operand)
        return self._add_node(op, inputs=(operand.id,))

    def binary(self, op: OpKind, left: Expr, right: Expr) -> Expr:
        if op not in _BINARY_OPS:
            msg = f"unsupported binary operation: {op}"
            raise ValueError(msg)
        self._check_same_graph(left)
        self._check_same_graph(right)
        return self._add_node(op, inputs=(left.id, right.id))

    def _add_node(
        self,
        op: OpKind,
        *,
        inputs: tuple[NodeId, ...] = (),
        name: str | None = None,
        value: float | None = None,
    ) -> Expr:
        node_id = NodeId(len(self._nodes))
        self._nodes.append(
            Node(id=node_id, op=op, inputs=inputs, name=name, value=value)
        )
        return Expr(graph=self, id=node_id)

    def _check_same_graph(self, expr: Expr) -> None:
        if expr.graph is not self:
            msg = "expression belongs to a different graph"
            raise ValueError(msg)


_BINARY_OPS: Final = frozenset({OpKind.ADD, OpKind.SUB, OpKind.MUL, OpKind.DIV})
