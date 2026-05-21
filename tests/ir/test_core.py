import pytest

from tinynlp.ir import Graph, NodeId, OpKind


def test_node_ids_are_assigned_by_insertion_order() -> None:
    graph = Graph()
    x = graph.variable("x")
    one = graph.constant(1)
    expr = x + one

    assert [node.id for node in graph.nodes] == [NodeId(0), NodeId(1), NodeId(2)]
    assert expr.id == NodeId(2)


def test_equivalent_construction_has_stable_structure() -> None:
    def build_ops() -> list[
        tuple[OpKind, tuple[NodeId, ...], str | None, float | None]
    ]:
        graph = Graph()
        x = graph.variable("x")
        y = graph.variable("y")
        expr = (x * 2) + (-y)
        assert expr.id == NodeId(5)
        return [(node.op, node.inputs, node.name, node.value) for node in graph.nodes]

    assert build_ops() == build_ops()


def test_variables_and_constants_do_not_bind_runtime_values() -> None:
    graph = Graph()
    x = graph.variable("x")
    c = graph.constant(3)

    assert x.node.op is OpKind.VARIABLE
    assert x.node.name == "x"
    assert x.node.value is None
    assert c.node.op is OpKind.CONSTANT
    assert c.node.value == 3.0


def test_arithmetic_operations_create_expected_nodes() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")

    expr = ((x + y) - 1) * (x / -y)

    ops = [node.op for node in graph.nodes]
    assert ops == [
        OpKind.VARIABLE,
        OpKind.VARIABLE,
        OpKind.ADD,
        OpKind.CONSTANT,
        OpKind.SUB,
        OpKind.NEG,
        OpKind.DIV,
        OpKind.MUL,
    ]
    assert graph.node(expr.id).inputs == (NodeId(4), NodeId(6))


def test_inspecting_nodes_does_not_mutate_graph() -> None:
    graph = Graph()
    expr = graph.variable("x") + graph.constant(1)
    before = graph.nodes

    assert expr.node == graph.node(expr.id)
    assert graph.nodes == before


def test_rejects_cross_graph_operations() -> None:
    left = Graph().variable("x")
    right = Graph().variable("x")

    with pytest.raises(ValueError, match="different graph"):
        _ = left + right
