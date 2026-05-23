import pytest

from tinynlp.ir import Graph, OpKind
from tinynlp.nlp import (
    Problem,
    ResidualKind,
    VariableArray,
    merge_value_maps,
    residual_block,
    sum_expressions,
    variable_array,
)


def test_variable_array_creates_stable_names_and_values() -> None:
    graph = Graph()
    variables = variable_array(graph, "x", 3)

    assert len(variables) == 3
    assert variables.names == ("x0", "x1", "x2")
    assert [variable.id.value for variable in variables] == [0, 1, 2]
    assert variables[1].node.name == "x1"
    assert variables.value_map([1, 2.5, 3]) == {
        "x0": 1.0,
        "x1": 2.5,
        "x2": 3.0,
    }


def test_variable_array_rejects_invalid_inputs() -> None:
    graph = Graph()

    with pytest.raises(ValueError, match="prefix"):
        variable_array(graph, "", 2)
    with pytest.raises(ValueError, match="positive integer"):
        variable_array(graph, "x", 0)
    with pytest.raises(ValueError, match="value count"):
        variable_array(graph, "x", 2).value_map([1.0])
    with pytest.raises(TypeError, match="real number"):
        variable_array(graph, "y", 1).value_map([True])
    with pytest.raises(ValueError, match="named variables"):
        VariableArray(prefix="bad", variables=(graph.constant(1.0),))


def test_merge_value_maps_preserves_values_and_rejects_duplicates() -> None:
    assert merge_value_maps({"x0": 1}, {"u0": 2.5}) == {"x0": 1.0, "u0": 2.5}

    with pytest.raises(ValueError, match="duplicate"):
        merge_value_maps({"x0": 1.0}, {"x0": 2.0})
    with pytest.raises(ValueError, match="non-empty"):
        merge_value_maps({"": 1.0})
    with pytest.raises(TypeError, match="real number"):
        merge_value_maps({"x0": object()})


def test_sum_expressions_requires_non_empty_same_graph_sequence() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")

    total = sum_expressions([x, y, 2.0 * x])

    assert total.graph is graph
    assert total.node.op is OpKind.ADD

    with pytest.raises(ValueError, match="at least one"):
        sum_expressions([])
    with pytest.raises(ValueError, match="same graph"):
        sum_expressions([x, Graph().variable("z")])


def test_residual_block_preserves_name_and_kind() -> None:
    graph = Graph()
    x = graph.variable("x")
    block = residual_block("dynamics", [x - 1.0])

    assert block.name == "dynamics"
    assert block.kind is ResidualKind.EQUALITY
    assert len(block.expressions) == 1

    with pytest.raises(ValueError, match="non-empty"):
        residual_block("", [x])
    with pytest.raises(ValueError, match="at least one"):
        residual_block("empty", [])


def test_problem_from_blocks_preserves_block_order_and_variables() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    objective = (x - 1.0) * (x - 1.0) + y * y
    blocks = (
        residual_block("first", [x + y - 1.0]),
        residual_block("second", [x * x - y]),
    )

    problem = Problem.from_blocks(blocks, name="blocked", objective=objective)

    assert problem.name == "blocked"
    assert [block.name for block in problem.residual_blocks] == ["first", "second"]
    assert [variable.name for variable in problem.variables] == ["x", "y"]
    assert problem.variable_dimension == 2
    assert problem.residual_dimension == 2
    assert problem.objective is objective


def test_problem_from_blocks_rejects_invalid_structures() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = Graph().variable("y")

    with pytest.raises(ValueError, match="at least one residual block"):
        Problem.from_blocks(())
    with pytest.raises(ValueError, match="same graph"):
        Problem.from_blocks(
            (
                residual_block("x", [x]),
                residual_block("y", [y]),
            )
        )
    with pytest.raises(ValueError, match="same graph"):
        Problem.from_blocks(
            (residual_block("x", [x]),),
            objective=Graph().variable("z"),
        )


def test_problem_from_residuals_still_uses_default_block_name() -> None:
    graph = Graph()
    x = graph.variable("x")

    problem = Problem.from_residuals([x - 1.0], name="legacy")

    assert [block.name for block in problem.residual_blocks] == ["residuals"]
