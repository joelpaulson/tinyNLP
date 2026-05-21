from tinynlp.backends import evaluate
from tinynlp.ir import Graph, NodeId, VariableRef
from tinynlp.nlp import (
    SparsityEntry,
    expression_dependencies,
    format_sparsity,
    jacobian_sparsity,
)


def test_expression_dependencies_ignore_unused_variables_and_constants() -> None:
    graph = Graph()
    _unused = graph.variable("unused")
    x = graph.variable("x")
    expr = x + 1

    dependencies = expression_dependencies(expr)

    assert dependencies == (VariableRef(name="x", node_id=NodeId(1)),)
    assert expression_dependencies(graph.constant(3)) == ()


def test_jacobian_sparsity_is_deterministic_and_row_major() -> None:
    def build_pattern_signature() -> tuple[object, ...]:
        graph = Graph()
        x = graph.variable("x")
        y = graph.variable("y")
        outputs = [x + y - 1, (x * x) + y - 2]
        pattern = jacobian_sparsity(outputs)
        return (
            pattern.shape,
            pattern.variables,
            pattern.entries,
            pattern.trace,
        )

    assert build_pattern_signature() == build_pattern_signature()


def test_jacobian_sparsity_entries_match_residual_like_outputs() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    outputs = [x + y - 1, (x * x) + y - 2]

    pattern = jacobian_sparsity(outputs)

    assert pattern.shape == (2, 2)
    assert pattern.variables == (
        VariableRef(name="x", node_id=NodeId(0)),
        VariableRef(name="y", node_id=NodeId(1)),
    )
    assert pattern.entries == (
        SparsityEntry(row=0, column=0, variable=pattern.variables[0]),
        SparsityEntry(row=0, column=1, variable=pattern.variables[1]),
        SparsityEntry(row=1, column=0, variable=pattern.variables[0]),
        SparsityEntry(row=1, column=1, variable=pattern.variables[1]),
    )


def test_sparsity_pattern_is_stable_across_value_changes() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    outputs = [x + y - 1, (x * x) + y - 2]

    before = jacobian_sparsity(outputs)
    assert evaluate(outputs[0], {"x": 1, "y": 2}) == 2.0
    assert evaluate(outputs[1], {"x": 3, "y": 4}) == 11.0
    after = jacobian_sparsity(outputs)

    assert after == before


def test_sparsity_is_conservative_without_algebraic_cancellation() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = x - x

    assert expression_dependencies(expr) == (VariableRef(name="x", node_id=NodeId(0)),)
    assert jacobian_sparsity([expr]).entries == (
        SparsityEntry(
            row=0,
            column=0,
            variable=VariableRef(name="x", node_id=NodeId(0)),
        ),
    )


def test_formatted_sparsity_is_stable_and_human_readable() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    pattern = jacobian_sparsity([x + y - 1])

    formatted = format_sparsity(pattern)

    assert formatted == "\n".join(
        [
            "SparsityPattern shape=(1, 2)",
            "variables:",
            "  col=0 node=0 name=x",
            "  col=1 node=1 name=y",
            "entries:",
            "  row=0 col=0 node=0 name=x",
            "  row=0 col=1 node=1 name=y",
            "trace:",
            "  row=0 node=0 op=variable inputs=[] deps=[x@0]",
            "  row=0 node=1 op=variable inputs=[] deps=[y@1]",
            "  row=0 node=2 op=add inputs=[0, 1] deps=[x@0, y@1]",
            "  row=0 node=3 op=constant inputs=[] deps=[]",
            "  row=0 node=4 op=sub inputs=[2, 3] deps=[x@0, y@1]",
        ]
    )
    assert "object at" not in formatted
