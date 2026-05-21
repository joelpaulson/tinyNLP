from tinynlp.ir import Graph
from tinynlp.profiling import TraceEvent, format_trace, trace_expression


def test_trace_event_order_is_stable() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    expr = (x + 1) * y

    first = trace_expression(expr)
    second = trace_expression(expr)

    assert first == second
    assert [int(event.node_id) for event in first] == [0, 1, 2, 3, 4]
    assert all(isinstance(event, TraceEvent) for event in first)


def test_trace_only_includes_reachable_nodes() -> None:
    graph = Graph()
    _unused = graph.variable("unused")
    x = graph.variable("x")
    expr = x + 1

    events = trace_expression(expr)

    assert [int(event.node_id) for event in events] == [1, 2, 3]


def test_formatted_trace_is_deterministic() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = (x + 2) * -x

    formatted = format_trace(trace_expression(expr))

    assert formatted == "\n".join(
        [
            "0: variable inputs=[] name=x",
            "1: constant inputs=[] value=2",
            "2: add inputs=[0, 1]",
            "3: neg inputs=[0]",
            "4: mul inputs=[2, 3]",
        ]
    )
    assert "object at" not in formatted
