from tinynlp.backends import build_kernel_plan, format_kernel_plan
from tinynlp.ir import Graph, NodeId, OpKind


def test_kernel_plan_is_deterministic() -> None:
    def build_plan_signature() -> tuple[object, ...]:
        graph = Graph()
        x = graph.variable("x")
        expr = (x + 2) * -x
        plan = build_kernel_plan(expr)
        return (
            plan.output,
            plan.variables,
            plan.constants,
            plan.steps,
            plan.stats.operation_counts,
            plan.stats.temporary_count,
        )

    assert build_plan_signature() == build_plan_signature()


def test_kernel_plan_uses_reachable_topological_order() -> None:
    graph = Graph()
    _unused = graph.variable("unused")
    x = graph.variable("x")
    expr = (x + 2) * -x

    plan = build_kernel_plan(expr)

    assert plan.variables == ((NodeId(1), "x"),)
    assert plan.constants == ((NodeId(2), 2.0),)
    assert [step.node_id for step in plan.steps] == [NodeId(3), NodeId(4), NodeId(5)]
    assert [step.inputs for step in plan.steps] == [
        (NodeId(1), NodeId(2)),
        (NodeId(1),),
        (NodeId(3), NodeId(4)),
    ]


def test_kernel_plan_counts_operations_and_temporaries() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = (x + 2) * -x

    plan = build_kernel_plan(expr)

    assert plan.stats.operation_counts == (
        (OpKind.VARIABLE, 1),
        (OpKind.CONSTANT, 1),
        (OpKind.ADD, 1),
        (OpKind.NEG, 1),
        (OpKind.MUL, 1),
    )
    assert plan.stats.count(OpKind.SUB) == 0
    assert plan.stats.temporary_count == 3


def test_kernel_plan_report_is_stable_and_human_readable() -> None:
    graph = Graph()
    x = graph.variable("x")
    expr = (x + 2) * -x

    report = format_kernel_plan(build_kernel_plan(expr))

    assert report == "\n".join(
        [
            "KernelPlan output=4",
            "variables:",
            "  0: x",
            "constants:",
            "  1: 2",
            "steps:",
            "  2: add inputs=[0, 1]",
            "  3: neg inputs=[0]",
            "  4: mul inputs=[2, 3]",
            "counts:",
            "  variable: 1",
            "  constant: 1",
            "  add: 1",
            "  neg: 1",
            "  mul: 1",
            "temporaries: 3",
        ]
    )
    assert "object at" not in report
