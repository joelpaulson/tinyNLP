import importlib.util
from pathlib import Path
from types import ModuleType

from tinynlp.ir import Graph
from tinynlp.nlp import Problem, build_assembly_contract, format_assembly_contract


def _load_banded_example() -> ModuleType:
    examples_path = Path(__file__).resolve().parents[2] / "examples"
    module_path = examples_path / "banded_residual_system.py"
    spec = importlib.util.spec_from_file_location("banded_residual_system", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


banded_example = _load_banded_example()


def test_assembly_contract_has_banded_terms_in_row_major_order() -> None:
    residuals = banded_example.banded_residual_expressions(size=6)
    problem = Problem.from_residuals(residuals, name="banded")

    contract = build_assembly_contract(problem)

    assert len(contract.residual_terms) == 5
    assert len(contract.jacobian_terms) == 10
    assert contract.sparsity.shape == (5, 6)
    assert [
        (term.row, term.column, term.variable.name) for term in contract.jacobian_terms
    ] == [
        item
        for row in range(5)
        for item in [(row, row, f"x{row}"), (row, row + 1, f"x{row + 1}")]
    ]


def test_assembly_contract_format_is_deterministic_and_address_free() -> None:
    first = format_assembly_contract(build_assembly_contract(_tiny_problem()))
    second = format_assembly_contract(build_assembly_contract(_tiny_problem()))

    assert first == second
    assert "AssemblyContract problem=tiny" in first
    assert "dimensions: variables=2 residuals=1 jacobian_terms=2" in first
    assert "row=0 col=0 variable=x@0" in first
    assert "row=0 col=1 variable=y@1" in first
    assert "object at" not in first


def test_assembly_contract_needs_no_numeric_values() -> None:
    residuals = banded_example.banded_residual_expressions(size=6)
    problem = Problem.from_residuals(residuals, name="banded")

    contract = build_assembly_contract(problem)

    assert [term.kernel_plan.output for term in contract.residual_terms] == [
        expr.id for expr in residuals
    ]


def test_assembly_contract_uses_problem_variable_columns() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    problem = Problem.from_residuals([x - 1], objective=y * y)

    contract = build_assembly_contract(problem)

    assert contract.sparsity.shape == (1, 2)
    assert [variable.name for variable in contract.variables] == ["x", "y"]
    assert [
        (term.row, term.column, term.variable.name) for term in contract.jacobian_terms
    ] == [(0, 0, "x")]


def _tiny_problem() -> Problem:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    return Problem.from_residuals([x + y - 1], name="tiny")
