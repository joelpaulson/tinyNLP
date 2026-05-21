import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

from tinynlp.ir import Graph
from tinynlp.nlp import Problem, ResidualKind


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


def test_problem_from_banded_residuals_exposes_dimensions() -> None:
    residuals = banded_example.banded_residual_expressions(size=6)

    problem = Problem.from_residuals(residuals, name="banded")

    assert problem.name == "banded"
    assert problem.residual_dimension == 5
    assert problem.variable_dimension == 6
    assert [variable.name for variable in problem.variables] == [
        f"x{index}" for index in range(6)
    ]
    assert problem.residual_blocks[0].kind is ResidualKind.EQUALITY
    assert problem.residuals == tuple(residuals)


def test_problem_can_include_same_graph_objective_variables() -> None:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")

    problem = Problem.from_residuals([x - 1], objective=y * y)

    assert [variable.name for variable in problem.variables] == ["x", "y"]
    assert problem.objective is not None


def test_problem_rejects_empty_residuals() -> None:
    with pytest.raises(ValueError, match="at least one expression"):
        Problem.from_residuals([])


def test_problem_rejects_mixed_graph_residuals() -> None:
    first = Graph().variable("x")
    second = Graph().variable("y")

    with pytest.raises(ValueError, match="same graph"):
        Problem.from_residuals([first, second])


def test_problem_rejects_duplicate_variable_names() -> None:
    graph = Graph()
    first = graph.variable("x")
    second = graph.variable("x")

    with pytest.raises(ValueError, match="unique names"):
        Problem.from_residuals([first + second])


def test_problem_rejects_objective_from_different_graph() -> None:
    residual = Graph().variable("x")
    objective = Graph().variable("y")

    with pytest.raises(ValueError, match="same graph"):
        Problem.from_residuals([residual], objective=objective)


def test_problem_rejects_empty_name() -> None:
    residual = Graph().variable("x")

    with pytest.raises(ValueError, match="problem name"):
        Problem.from_residuals([residual], name="")
