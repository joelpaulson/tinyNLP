import pytest

import tinynlp.solvers as solvers
from tinynlp.ir import Graph
from tinynlp.nlp import (
    Problem,
    SparseMatrixAssembly,
    assemble_jacobian,
    build_assembly_contract,
)
from tinynlp.solvers import (
    DenseReferenceLinearSolver,
    LinearSolveError,
    build_kkt_system,
)


def test_dense_reference_linear_solver_solves_tiny_kkt_system() -> None:
    system = _tiny_system()

    result = DenseReferenceLinearSolver().solve(system, [1.0, 2.0, 3.0])

    assert result.solver_name == "dense-reference"
    assert result.solution == pytest.approx((1.0, 2.0, 0.0))
    assert result.residual == pytest.approx((0.0, 0.0, 0.0))


def test_dense_reference_linear_solver_is_deterministic_and_non_mutating() -> None:
    system = _tiny_system()
    signature = tuple(
        (entry.row, entry.column, entry.value) for entry in system.entries
    )
    solver = DenseReferenceLinearSolver()

    first = solver.solve(system, [1.0, 2.0, 3.0])
    second = solver.solve(system, [1.0, 2.0, 3.0])

    assert second == first
    assert (
        tuple((entry.row, entry.column, entry.value) for entry in system.entries)
        == signature
    )


def test_dense_reference_linear_solver_rejects_invalid_rhs_length() -> None:
    with pytest.raises(LinearSolveError, match="right-hand side length"):
        DenseReferenceLinearSolver().solve(_tiny_system(), [1.0, 2.0])


def test_dense_reference_linear_solver_rejects_singular_systems() -> None:
    singular = build_kkt_system(
        SparseMatrixAssembly(shape=(0, 1), entries=()),
        primal_block=SparseMatrixAssembly(shape=(1, 1), entries=()),
    )

    with pytest.raises(LinearSolveError, match="singular KKT system"):
        DenseReferenceLinearSolver().solve(singular, [1.0])


def test_solver_namespace_exposes_only_prototype_solver_api() -> None:
    assert hasattr(solvers, "solve_constraints")
    assert hasattr(solvers, "SolverResult")
    assert hasattr(solvers, "IterationRecord")
    assert not hasattr(solvers, "solve")
    assert not hasattr(solvers, "SolverStep")
    assert not hasattr(solvers, "ConvergencePolicy")


def _tiny_system():
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    problem = Problem.from_residuals([x + y])
    contract = build_assembly_contract(problem)
    jacobian = assemble_jacobian(contract, {"x": 1.0, "y": 2.0})
    return build_kkt_system(jacobian)
