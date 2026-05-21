import pytest

from tinynlp.ir import Graph, NodeId
from tinynlp.nlp import (
    AssemblyProvenance,
    CoordinateEntry,
    Problem,
    SparseMatrixAssembly,
    assemble_jacobian,
    build_assembly_contract,
)
from tinynlp.solvers import (
    KKTBlockKind,
    KKTError,
    build_kkt_system,
    format_kkt_system,
    kkt_to_dense,
)


def test_kkt_from_tiny_problem_has_expected_blocks_and_layout() -> None:
    jacobian = _tiny_jacobian()

    system = build_kkt_system(jacobian)

    assert system.primal_size == 2
    assert system.residual_size == 1
    assert system.shape == (3, 3)
    assert [
        (block.kind, block.row_range, block.column_range) for block in system.blocks
    ] == [
        (KKTBlockKind.PRIMAL, (0, 2), (0, 2)),
        (KKTBlockKind.JACOBIAN_TRANSPOSE, (0, 2), (2, 3)),
        (KKTBlockKind.JACOBIAN, (2, 3), (0, 2)),
        (KKTBlockKind.CONSTRAINT_ZERO, (2, 3), (2, 3)),
    ]
    assert kkt_to_dense(system) == [
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ]


def test_kkt_provenance_distinguishes_blocks() -> None:
    system = build_kkt_system(_tiny_jacobian())

    assert [entry.provenance.block for entry in system.entries] == [
        KKTBlockKind.PRIMAL,
        KKTBlockKind.PRIMAL,
        KKTBlockKind.JACOBIAN_TRANSPOSE,
        KKTBlockKind.JACOBIAN_TRANSPOSE,
        KKTBlockKind.JACOBIAN,
        KKTBlockKind.JACOBIAN,
    ]
    assert all(entry.provenance.source is None for entry in system.entries[:2])
    assert all(entry.provenance.source is not None for entry in system.entries[2:])
    assert all(
        entry.provenance.block is not KKTBlockKind.CONSTRAINT_ZERO
        for entry in system.entries
    )
    assert any(block.kind is KKTBlockKind.CONSTRAINT_ZERO for block in system.blocks)


def test_kkt_format_is_deterministic_and_address_free() -> None:
    first = format_kkt_system(build_kkt_system(_tiny_jacobian()))
    second = format_kkt_system(build_kkt_system(_tiny_jacobian()))

    assert first == second
    assert "KKTSystem shape=(3, 3)" in first
    assert "jacobian_transpose rows=[0, 2) cols=[2, 3)" in first
    assert "constraint_zero rows=[2, 3) cols=[2, 3)" in first
    assert "object at" not in first


def test_kkt_stress_problem_preserves_shape_and_structural_zeroes() -> None:
    jacobian = _coupled_jacobian()

    system = build_kkt_system(jacobian)

    assert system.primal_size == 7
    assert system.residual_size == 4
    assert system.shape == (11, 11)
    assert len(system.entries) == 31
    assert any(
        entry.row == 2
        and entry.column == 9
        and entry.value == 0.0
        and entry.provenance.block is KKTBlockKind.JACOBIAN_TRANSPOSE
        for entry in system.entries
    )
    assert any(
        entry.row == 9
        and entry.column == 2
        and entry.value == 0.0
        and entry.provenance.block is KKTBlockKind.JACOBIAN
        for entry in system.entries
    )
    assert all(
        not (
            entry.provenance.source is not None
            and entry.provenance.source.variable is not None
            and entry.provenance.source.variable.name == "q"
        )
        for entry in system.entries
    )


def test_custom_primal_block_is_accepted_as_coordinate_assembly() -> None:
    jacobian = _tiny_jacobian()
    primal = SparseMatrixAssembly(
        shape=(2, 2),
        entries=(
            CoordinateEntry(0, 0, 2.0, _provenance()),
            CoordinateEntry(1, 1, 3.0, _provenance()),
        ),
    )

    system = build_kkt_system(jacobian, primal_block=primal)

    assert kkt_to_dense(system) == [
        [2.0, 0.0, 1.0],
        [0.0, 3.0, 1.0],
        [1.0, 1.0, 0.0],
    ]
    assert system.entries[0].provenance.source is not None


def test_kkt_rejects_invalid_primal_blocks() -> None:
    jacobian = _tiny_jacobian()

    with pytest.raises(KKTError, match="square"):
        build_kkt_system(
            jacobian,
            primal_block=SparseMatrixAssembly(shape=(2, 1), entries=()),
        )

    with pytest.raises(KKTError, match="match Jacobian column dimension"):
        build_kkt_system(
            jacobian,
            primal_block=SparseMatrixAssembly(shape=(1, 1), entries=()),
        )


def test_kkt_rejects_duplicate_and_out_of_bounds_coordinates() -> None:
    with pytest.raises(KKTError, match="duplicate coordinate"):
        build_kkt_system(
            SparseMatrixAssembly(
                shape=(1, 1),
                entries=(
                    CoordinateEntry(0, 0, 1.0, _provenance()),
                    CoordinateEntry(0, 0, 2.0, _provenance()),
                ),
            )
        )

    with pytest.raises(KKTError, match="outside shape"):
        build_kkt_system(
            SparseMatrixAssembly(
                shape=(1, 1),
                entries=(CoordinateEntry(1, 0, 1.0, _provenance()),),
            )
        )


def _tiny_jacobian() -> SparseMatrixAssembly:
    graph = Graph()
    x = graph.variable("x")
    y = graph.variable("y")
    problem = Problem.from_residuals([x + y])
    contract = build_assembly_contract(problem)
    return assemble_jacobian(contract, {"x": 1.0, "y": 2.0})


def _coupled_jacobian() -> SparseMatrixAssembly:
    problem, values = _coupled_problem()
    contract = build_assembly_contract(problem)
    return assemble_jacobian(contract, values)


def _coupled_problem() -> tuple[Problem, dict[str, float]]:
    graph = Graph()
    x0 = graph.variable("x0")
    x1 = graph.variable("x1")
    x2 = graph.variable("x2")
    x3 = graph.variable("x3")
    x4 = graph.variable("x4")
    x5 = graph.variable("x5")
    q = graph.variable("q")
    residuals = [
        ((x0 * x0) + x1) / (x1 + 1) - 1,
        (x1 * x2) - (x2 / x3),
        (x2 - x2) + (x3 * x4),
        ((x0 + x4) / (x2 + 1)) - x5,
    ]
    values = {
        "x0": 2.0,
        "x1": 3.0,
        "x2": 4.0,
        "x3": 5.0,
        "x4": 6.0,
        "x5": 7.0,
        "q": 11.0,
    }
    return Problem.from_residuals(
        residuals,
        name="coupled",
        objective=(q * q) + x0,
    ), values


def _provenance() -> AssemblyProvenance:
    return AssemblyProvenance(
        kind="test",
        row=0,
        column=0,
        source_node_id=NodeId(0),
    )
