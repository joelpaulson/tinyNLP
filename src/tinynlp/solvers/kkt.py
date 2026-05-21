"""Explicit KKT system objects for assembled Jacobians."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tinynlp.nlp import (
    AssemblyProvenance,
    CoordinateEntry,
    SparseMatrixAssembly,
)


class KKTError(ValueError):
    """Raised when a KKT system cannot be built from supplied blocks."""


class KKTBlockKind(StrEnum):
    """Named KKT matrix blocks."""

    PRIMAL = "primal"
    JACOBIAN_TRANSPOSE = "jacobian_transpose"
    JACOBIAN = "jacobian"
    CONSTRAINT_ZERO = "constraint_zero"


@dataclass(frozen=True)
class KKTBlock:
    """Half-open KKT matrix block ranges."""

    kind: KKTBlockKind
    row_range: tuple[int, int]
    column_range: tuple[int, int]


@dataclass(frozen=True)
class KKTProvenance:
    """Stable provenance for one KKT matrix entry."""

    block: KKTBlockKind
    source: AssemblyProvenance | None = None


@dataclass(frozen=True)
class KKTEntry:
    """One coordinate entry in a KKT matrix."""

    row: int
    column: int
    value: float
    provenance: KKTProvenance


@dataclass(frozen=True)
class KKTSystem:
    """Explicit KKT coordinate matrix plus block metadata."""

    primal_size: int
    residual_size: int
    shape: tuple[int, int]
    entries: tuple[KKTEntry, ...]
    blocks: tuple[KKTBlock, ...]


def build_kkt_system(
    jacobian_assembly: SparseMatrixAssembly,
    *,
    primal_block: SparseMatrixAssembly | None = None,
) -> KKTSystem:
    """Build a KKT system from an assembled residual Jacobian.

    The default primal block is an identity reference block. It is not a
    Hessian assembly path.
    """

    residual_size, primal_size = _validate_matrix("jacobian", jacobian_assembly)
    if primal_block is None:
        primal_entries = _identity_primal_entries(primal_size)
    else:
        primal_entries = _custom_primal_entries(primal_block, primal_size)

    total_size = primal_size + residual_size
    entries = (
        *primal_entries,
        *_jacobian_transpose_entries(jacobian_assembly, primal_size),
        *_jacobian_entries(jacobian_assembly, primal_size),
    )
    return KKTSystem(
        primal_size=primal_size,
        residual_size=residual_size,
        shape=(total_size, total_size),
        entries=entries,
        blocks=_blocks(primal_size, residual_size),
    )


def kkt_to_dense(system: KKTSystem) -> list[list[float]]:
    """Convert a KKT coordinate system to a dense matrix for reference checks."""

    rows, columns = system.shape
    dense = [[0.0 for _ in range(columns)] for _ in range(rows)]
    _validate_kkt_entries(system)
    for entry in system.entries:
        dense[entry.row][entry.column] = entry.value
    return dense


def format_kkt_system(system: KKTSystem) -> str:
    """Format KKT metadata and entries deterministically."""

    lines = [
        (
            "KKTSystem "
            f"shape={system.shape} "
            f"primal_size={system.primal_size} "
            f"residual_size={system.residual_size}"
        ),
        "blocks:",
    ]
    lines.extend(_format_block(block) for block in system.blocks)
    lines.append("entries:")
    lines.extend(_format_entry(entry) for entry in system.entries)
    return "\n".join(lines)


def _validate_matrix(
    name: str,
    matrix: SparseMatrixAssembly,
) -> tuple[int, int]:
    rows, columns = matrix.shape
    if rows < 0 or columns < 0:
        msg = f"{name} shape dimensions must be non-negative"
        raise KKTError(msg)

    seen: set[tuple[int, int]] = set()
    for entry in matrix.entries:
        if entry.row < 0 or entry.row >= rows:
            msg = f"{name} entry row {entry.row} is outside shape {matrix.shape}"
            raise KKTError(msg)
        if entry.column < 0 or entry.column >= columns:
            msg = f"{name} entry column {entry.column} is outside shape {matrix.shape}"
            raise KKTError(msg)
        coordinate = (entry.row, entry.column)
        if coordinate in seen:
            msg = f"{name} contains duplicate coordinate {coordinate}"
            raise KKTError(msg)
        seen.add(coordinate)
    return rows, columns


def _identity_primal_entries(primal_size: int) -> tuple[KKTEntry, ...]:
    return tuple(
        KKTEntry(
            row=index,
            column=index,
            value=1.0,
            provenance=KKTProvenance(block=KKTBlockKind.PRIMAL),
        )
        for index in range(primal_size)
    )


def _custom_primal_entries(
    primal_block: SparseMatrixAssembly,
    primal_size: int,
) -> tuple[KKTEntry, ...]:
    rows, columns = _validate_matrix("primal block", primal_block)
    if rows != columns:
        msg = f"primal block must be square, got shape {primal_block.shape}"
        raise KKTError(msg)
    if rows != primal_size:
        msg = (
            "primal block shape must match Jacobian column dimension; "
            f"got {primal_block.shape} for primal size {primal_size}"
        )
        raise KKTError(msg)
    return tuple(
        KKTEntry(
            row=entry.row,
            column=entry.column,
            value=entry.value,
            provenance=KKTProvenance(
                block=KKTBlockKind.PRIMAL,
                source=entry.provenance,
            ),
        )
        for entry in _row_major(primal_block.entries)
    )


def _jacobian_transpose_entries(
    jacobian_assembly: SparseMatrixAssembly,
    primal_size: int,
) -> tuple[KKTEntry, ...]:
    return tuple(
        KKTEntry(
            row=entry.column,
            column=primal_size + entry.row,
            value=entry.value,
            provenance=KKTProvenance(
                block=KKTBlockKind.JACOBIAN_TRANSPOSE,
                source=entry.provenance,
            ),
        )
        for entry in _row_major(jacobian_assembly.entries)
    )


def _jacobian_entries(
    jacobian_assembly: SparseMatrixAssembly,
    primal_size: int,
) -> tuple[KKTEntry, ...]:
    return tuple(
        KKTEntry(
            row=primal_size + entry.row,
            column=entry.column,
            value=entry.value,
            provenance=KKTProvenance(
                block=KKTBlockKind.JACOBIAN,
                source=entry.provenance,
            ),
        )
        for entry in _row_major(jacobian_assembly.entries)
    )


def _blocks(primal_size: int, residual_size: int) -> tuple[KKTBlock, ...]:
    total_size = primal_size + residual_size
    return (
        KKTBlock(
            kind=KKTBlockKind.PRIMAL,
            row_range=(0, primal_size),
            column_range=(0, primal_size),
        ),
        KKTBlock(
            kind=KKTBlockKind.JACOBIAN_TRANSPOSE,
            row_range=(0, primal_size),
            column_range=(primal_size, total_size),
        ),
        KKTBlock(
            kind=KKTBlockKind.JACOBIAN,
            row_range=(primal_size, total_size),
            column_range=(0, primal_size),
        ),
        KKTBlock(
            kind=KKTBlockKind.CONSTRAINT_ZERO,
            row_range=(primal_size, total_size),
            column_range=(primal_size, total_size),
        ),
    )


def _validate_kkt_entries(system: KKTSystem) -> None:
    rows, columns = system.shape
    seen: set[tuple[int, int]] = set()
    for entry in system.entries:
        if entry.row < 0 or entry.row >= rows:
            msg = f"KKT entry row {entry.row} is outside shape {system.shape}"
            raise KKTError(msg)
        if entry.column < 0 or entry.column >= columns:
            msg = f"KKT entry column {entry.column} is outside shape {system.shape}"
            raise KKTError(msg)
        coordinate = (entry.row, entry.column)
        if coordinate in seen:
            msg = f"KKT system contains duplicate coordinate {coordinate}"
            raise KKTError(msg)
        seen.add(coordinate)


def _row_major(entries: tuple[CoordinateEntry, ...]) -> tuple[CoordinateEntry, ...]:
    return tuple(sorted(entries, key=lambda entry: (entry.row, entry.column)))


def _format_block(block: KKTBlock) -> str:
    return (
        "  "
        f"{block.kind.value} "
        f"rows=[{block.row_range[0]}, {block.row_range[1]}) "
        f"cols=[{block.column_range[0]}, {block.column_range[1]})"
    )


def _format_entry(entry: KKTEntry) -> str:
    return (
        "  "
        f"row={entry.row} col={entry.column} value={entry.value:g} "
        f"provenance=[{_format_provenance(entry.provenance)}]"
    )


def _format_provenance(provenance: KKTProvenance) -> str:
    if provenance.source is None:
        return f"block={provenance.block.value}"
    return (
        f"block={provenance.block.value} "
        f"source=[{_format_assembly_provenance(provenance.source)}]"
    )


def _format_assembly_provenance(provenance: AssemblyProvenance) -> str:
    parts = [
        f"kind={provenance.kind}",
        f"row={provenance.row}",
        f"source={provenance.source_node_id}",
    ]
    if provenance.column is not None:
        parts.append(f"col={provenance.column}")
    if provenance.variable is not None:
        parts.append(
            f"variable={provenance.variable.name}@{provenance.variable.node_id}"
        )
    if provenance.derivative_node_id is not None:
        parts.append(f"derivative={provenance.derivative_node_id}")
    return " ".join(parts)
