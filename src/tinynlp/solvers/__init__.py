"""KKT and reference linear-solve helpers for tinyNLP."""

from tinynlp.solvers.kkt import (
    KKTBlock,
    KKTBlockKind,
    KKTEntry,
    KKTError,
    KKTProvenance,
    KKTSystem,
    build_kkt_system,
    format_kkt_system,
    kkt_to_dense,
)
from tinynlp.solvers.linear import (
    DenseReferenceLinearSolver,
    LinearSolveError,
    LinearSolver,
    LinearSolveResult,
)

__all__ = [
    "DenseReferenceLinearSolver",
    "KKTBlock",
    "KKTBlockKind",
    "KKTEntry",
    "KKTError",
    "KKTProvenance",
    "KKTSystem",
    "LinearSolveError",
    "LinearSolveResult",
    "LinearSolver",
    "build_kkt_system",
    "format_kkt_system",
    "kkt_to_dense",
]
