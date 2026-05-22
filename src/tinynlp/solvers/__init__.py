"""KKT, reference linear-solve, and prototype solver helpers for tinyNLP."""

from tinynlp.solvers.constrained import (
    IterationRecord,
    SolverError,
    SolverResult,
    SolverStatus,
    VariableValue,
    format_solver_trace,
    solve_constraints,
)
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
from tinynlp.solvers.sensitivity import (
    SensitivityEntry,
    SensitivityError,
    SensitivityResult,
    SensitivityRhsEntry,
    SensitivityTrace,
    format_sensitivity,
    implicit_sensitivity,
)

__all__ = [
    "DenseReferenceLinearSolver",
    "IterationRecord",
    "KKTBlock",
    "KKTBlockKind",
    "KKTEntry",
    "KKTError",
    "KKTProvenance",
    "KKTSystem",
    "LinearSolveError",
    "LinearSolveResult",
    "LinearSolver",
    "SensitivityEntry",
    "SensitivityError",
    "SensitivityResult",
    "SensitivityRhsEntry",
    "SensitivityTrace",
    "SolverError",
    "SolverResult",
    "SolverStatus",
    "VariableValue",
    "build_kkt_system",
    "format_sensitivity",
    "format_solver_trace",
    "implicit_sensitivity",
    "format_kkt_system",
    "kkt_to_dense",
    "solve_constraints",
]
