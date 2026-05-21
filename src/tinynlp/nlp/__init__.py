"""Structural NLP helpers for tinyNLP."""

from tinynlp.nlp.assembly import (
    AssemblyContract,
    JacobianAssemblyTerm,
    ResidualAssemblyTerm,
    build_assembly_contract,
    format_assembly_contract,
)
from tinynlp.nlp.problem import Problem, ResidualBlock, ResidualKind
from tinynlp.nlp.sparsity import (
    SparsityEntry,
    SparsityPattern,
    SparsityTraceEvent,
    expression_dependencies,
    format_sparsity,
    jacobian_sparsity,
)

__all__ = [
    "AssemblyContract",
    "JacobianAssemblyTerm",
    "Problem",
    "ResidualAssemblyTerm",
    "ResidualBlock",
    "ResidualKind",
    "SparsityEntry",
    "SparsityPattern",
    "SparsityTraceEvent",
    "build_assembly_contract",
    "expression_dependencies",
    "format_assembly_contract",
    "format_sparsity",
    "jacobian_sparsity",
]
