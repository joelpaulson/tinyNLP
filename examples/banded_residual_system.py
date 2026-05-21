"""Banded residual-system example for the current tinyNLP pipeline."""

from tinynlp.autodiff import evaluate_jacobian, jacobian
from tinynlp.backends import evaluate
from tinynlp.ir import Expr, Graph
from tinynlp.nlp import SparsityPattern, jacobian_sparsity


def banded_residual_expressions(size: int = 6) -> list[Expr]:
    """Build a banded residual expression list.

    The residuals are:

        r_i = ((x_i * x_i) + x_{i+1}) / (x_{i+1} + 1) - 1
    """

    _validate_size(size)
    graph = Graph()
    variables = [graph.variable(f"x{index}") for index in range(size)]
    return [
        ((variables[index] * variables[index]) + variables[index + 1])
        / (variables[index + 1] + 1)
        - 1
        for index in range(size - 1)
    ]


def banded_values(size: int = 6, value: float = 2.0) -> dict[str, float]:
    """Return a uniform value binding for the banded system."""

    _validate_size(size)
    return {f"x{index}": float(value) for index in range(size)}


def evaluate_banded_residuals(size: int = 6, value: float = 2.0) -> list[float]:
    """Evaluate the banded residual system at uniform values."""

    values = banded_values(size=size, value=value)
    return [evaluate(expr, values) for expr in banded_residual_expressions(size)]


def evaluate_banded_jacobian(size: int = 6, value: float = 2.0) -> list[list[float]]:
    """Evaluate the dense Jacobian for the banded residual system."""

    outputs = banded_residual_expressions(size)
    return evaluate_jacobian(jacobian(outputs), banded_values(size=size, value=value))


def banded_sparsity(size: int = 6) -> SparsityPattern:
    """Return the structural Jacobian sparsity for the banded system."""

    return jacobian_sparsity(banded_residual_expressions(size))


def _validate_size(size: int) -> None:
    if size < 2:
        msg = "banded residual system size must be at least 2"
        raise ValueError(msg)
