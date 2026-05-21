# Examples

This directory holds canonical expression examples for the current IR/evaluator
path.

Examples should make the computational path visible. Prefer a small number of
traceable examples over broad coverage.

Current examples include an affine expression, a quadratic expression, a
residual-like expression list, and a larger banded residual system. The banded
system exercises the current expression, Jacobian, verification, and structural
sparsity path without becoming a Problem API, sparse assembly, or solver
example.
