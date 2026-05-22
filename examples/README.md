# Examples

This directory holds canonical examples for the current visible pipeline.

Examples should make the computational path visible. Prefer a small number of
traceable examples over broad coverage.

Current examples include an affine expression, a quadratic expression, a
residual-like expression list, a larger banded residual system, and a chain
dynamics problem. The chain dynamics problem is the first problem/assembly/KKT
smoke case for the pre-solver pipeline and the canonical smoke case for the
simple constrained solver prototype. It is not a production nonlinear solver
example.

Examples that are meant to run as scripts use `_bootstrap.py` so they can import
the local `src/tinynlp` package directly from a source checkout.
