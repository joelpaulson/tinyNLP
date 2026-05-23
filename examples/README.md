# Examples

This directory holds canonical examples for the current visible pipeline.

Examples should make the computational path visible. Prefer a small number of
traceable examples over broad coverage.

Current examples include an affine expression, a quadratic expression, a
residual-like expression list, a larger banded residual system, an explicit
chain dynamics problem, and a helper-built flagship chain model. The chain
dynamics problem is the first problem/assembly/KKT smoke case for the pre-solver
pipeline and the canonical smoke case for the simple constrained solver
prototype. The flagship chain model shows the same style of workflow with
ergonomic variable-array, value-map, sum, residual-block, and `Problem` helpers.
The flagship least-squares trace shows a transparent residual Gauss-Newton
reference path with normal-equation metadata. These are not production nonlinear
solver examples. The parameter sensitivity problem is the first scalar-parameter
implicit sensitivity smoke example and is not a production differentiable
optimization workflow. The sensitivity example treats the parameter as a
symbolic variable and keeps solve variables explicit.

The scheduled pipeline report examples show how to inspect the current
frontend -> schedule -> backend metadata path for chain assembly/KKT work, the
parameter sensitivity workflow, prepared residual evaluation, and prepared
residual+Jacobian evaluation. They are inspection and audit examples, not
benchmark reports, solver comparisons, or bridge examples.
`prepared_residual_schedule_report.py` is the easiest way to see the M19
scheduler-backed residual path by eye, while
`prepared_residual_jacobian_schedule_report.py` shows the F3 flagship
residual+Jacobian path.

`casadi_correctness_report.py` is an optional correctness-only bridge example.
It compares the flagship chain residual/Jacobian assembly path against CasADi
when CasADi is installed, and otherwise prints a skip-safe install hint. It is
not a solver wrapper, speed comparison, performance baseline, or benchmark
report.

Examples that are meant to run as scripts use `_bootstrap.py` so they can import
the local `src/tinynlp` package directly from a source checkout.

Run the prepared residual schedule inspection example with:

```sh
uv run python examples/prepared_residual_schedule_report.py
```

Run the prepared residual+Jacobian schedule inspection example with:

```sh
uv run python examples/prepared_residual_jacobian_schedule_report.py
```

Run the helper-built flagship chain model with:

```sh
uv run python examples/flagship_chain_modeling.py
```

Run the transparent flagship residual least-squares trace with:

```sh
uv run python examples/flagship_least_squares_trace.py
```

Run the full assembly/Jacobian/KKT schedule inspection example with:

```sh
uv run python examples/scheduled_pipeline_report.py
```

Run the optional CasADi correctness report with:

```sh
uv sync --extra casadi
uv run python examples/casadi_correctness_report.py
```
