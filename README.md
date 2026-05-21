# tinyNLP

tinyNLP is an experimental Python package for building a tinygrad-inspired,
CPU-first nonlinear programming substrate.

The project makes the computational path in nonlinear programming explicit and
optimizable:

```text
expression graph
  -> derivative graph
  -> sparsity
  -> residual / Jacobian / Hessian assembly
  -> KKT systems
  -> solver steps
  -> sensitivities
  -> benchmark reports
```

## Why tinyNLP Exists

Nonlinear programming systems often hide important execution details behind
large modeling layers, automatic differentiation systems, sparse assembly code,
and solver interfaces. tinyNLP exists to make that pipeline inspectable end to
end.

The goal is to provide a cleaner, more transparent substrate that creates room
for hardware-aware optimization and benchmark-backed comparisons. The initial
implementation may start with smooth structured equality-constrained problems,
but the public direction is a broader NLP pipeline with planned growth toward
inequalities, bounds, additional solver backends, sensitivities, and benchmark
reporting.

## Current Status

tinyNLP is experimental and pre-implementation. The repository currently defines
the package skeleton, contributor rules, documentation placeholders, and
validation tooling. It does not yet implement an expression IR, derivative
engine, solver, or benchmark suite.

No performance claims are made yet. Performance claims must be backed by
committed benchmark output.

## Installation From Source

Install the project from a local checkout with uv:

```sh
git clone https://github.com/joelpaulson/tinyNLP.git
cd tinyNLP
uv sync
```

Import the package:

```sh
uv run python -c "import tinynlp; print(tinynlp.__version__)"
```

## Development Commands

Run the test suite:

```sh
uv run pytest
```

Run lint checks:

```sh
uv run ruff check .
```

Check formatting:

```sh
uv run ruff format --check .
```

Run the full milestone validation set:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Repository Layout

```text
src/tinynlp/
  ir/          expression graph representation
  autodiff/    derivative graph construction
  nlp/         problem structure and assembly contracts
  solvers/     solver steps and KKT workflows
  backends/    numeric kernels and execution backends
  bridges/     import/export bridges to other formats
  profiling/   timing, tracing, and benchmark helpers
```

These directories are placeholders for the planned architecture. They should
stay small, inspectable, and traceable as features are added.
