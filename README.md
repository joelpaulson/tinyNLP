# tinyNLP

tinyNLP is an experimental, tinygrad-inspired, CPU-first nonlinear programming
execution substrate. Its public hook is simple: make the NLP execution pipeline
visible, traceable, benchmarkable, and eventually optimizable for available
hardware.

## Core Pipeline

```text
model expression
  -> IR
  -> derivative construction
  -> sparsity / structure
  -> residual / Jacobian / Hessian assembly
  -> KKT systems
  -> solver steps
  -> sensitivities
  -> benchmark reports
```

## Why tinyNLP Exists

Nonlinear programming workflows often spread important execution details across
modeling layers, automatic differentiation, sparse assembly code, solver
interfaces, and benchmark scripts. tinyNLP exists to make those stages explicit
enough to inspect, test, trace, and improve.

The goal is to provide a cleaner, more inspectable pipeline that creates room
for hardware-aware optimization and benchmark-backed comparisons. The initial
implementation path starts with smooth structured constrained problems, with
inequalities, bounds, additional solver backends, code generation, and
hardware-specific execution planned later.

## Current Status

tinyNLP is experimental and pre-alpha. M1-M5 are complete: the repository now
includes a minimal scalar expression IR, CPU-first reference evaluator,
deterministic structural traces, canonical expression examples, a deterministic
KernelPlan, a small backend protocol, a registered Python reference backend, and
pytest-benchmark smoke sources.

It does not yet implement autodiff, vector Jacobians, sparsity discovery,
problem APIs, residual/Jacobian/Hessian assembly, KKT systems, solver steps,
sensitivities, bridges, optimized backends, inequalities, or bounds.

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

The public project name is `tinyNLP`. The distribution name is `tinynlp-opt`,
and the import package is `tinynlp`.

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

Run the standard milestone checks:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If `pyproject.toml` changes, also run:

```sh
uv sync
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

## Benchmark Policy

Benchmarks are correctness checks plus timing, not demos. Benchmark results must
identify what stage is measured, validate outputs before timing claims, and
include the benchmark source, command, environment metadata, and committed
result summary.

The repository has benchmark smoke sources, but no committed benchmark result
summaries and no performance claims.

No speed or performance claims should appear in README or project documentation
without committed benchmark evidence.

## Repository Layout

```text
src/tinynlp/
  ir/          minimal scalar expression graph and IR structures
  autodiff/    future derivative construction
  nlp/         future problem API and assembly contracts
  solvers/     future KKT and solver-step workflows
  backends/    KernelPlan, backend protocol, registry, and Python reference backend
  bridges/     future import/export adapters
  profiling/   deterministic structural trace helpers

docs/          design notes and architecture sketches
examples/      canonical expression examples for the current IR/evaluator path
benchmarks/    benchmark smoke sources and future committed summaries
tests/         pytest suite
```
