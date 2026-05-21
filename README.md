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

tinyNLP is experimental and pre-implementation. The repository currently
contains the package skeleton, documentation/control packet, CI, and validation
tooling. It does not yet implement expression IR, autodiff, sparsity discovery,
KKT assembly, solvers, bridges, or benchmarks.

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

No speed or performance claims should appear in README or project documentation
without committed benchmark evidence.

## Repository Layout

```text
src/tinynlp/
  ir/          future expression graph and IR structures
  autodiff/    future derivative construction
  nlp/         future problem API and assembly contracts
  solvers/     future KKT and solver-step workflows
  backends/    future CPU reference and optimized execution backends
  bridges/     future import/export adapters
  profiling/   future tracing, timing, and benchmark helpers

docs/          design notes and architecture sketches
examples/      future canonical examples
benchmarks/    future benchmark sources and committed summaries
tests/         pytest suite
```
