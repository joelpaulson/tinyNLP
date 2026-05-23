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
  -> execution scheduling
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

tinyNLP is experimental and pre-alpha. M1-M20 and F1-F3 are complete: the
repository now includes a minimal scalar expression IR, CPU-first reference
evaluator, deterministic structural traces, canonical expression examples, a
deterministic KernelPlan, a small backend protocol, a registered Python
reference backend, reverse-mode scalar autodiff, vector Jacobian construction
and verification, structural sparsity discovery, problem containers,
residual/Jacobian assembly contracts, a small ergonomic modeling-helper layer,
dependency-free sparse coordinate assembly, explicit KKT system objects, a dense
reference linear-solve interface, a simple constrained residual-reduction solver
prototype, a transparent residual least-squares/Gauss-Newton reference
prototype, a scalar-parameter implicit sensitivity prototype, execution
schedule metadata, scheduled pipeline reports, audit examples for assembly/KKT
and sensitivity paths, an optional CasADi correctness bridge, a prepared
scheduler-backed residual-evaluation backend, a prepared scheduler-backed
residual+Jacobian evaluation path, a narrow scheduled-stage benchmark result
summary, pytest-benchmark smoke sources, and a benchmark-claim audit.

It does not yet implement Hessian assembly, production nonlinear solver methods,
production sensitivity workflows, broad scheduler-driven execution, external
solver wrappers, broad optimized backends, inequalities, or bounds. The current
committed benchmark result is limited to scheduled residual evaluation for the
canonical chain dynamics problem.

## Installation From Source

Install the project from a local checkout with uv:

```sh
git clone https://github.com/joelpaulson/tinyNLP.git
cd tinyNLP
uv sync
```

Install the optional CasADi correctness bridge only when needed:

```sh
uv sync --extra casadi
```

Import the package:

```sh
uv run python -c "import tinynlp; print(tinynlp.__version__)"
```

The public project name is `tinyNLP`. The distribution name is `tinynlp-opt`,
and the import package is `tinynlp`.

## Try The Current Pipeline

Inspect the prepared scheduler-backed residual path:

```sh
uv run python examples/prepared_residual_schedule_report.py
```

Inspect the prepared scheduler-backed residual+Jacobian path:

```sh
uv run python examples/prepared_residual_jacobian_schedule_report.py
```

Inspect the helper-built flagship chain model:

```sh
uv run python examples/flagship_chain_modeling.py
```

Inspect the transparent residual least-squares trace:

```sh
uv run python examples/flagship_least_squares_trace.py
```

Inspect the assembly/Jacobian/KKT schedule:

```sh
uv run python examples/scheduled_pipeline_report.py
```

Inspect the optional CasADi correctness bridge when CasADi is installed:

```sh
uv sync --extra casadi
uv run python examples/casadi_correctness_report.py
```

Run the narrow scheduler-backed benchmark source:

```sh
uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json /private/tmp/tinynlp-scheduled-residual-benchmark.json
```

That benchmark is limited to scheduled `evaluate_residuals` on the canonical
chain dynamics problem and validates outputs before timing. It is not a solver,
Jacobian, KKT, sensitivity, CasADi, or package-wide speed comparison.

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

A narrow committed benchmark result summary exists for scheduled
`evaluate_residuals` on `chain_dynamics_case(horizon=100)`. It is documented in
`benchmarks/results/scheduled_residual_evaluation.md` and is not a solver,
Jacobian, KKT, sensitivity, or package-wide speed claim.

No speed or performance claims should appear in README or project documentation
without committed benchmark evidence.

## Repository Layout

```text
src/tinynlp/
  ir/          minimal scalar expression graph and IR structures
  autodiff/    reverse-mode scalar gradients, Jacobians, and verification
  nlp/         modeling helpers, structural sparsity, problem APIs, and assembly
  solvers/     KKT systems, reference solves, solver prototypes, sensitivities
  schedule/    execution schedule metadata, reports, and scheduled execution
  backends/    KernelPlan, backend protocol, registry, reference and prepared backends
  bridges/     optional CasADi correctness bridge and future adapters
  profiling/   deterministic structural trace helpers

docs/          design notes and architecture sketches
examples/      canonical examples for the current visible pipeline
benchmarks/    benchmark smoke sources and future committed summaries
tests/         pytest suite
```
