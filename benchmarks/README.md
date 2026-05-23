# Benchmarks

This directory holds benchmark sources for tinyNLP. A narrow committed result
summary exists for scheduled residual evaluation, and this repository does not
make broad speed claims.

## Current Scaffold

- `test_reference_evaluator_benchmark.py` measures expression/residual
  evaluation through the backend protocol and registered Python reference
  backend.
- `test_chain_pipeline_benchmark.py` measures chain dynamics residual assembly,
  sparse Jacobian assembly, and KKT construction for correctness-guarded smoke
  cases.
- Benchmark sources validate expected outputs, dimensions, or entry counts
  before timing.
- Most scaffold baselines are expected numeric outputs for canonical examples.
  The M19 scheduled residual benchmark also compares against the Python
  reference backend for the same scheduled stage.

## Running Benchmarks

Use pytest-benchmark through uv:

```sh
uv run pytest benchmarks
```

## Result Summary Requirements

Committed benchmark result summaries must be tied to milestones that ask for
them. Summaries must include:

- Benchmark source and command.
- Measured pipeline stage.
- Problem definition and expected output.
- Dependency versions.
- Machine and environment metadata.
- Result summary tied to committed code.

Do not make README performance claims without benchmark source, command,
environment metadata, and committed result summary.

## First Optimized Benchmark

The first optimized backend benchmark is narrow and scheduler-backed:

- Source: `benchmarks/test_scheduler_backend_benchmark.py`.
- Stage: scheduled `evaluate_residuals`.
- Problem: `chain_dynamics_case(horizon=N)`.
- Baseline: existing residual assembly with a cached `AssemblyContract` and the
  registered Python backend.
- Optimized path: prepared `KernelPlan` CPU residual execution attached to the
  scheduled residual-evaluation task.
- Command:

  ```sh
  uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json <result-json>
  ```
- Result summary: `benchmarks/results/scheduled_residual_evaluation.md`.

The benchmark must validate residual values before timing. Any committed result
summary must state that the result is limited to this scheduled stage and is not
a solver, Jacobian, KKT, sensitivity, or package-wide performance claim.
