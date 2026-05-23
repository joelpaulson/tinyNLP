# Benchmarks

This directory holds benchmark sources for tinyNLP. There are no committed
benchmark result summaries yet, and this repository does not make speed claims.

## Current Scaffold

- `test_reference_evaluator_benchmark.py` measures expression/residual
  evaluation through the backend protocol and registered Python reference
  backend.
- `test_chain_pipeline_benchmark.py` measures chain dynamics residual assembly,
  sparse Jacobian assembly, and KKT construction for correctness-guarded smoke
  cases.
- Benchmark sources validate expected outputs, dimensions, or entry counts
  before timing.
- The only baseline in this scaffold is the expected numeric output for the
  canonical examples, not another package, method, or optimized backend.

## Running Benchmarks

Use pytest-benchmark through uv:

```sh
uv run pytest benchmarks
```

## Result Summary Requirements

Do not add benchmark result summaries until there is a milestone that asks for
them. Future committed summaries must include:

- Benchmark source and command.
- Measured pipeline stage.
- Problem definition and expected output.
- Dependency versions.
- Machine and environment metadata.
- Result summary tied to committed code.

Do not make README performance claims without benchmark source, command,
environment metadata, and committed result summary.

## Planned First Optimized Benchmark

The first optimized backend benchmark should be narrow and scheduler-backed:

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

The benchmark must validate residual values before timing. Any committed result
summary must state that the result is limited to this scheduled stage and is not
a solver, Jacobian, KKT, sensitivity, or package-wide performance claim.
