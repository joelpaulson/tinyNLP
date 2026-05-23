# BENCHMARKING

tinyNLP does not make performance claims yet. Benchmarks are correctness checks
plus timing, not demos.

## Policy

- Benchmarks must validate outputs before any timing result is used in a claim.
- Benchmarks must identify exactly what is measured.
- Benchmarks must live in `benchmarks/` or a documented benchmark support
  module.
- Benchmark reports must include the benchmark source, command, environment
  metadata, problem definition, dependency versions, and committed result
  summary.
- README speed claims require benchmark source, command, environment metadata,
  and committed result summary.
- Do not report speedups without a reproducible baseline.
- Optional correctness bridges such as CasADi are not benchmark result
  summaries and must not be described as performance baselines.

## Measurement Targets

Benchmarks should name the pipeline stage they measure:

- Expression and residual evaluation.
- Derivative evaluation.
- Sparsity discovery.
- Sparse assembly.
- KKT assembly.
- Sensitivities.
- End-to-end solve traces.

## Initial Benchmark Families

- Expression/residual evaluation.
- Derivative evaluation.
- Sparsity discovery.
- Sparse assembly.
- KKT assembly.
- Sensitivities.
- End-to-end solve traces.

## Current State

Benchmark smoke sources now exist for expression/residual evaluation through the
backend protocol and registered Python reference backend, and for chain
dynamics residual assembly, sparse Jacobian assembly, and KKT construction.
They validate expected outputs before timing and are used as test/collection
scaffolding.

There are no committed benchmark result summaries yet, and tinyNLP still makes
no speed or performance claims.

The optional CasADi bridge is a correctness comparison path for supported
canonical problems only. It does not call external solvers, run code generation,
or support speed claims.

## First Optimized Target Plan

The first optimized backend target is scheduled chain residual evaluation. This
is deliberately narrower than solver speed, Jacobian speed, KKT speed,
sensitivity speed, or package-wide performance.

- Future benchmark source: `benchmarks/test_scheduler_backend_benchmark.py`.
- Scheduled stage: `evaluate_residuals` for `chain_dynamics_case(horizon=N)`.
- Reference baseline: existing residual assembly using a cached
  `AssemblyContract` and the registered Python backend.
- Optimized path: a scheduler-backed prepared `KernelPlan` CPU residual
  evaluator, with no new dependencies, GPU support, or code generation.
- Correctness gate: optimized residual values must match the reference residual
  values for the same scheduled task before timing is used.
- Required command:

  ```sh
  uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json <result-json>
  ```

Committed result summaries for this target must include the git commit, command,
Python version, OS and CPU metadata, dependency versions, problem horizon,
scheduled stage, validation result, baseline measurements, and optimized
measurements.

If the result succeeds before the benchmark-claim audit, the only acceptable
claim wording is a narrow result-summary statement:

> On `<environment>`, for scheduled `evaluate_residuals` on
> `chain_dynamics_case(horizon=N)`, the prepared KernelPlan backend measured
> `<result>` against the Python reference backend using `<benchmark source>` at
> `<commit>`. This is a narrow scheduled-stage result, not a solver, Jacobian,
> KKT, sensitivity, or package-wide speed claim.
