# BENCHMARKING

tinyNLP does not make broad performance claims. Benchmarks are correctness
checks plus timing, not demos.

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

A narrow committed benchmark result summary now exists for scheduled
`evaluate_residuals` on `chain_dynamics_case(horizon=100)`:
`benchmarks/results/scheduled_residual_evaluation.md`. It is limited to that
scheduled stage and must not be described as solver, Jacobian, KKT, sensitivity,
or package-wide performance.

The optional CasADi bridge is a correctness comparison path for supported
canonical problems only. It does not call external solvers, run code generation,
or support speed claims.

## Human-Facing Checks

Use these commands to inspect the current benchmark-adjacent paths without
turning them into broad claims:

```sh
uv run python examples/prepared_residual_schedule_report.py
uv run python examples/scheduled_pipeline_report.py
uv run python examples/casadi_correctness_report.py
```

The CasADi report is skip-safe when CasADi is not installed. Install the
optional extra only when you want that correctness comparison:

```sh
uv sync --extra casadi
```

## First Optimized Target

The first optimized backend target is scheduled chain residual evaluation. The
M19 implementation is deliberately narrower than solver speed, Jacobian speed,
KKT speed, sensitivity speed, or package-wide performance.

- Benchmark source: `benchmarks/test_scheduler_backend_benchmark.py`.
- Scheduled stage: `evaluate_residuals` for `chain_dynamics_case(horizon=N)`.
- Reference baseline: existing residual assembly using a cached
  `AssemblyContract` and the registered Python backend.
- Optimized path: a scheduler-backed prepared `KernelPlan` CPU residual
  evaluator, with no new dependencies, GPU support, or code generation.
- Correctness gate: optimized residual values must match the reference residual
  values for the same scheduled task before timing is used.
- Result summary: `benchmarks/results/scheduled_residual_evaluation.md`.
- Required command shape:

  ```sh
  uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json <result-json>
  ```

Committed result summaries for this target must include the command, Python
version, OS and CPU metadata, dependency versions, problem horizon, scheduled
stage, validation result, baseline measurements, and optimized measurements.

The M20 benchmark-claim audit accepts only narrow scheduled-stage wording for
this result:

> On `<environment>`, for scheduled `evaluate_residuals` on
> `chain_dynamics_case(horizon=N)`, the prepared KernelPlan backend measured
> `<result>` against the Python reference backend using `<benchmark source>` at
> `<commit>`. This is a narrow scheduled-stage result, not a solver, Jacobian,
> KKT, sensitivity, or package-wide speed claim.

## Flagship Benchmark Rules

The next roadmap should use the flagship chain-structured workflow to make
benchmark evidence easier to understand. Keep these rules:

- Validate correctness before timing every benchmark.
- Benchmark named scheduled stages before making broader workflow claims.
- Measure residual and Jacobian scheduled stages before claiming solver speed.
- Treat CasADi as an optional correctness comparison only, not a performance
  baseline.
- Keep result summaries narrow, stage-specific, and tied to one command,
  environment, problem size, reference baseline, optimized path, and validation
  result.
- Do not add README speed wording until the exact claim appears in a committed
  benchmark result summary.

Flagship evidence should stay separated into three categories:

- Correctness: tests and optional CasADi comparisons.
- Usability: runnable examples and readable reports.
- Speed: committed benchmark summaries for named scheduled stages.
