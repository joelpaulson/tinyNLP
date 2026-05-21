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

There are no benchmark results yet. `pytest-benchmark` is included as a
development dependency so benchmark work can be added when the implementation
has measurable behavior.
