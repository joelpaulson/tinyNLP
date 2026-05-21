# BENCHMARKING

tinyNLP does not make performance claims yet. Any future performance claim must
be backed by benchmark code and committed benchmark output.

## Policy

- Benchmarks must live in `benchmarks/` or a clearly documented benchmark
  support module.
- Benchmark reports should include the command, problem definition, dependency
  versions, machine information, and relevant configuration.
- Claims in README or project documentation must link to committed benchmark
  output.
- Benchmark comparisons should describe exactly what is being measured:
  expression graph construction, derivative graph construction, sparsity
  discovery, assembly, KKT system construction, solver steps, sensitivities, or
  end-to-end runtime.
- Do not report speedups without a reproducible baseline and committed output.

## Current State

There are no benchmark results yet. `pytest-benchmark` is included as a
development dependency so benchmark work can be added when the implementation
has measurable behavior.
