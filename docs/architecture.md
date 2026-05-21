# Architecture

tinyNLP is organized around an explicit NLP execution pipeline:

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

## Package Areas

- `ir`: future expression graph and IR structures.
- `autodiff`: future derivative construction and verification.
- `nlp`: future problem API and assembly contracts.
- `solvers`: future KKT and solver-step workflows.
- `backends`: future CPU reference and optimized execution backends.
- `bridges`: future import/export adapters.
- `profiling`: future tracing, timing, and benchmark helpers.

## Current Boundary

The repository currently contains structure, documentation, CI, and validation
tooling only. It intentionally does not implement expression IR, autodiff,
sparsity discovery, assembly, KKT systems, solvers, bridges, or benchmarks yet.
