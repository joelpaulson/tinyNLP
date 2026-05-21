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

- `ir`: minimal expression graph and scalar operation structures.
- `autodiff`: future derivative construction and verification.
- `nlp`: future problem API and assembly contracts.
- `solvers`: future KKT and solver-step workflows.
- `backends`: KernelPlan, backend protocol, registry, and Python reference
  backend.
- `bridges`: future import/export adapters.
- `profiling`: deterministic structural trace helpers.

## Current Boundary

The current boundary is scalar expression construction/evaluation and plan
visibility only. The repository intentionally does not implement autodiff,
vector Jacobians, sparsity discovery, problem APIs, residual/Jacobian/Hessian
assembly, KKT systems, solvers, sensitivities, bridges, optimized backends,
inequalities, or bounds.
