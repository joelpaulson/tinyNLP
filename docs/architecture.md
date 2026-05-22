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
- `autodiff`: reverse-mode scalar gradients, Jacobians, and verification.
- `nlp`: structural sparsity, problem APIs, and residual/Jacobian assembly.
- `solvers`: explicit KKT systems, reference linear-solve workflows, and the
  simple constrained residual-reduction solver prototype.
- `backends`: KernelPlan, backend protocol, registry, and Python reference
  backend.
- `bridges`: future import/export adapters.
- `profiling`: deterministic structural trace helpers.

## Current Boundary

The current boundary is scalar expression construction/evaluation, plan
visibility, derivative construction and verification, symbolic sparsity
discovery, residual/Jacobian assembly, explicit KKT system construction, and a
dense reference linear-solve interface, plus a simple constrained
residual-reduction solver prototype. The repository intentionally does not
implement Hessian assembly, production nonlinear solver methods, sensitivities,
bridges, optimized backends, inequalities, or bounds.
