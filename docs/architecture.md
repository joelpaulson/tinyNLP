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
  -> execution scheduling
  -> benchmark reports
```

## Package Areas

- `ir`: minimal expression graph and scalar operation structures.
- `autodiff`: reverse-mode scalar gradients, Jacobians, and verification.
- `nlp`: structural sparsity, problem APIs, and residual/Jacobian assembly.
- `solvers`: explicit KKT systems, reference linear-solve workflows, and the
  simple constrained residual-reduction solver and implicit sensitivity
  prototypes.
- `backends`: KernelPlan, backend protocol, registry, and Python reference
  backend.
- `bridges`: future import/export adapters.
- `profiling`: deterministic structural trace helpers.

## Planned Scheduler Layer

The next architecture layer should distinguish four responsibilities:

- Frontend/model representation: expression IR, `Problem` objects, assembly
  contracts, KKT systems, solver workflows, and sensitivity workflows.
- Scheduling: deterministic grouping of pipeline work into scheduled tasks with
  explicit dependencies.
- Backend execution: the reference Python backend first, with optimized backends
  added later through scheduled tasks rather than ad hoc fast paths.
- Reports: printable schedules showing tasks, inputs, outputs, cached
  structures, materialized values, backend choice, provenance, and validation
  status.

Initial scheduled stages should cover expression evaluation, residual
evaluation, Jacobian evaluation, sparse coordinate assembly, KKT assembly,
solver iteration steps, sensitivity RHS construction, and sensitivity solves.

## Current Boundary

The current boundary is scalar expression construction/evaluation, plan
visibility, derivative construction and verification, symbolic sparsity
discovery, residual/Jacobian assembly, explicit KKT system construction, and a
dense reference linear-solve interface, plus a simple constrained
residual-reduction solver prototype and scalar-parameter implicit sensitivity
prototype. The repository intentionally does not implement Hessian assembly,
production nonlinear solver methods, production sensitivity workflows, bridges,
runtime execution schedules, optimized backends, inequalities, or bounds.
