# ROADMAP

This roadmap is complete. M0-M20 established the first end-to-end tinyNLP
execution path and the controls around it. Detailed milestone history remains in
`TASKS.md`; this file now keeps a concise completed-roadmap snapshot and leaves
space for the next major planning pass.

## Completed Roadmap Snapshot

### Project Foundation

- Created the package skeleton, CI, contributor rules, benchmark policy, and
  design notes.
- Standardized on `uv`, `pytest`, `ruff`, permissive `mypy`, and
  `pytest-benchmark`.
- Preserved the public name `tinyNLP`, distribution name `tinynlp-opt`, and
  import package `tinynlp`.

### Expression Execution

- Added a minimal scalar expression IR with variables, constants, and basic
  arithmetic.
- Added a CPU-first reference evaluator.
- Made expression execution visible through deterministic `KernelPlan` objects,
  backend protocol metadata, structural traces, and canonical examples.

### Derivatives and Structure

- Added reverse-mode scalar gradients, vector Jacobians, derivative
  verification, and deterministic derivative traces.
- Added symbolic dependency and sparsity discovery without numeric sampling.
- Kept symbolic structure separate from runtime numeric values.

### Problem Assembly and KKT Layer

- Added smooth structured constrained `Problem` objects and assembly contracts.
- Added dependency-free residual/Jacobian coordinate assembly.
- Added explicit KKT system objects, KKT block provenance, and a dense reference
  linear-solve interface.

### Solver and Sensitivity Prototypes

- Added a simple constrained residual-reduction solver prototype using the
  current identity/reference primal block convention.
- Added a scalar-parameter implicit sensitivity prototype tied to visible KKT
  metadata and explicit RHS construction.
- Documented assumptions and limits so these prototypes are not presented as
  production NLP methods.

### Scheduling and Reports

- Added execution schedule metadata for expression, assembly, KKT, solver, and
  sensitivity stages.
- Added scheduled pipeline reports that show task order, dependencies, inputs,
  outputs, cached structures, materialized values, backend choice, provenance,
  and validation status.
- Added human-facing audit examples for the chain pipeline, sensitivity path,
  and prepared residual schedule.

### Correctness Bridges and Benchmark Evidence

- Added an optional, isolated CasADi correctness bridge for supported canonical
  problems. CasADi remains optional and is not a performance baseline.
- Added the first scheduler-backed prepared residual-evaluation backend for the
  canonical chain dynamics problem.
- Added a narrow committed benchmark result summary for scheduled
  `evaluate_residuals` on `chain_dynamics_case(horizon=100)`.
- Completed a benchmark-claim audit to keep public wording limited to committed
  evidence.

## Current Boundary

The completed roadmap covers the visible reference pipeline through scheduled
residual evaluation. tinyNLP still intentionally does not implement Hessian
assembly, production nonlinear solver methods, production sensitivity workflows,
broad scheduler-driven execution, broad optimized backends, external solver
wrappers, inequalities, or bounds.

## Next Roadmap

The next major task is a fresh roadmap planning pass. Future work can decide
whether to deepen scheduled execution, add more benchmark-backed optimized
stages, expand supported operations, introduce Hessian/objective-gradient
workflows, or plan inequalities and bounds. New runtime work should wait for
that next roadmap.
