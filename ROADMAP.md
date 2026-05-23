# ROADMAP

This roadmap is complete. M0-M20 established the first end-to-end tinyNLP
execution path and the controls around it. Detailed milestone history remains in
`TASKS.md`; this file now keeps a concise completed-roadmap snapshot and records
the next flagship planning pass.

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

## Next Roadmap: Flagship Usability and Speed Track

The next roadmap should turn the existing visible pipeline into a compelling
flagship workflow. The aim is one chain-structured nonlinear example that is
easy to model, inspect, solve, validate, and benchmark narrowly. This track
prioritizes lightweight usability plus inspectability before expanding the math
operation set.

### F0: Flagship Story and Acceptance Criteria

- Define the flagship problem, audience, and success criteria.
- Keep the public value proposition narrow: a lightweight, inspectable
  chain-structured nonlinear workflow.
- Separate correctness evidence, usability evidence, and speed evidence.

### F1: Ergonomic Modeling Layer

- Add small helpers that make structured examples easier to write without
  hiding the IR, problem, schedule, or assembly path.
- Prefer variable arrays, named residual blocks, explicit values, and concise
  example construction over broad modeling syntax.
- Do not add new operations unless the flagship workflow proves they are needed.

### F2: Transparent Least-Squares / Gauss-Newton Reference Prototype

- Add a reference residual/Jacobian-based least-squares path for the flagship
  workflow.
- Report objective metric, residual norm, step norm, damping, and solve status.
- Keep this a transparent reference prototype, not an IPOPT-style production
  solver.

### F3: Scheduler-Backed Residual + Jacobian Execution

- Extend scheduler-backed prepared execution from residual-only to a
  residual-plus-Jacobian scheduled stage.
- Validate outputs against the reference assembly path before timing.
- Keep optimized work dependency-free and attached to schedule metadata.

### F4: Flagship Benchmark and Optional CasADi Correctness Comparison

- Add benchmark sources and committed result summaries only for named scheduled
  stages.
- Use CasADi as an optional correctness comparison, not a performance baseline.
- Keep claims limited to committed evidence for the flagship problem and stage.

### F5: Polished User-Facing Flagship Example and README Audit

- Add a single "start here" example that prints or exposes the model, schedule,
  residual/Jacobian assembly, solver trace, optional CasADi correctness report,
  and benchmark command.
- Update README only after the example and evidence exist.
- Preserve narrow wording around speed, solver scope, and supported problem
  classes.

New runtime work should proceed one F milestone at a time. Each milestone must
leave tests passing, reports deterministic, docs honest, and any performance
claim tied to committed benchmark evidence.
