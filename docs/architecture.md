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
- `nlp`: ergonomic modeling helpers, structural sparsity, problem APIs, and
  residual/Jacobian assembly.
- `solvers`: explicit KKT systems, reference linear-solve workflows, and the
  simple constrained residual-reduction solver and implicit sensitivity
  prototypes.
- `schedule`: execution schedule metadata, deterministic reports, and scheduled
  residual-evaluation helpers.
- `backends`: KernelPlan, backend protocol, registry, Python reference backend,
  and prepared KernelPlan backend.
- `bridges`: optional correctness bridges, currently CasADi for supported
  canonical problems, plus future import/export adapters.
- `profiling`: deterministic structural trace helpers.

## Scheduler Layer

The scheduler layer distinguishes four responsibilities:

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

`KernelPlan` remains the expression-level execution plan for one scalar
expression graph. `ExecutionSchedule` is broader metadata over NLP pipeline
tasks; it may cache or reference `KernelPlan` summaries when a scheduled task
uses expression evaluation, but it does not replace the backend protocol. The
current optimized path is a small helper around a scheduled residual-evaluation
task, not a general schedule executor.

Scheduled reports are the current audit surface for the tinyNLP analog of
frontend -> scheduler -> backend execution. They show the frontend structures
that define the work, the scheduled task order and dependencies, and the
reference or prepared backend, or linear-solve component, selected for each
task. They are inspection artifacts, not a general optimized runtime log.

The first optimized backend target is scheduled residual evaluation for the
canonical chain dynamics problem. That target uses the scheduler to identify the
`evaluate_residuals` stage and uses prepared residual `KernelPlan`s as the
backend-facing work unit. It is not a solver, Jacobian, KKT, sensitivity, or
package-wide performance target.

## Current Boundary

The current boundary is scalar expression construction/evaluation, small
modeling helpers for structured examples, plan visibility, derivative
construction and verification, symbolic sparsity discovery, residual/Jacobian
assembly, explicit KKT system construction, and a dense reference linear-solve
interface, plus a simple constrained residual-reduction solver prototype and
scalar-parameter implicit sensitivity prototype. The repository also has
execution schedule metadata for expression, assembly, KKT, and sensitivity
stages, deterministic scheduled pipeline reports and audit examples, and one
scheduler-backed prepared residual-evaluation path for the canonical chain
dynamics problem. It intentionally does not implement Hessian assembly,
production nonlinear solver methods, production sensitivity workflows, broad
scheduler-driven execution, external solver wrappers, broad optimized backends,
inequalities, or bounds.

The optional CasADi bridge is isolated under `tinynlp.bridges` and is a
correctness comparison path only. CasADi symbols and arrays do not enter core IR,
autodiff, NLP assembly, solver, schedule, or backend modules.

## Inspection Commands

The most useful human-facing checks for the completed roadmap are:

```sh
uv run python examples/prepared_residual_schedule_report.py
uv run python examples/flagship_chain_modeling.py
uv run python examples/scheduled_pipeline_report.py
uv run python examples/casadi_correctness_report.py
```

The first command shows the prepared scheduler-backed residual path and its
reference validation. The flagship command shows the helper-built chain model.
The scheduled pipeline command shows assembly/Jacobian/KKT schedule metadata.
The CasADi command is an optional correctness comparison; it prints a skip-safe
message if CasADi is not installed.
