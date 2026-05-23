# DECISIONS

This file records architecture decisions for tinyNLP. Entries follow a compact
ADR style: context, decision, and consequences.

## ADR 0001: CPU-First Execution

- Status: accepted
- Date: 2026-05-21

### Context

tinyNLP needs a reference execution path that is easy to inspect before it adds
hardware-specific optimization.

### Decision

Start with CPU-first reference backends. Hardware-specific code generation and
accelerated execution are later roadmap items.

### Consequences

- Early correctness and traceability take priority over throughput.
- Optimized backends must match reference behavior before they can support
  performance claims.
- Hardware-aware work must be benchmark-backed.

## ADR 0002: Inspectable Core

- Status: accepted
- Date: 2026-05-21

### Context

The project exists to expose the nonlinear-programming execution pipeline, so
the core cannot become a black box.

### Decision

Keep the core small and inspectable. Add abstractions only when they make the
pipeline clearer, remove meaningful duplication, or stabilize a proven
interface.

### Consequences

- Public APIs should stay narrow until examples and tests prove them.
- Traceability is part of feature acceptance, not a later polish step.
- Large wrappers and broad integrations wait for explicit roadmap phases.

## ADR 0003: Separate Symbolic Structure From Numeric Values

- Status: accepted
- Date: 2026-05-21

### Context

NLP pipelines reuse symbolic and sparse structure across many numeric
evaluations.

### Decision

Represent symbolic structure separately from numeric values.

### Consequences

- Sparsity, derivative structure, and assembly plans can be inspected without
  requiring all numeric values.
- Tests should cover structural stability across value changes.
- APIs should avoid mixing symbolic construction with numeric evaluation unless
  a milestone explicitly requires it.

## ADR 0004: uv, Ruff, pytest, pytest-benchmark, and Permissive mypy

- Status: accepted
- Date: 2026-05-21

### Context

The repository needs lightweight, repeatable validation before runtime features
exist.

### Decision

Use uv for environment management, Ruff for linting and formatting, pytest for
tests, pytest-benchmark for future benchmark work, and permissive mypy
configuration that can become stricter over time.

### Consequences

- The standard milestone checks are `uv run pytest`, `uv run ruff check .`, and
  `uv run ruff format --check .`.
- Changes to `pyproject.toml` also require `uv sync` and a `tomllib` parse check.
- Runtime dependencies should remain minimal until a roadmap phase requires
  them.

## ADR 0005: Public Project Name and Import Name

- Status: accepted
- Date: 2026-05-21

### Context

The public project name uses mixed-case branding, while Python imports should be
lowercase and conventional.

### Decision

Use `tinyNLP` as the public project name and `tinynlp` as the import package
name.

### Consequences

- Documentation should use `tinyNLP` for the project and `tinynlp` for imports.
- Source code lives under `src/tinynlp`.
- Tests should verify that `import tinynlp` works.

## ADR 0006: Distribution Name

- Status: accepted
- Date: 2026-05-21

### Context

Package-index name availability can differ from the desired import package.
Using a distinct distribution name leaves room to publish without changing the
import path later.

### Decision

Use `tinynlp-opt` as the likely PyPI distribution name while keeping the import
package `tinynlp`. Revisit this decision only if package-name availability or
publishing constraints require it.

### Consequences

- `pyproject.toml` uses `name = "tinynlp-opt"`.
- The import path remains `tinynlp`.
- Package metadata lookups should use the distribution name `tinynlp-opt`.

## ADR 0007: Milestone-Based Execution

- Status: accepted
- Date: 2026-05-21

### Context

tinyNLP has a long pipeline, and premature implementation would blur the
project boundaries.

### Decision

Execute work by roadmap milestone, with one commit per milestone unless the user
asks for a different commit shape.

### Consequences

- Do not implement future phases during control-packet or documentation work.
- Each milestone should leave the repo validated and easy to review.
- `TASKS.md` should translate the roadmap into concrete work only when the next
  planning step begins.

## ADR 0008: Kernel Plans and Backend Protocol

- Status: accepted
- Date: 2026-05-21

### Context

tinyNLP needs execution kernels to be visible before optimized backends are
added. The reference evaluator should remain usable while exposing the steps an
optimized backend would eventually implement.

### Decision

Represent supported expression execution as deterministic `KernelPlan` objects,
execute them through a small backend protocol, register the Python reference
backend by name, and preserve `evaluate(expr, values)` as a compatibility
wrapper.

### Consequences

- Optimized backends must match reference behavior before they can support
  performance claims.
- Runtime values stay separate from symbolic plans.
- Performance claims still require committed benchmark evidence.

## ADR 0009: Explicit KKT Systems and Reference Linear Solve

- Status: accepted
- Date: 2026-05-21

### Context

tinyNLP needs the transition from sparse Jacobian assembly to KKT systems to be
visible before nonlinear solver steps or production linear algebra are added.

### Decision

Represent KKT systems as explicit coordinate matrices with block metadata and
provenance. Use an identity/reference primal block by default, not a Hessian
claim, and provide a pure-Python dense reference linear solver for tiny
deterministic systems.

### Consequences

- KKT entries can be traced back to primal, Jacobian, transposed Jacobian, or
  constraint-zero block metadata.
- The dense reference solver exists for correctness and interface testing, not
  performance claims.
- Nonlinear solver loops, Hessian assembly, and production factorization remain
  later milestones.

## ADR 0010: Simple Residual-Reduction Solver Prototype

- Status: accepted
- Date: 2026-05-22

### Context

tinyNLP needs a first visible solver-step workflow after residual/Jacobian
assembly, KKT construction, and reference linear solves. The project does not
yet have objective gradients, Hessian assembly, or production nonlinear solver
policy.

### Decision

Add a simple constrained solver prototype that repeatedly solves the explicit
identity-primal KKT system:

```text
[I  J^T] [dx    ] = [0 ]
[J   0 ] [lambda]   [-r]
```

This step is a transparent minimum-norm linearized residual correction. It is
not a Hessian-backed Newton method or an IPOPT-style production solver.

### Consequences

- Solver traces expose residual norms, step norms, KKT solve residuals,
  objective metric values when available, and deterministic variable values.
- Objective values remain reported metrics until objective-gradient and Hessian
  support exists.
- Future production solver methods must preserve this inspectable reference path
  and avoid performance claims without committed benchmark evidence.

## ADR 0011: KKT-Based Implicit Sensitivity Prototype

- Status: accepted
- Date: 2026-05-22

### Context

tinyNLP needs a first sensitivity workflow that stays tied to the existing
visible pipeline. The repository has residual/Jacobian assembly, explicit KKT
systems, and a dense reference linear solver, but it does not yet have Hessian
assembly, production differentiable-optimization machinery, or inequality/bound
sensitivity support.

### Decision

Add a scalar-parameter implicit sensitivity prototype that partitions the
assembled residual Jacobian into solve-variable columns `J_z` and one parameter
column `J_p`, then solves the explicit identity-primal KKT system:

```text
[I  J_z^T] [dz/dp ] = [0    ]
[J_z  0  ] [lambda]   [-J_p]
```

The workflow accepts residual-satisfying solution values, exposes RHS entries,
KKT metadata, and solve residuals in a trace object, and verifies behavior
against finite differences on tiny deterministic examples.

The scalar parameter is represented by a symbolic problem variable in the M13
prototype. The caller chooses which variables are solve variables; for
rectangular reduced Jacobians, the identity-primal KKT block defines a
minimum-norm reference convention rather than a general NLP optimal-solution
sensitivity. The reduced KKT system must be nonsingular for the chosen solve
variables.

### Consequences

- Sensitivities remain a reference workflow for supported smooth equality
  residual systems, not a production differentiable-optimization framework.
- Runtime values, parameter columns, RHS construction, and KKT solves remain
  inspectable.
- Variables that are objective-only or otherwise not part of the sensitivity
  solve should be excluded with explicit `solve_variables`.
- Multi-parameter sensitivities, inequality/bound sensitivities, Hessian-backed
  methods, and performance claims remain later work.

## ADR 0012: Scheduler-First Optimization

- Status: accepted
- Date: 2026-05-22

### Context

tinyNLP now has visible pieces of the nonlinear-programming execution pipeline:
expression IR, derivatives, sparsity, residual/Jacobian assembly, KKT systems,
solver steps, and sensitivity workflows. The next optimization work needs to
stay aligned with the tinygrad-inspired goal of making execution explicit rather
than adding isolated fast paths.

### Decision

Introduce a scheduler layer before optimized backend work. The scheduler should
represent deterministic pipeline tasks, their dependencies, inputs, outputs,
cached symbolic structures, materialized numeric values, backend choices, and
validation status.

Optimized backends must target scheduled tasks and validate against the
reference path for the same task. Scheduler reports should become the place to
inspect what ran, what was cached or materialized, which backend executed it,
and what validation passed.

### Consequences

- Backend optimization is scheduler-backed instead of ad hoc.
- Benchmark claims must name the scheduled stage, reference baseline, validation
  result, command, environment metadata, and committed result summary.
- Optional correctness bridges can remain useful, but they should integrate with
  the scheduled pipeline rather than bypassing it.
- Runtime scheduler implementation, bridges, optimized backends, and benchmark
  result summaries remain separate later milestones.

## ADR 0013: Execution Schedule Core as Metadata

- Status: accepted
- Date: 2026-05-22

### Context

tinyNLP has expression-level `KernelPlan` objects and visible structures for
residual/Jacobian assembly, sparse coordinates, KKT systems, solver steps, and
sensitivities. Before adding optimized execution, the project needs a broader
pipeline-level schedule that keeps these stages inspectable without changing
their current execution behavior.

### Decision

Add `ExecutionSchedule` and `ExecutionTask` as dependency-free metadata objects.
They describe ordered pipeline tasks, dependencies, inputs, outputs, cached
structures, materialized-value placeholders, backend choices, provenance, and
validation status.

`KernelPlan` remains the expression-level executable plan. `ExecutionSchedule`
summarizes larger NLP pipeline tasks and may cache `KernelPlan` summaries when a
task depends on expression evaluation. Schedule construction is descriptive in
M15: it does not execute tasks, optimize backends, or change solver behavior.

### Consequences

- The scheduler layer can describe residual/Jacobian assembly and KKT assembly
  before it controls execution.
- Later reports and optimized backends have a stable task metadata surface.
- Schedule metadata must stay deterministic, address-free, and tied to existing
  provenance.

## ADR 0014: Scheduled Reports as Audit Artifacts

- Status: accepted
- Date: 2026-05-22

### Context

After M15, tinyNLP has execution schedule metadata for expression, assembly, and
KKT stages. The next scheduler layer needs readable reports that make scheduled
work inspectable before any backend starts using schedules to control optimized
execution.

### Decision

Add deterministic scheduled pipeline reports that print task order, dependency
edges, stage names, inputs, outputs, cached structures, materialized values,
backend choices, provenance, and validation status.

These reports are audit artifacts. They may summarize reference results that
were materialized by examples or tests, but they do not execute schedules,
optimize kernels, or replace existing solver and sensitivity traces.

### Consequences

- Users can inspect the frontend -> scheduler -> backend path before optimized
  execution exists.
- Report examples can validate the current reference pipeline without making
  speed claims.
- Later optimized backend work must preserve or extend the report surface rather
  than bypassing it.

## ADR 0015: Optional CasADi Correctness Bridge

- Status: accepted
- Date: 2026-05-22

### Context

tinyNLP needs a small external correctness check before scheduler-backed
optimized backend work. CasADi is useful as a symbolic reference for supported
expression and Jacobian calculations, but it must not become part of the core
runtime or solver path.

### Decision

Add CasADi as an optional extra and isolate all CasADi imports under
`tinynlp.bridges.casadi`. The bridge converts supported tinyNLP IR operations
to CasADi symbols internally and compares expression values plus
residual/Jacobian assembly on canonical problems.

The bridge does not call IPOPT, use CasADi code generation, wrap external
solvers, compare performance, or expose CasADi objects through core modules.

### Consequences

- Normal installs and tests remain skip-safe without CasADi.
- CasADi can validate current reference math on supported examples when the
  optional extra is installed.
- Scheduler/report integration for external validation metadata remains a later
  optimization and result-claim milestone concern.

## ADR 0016: First Optimized Target Is Scheduled Residual Evaluation

- Status: accepted
- Date: 2026-05-22

### Context

tinyNLP now has `KernelPlan`, backend protocol, problem assembly contracts,
execution schedules, scheduled reports, and optional CasADi correctness checks.
Before adding an optimized backend, the project needs a narrow target that is
large enough to exercise the NLP pipeline but small enough to validate and
benchmark without making broad performance claims.

Candidate stages were expression evaluation through `KernelPlan`, residual
evaluation for the chain problem, residual/Jacobian fusion, cached sparse
coordinate assembly, KKT assembly, and sensitivity workflows.

### Decision

The first optimized backend target is scheduled residual evaluation for the
canonical chain dynamics problem. The optimized path should attach to the
scheduled `evaluate_residuals` task and execute prepared residual `KernelPlan`s
through a dependency-free CPU backend.

The reference baseline is existing residual assembly using a cached
`AssemblyContract` and the registered Python backend. CasADi remains an optional
correctness bridge and is not a performance baseline.

### Consequences

- The first optimized backend must be scheduler-backed rather than an ad hoc
  fast path.
- Correctness validation against the reference residual values is required
  before timing can support any result summary.
- Any result claim is limited to scheduled chain residual evaluation and must
  not imply solver, Jacobian, KKT, sensitivity, or package-wide speed.
- Residual/Jacobian fusion, KKT assembly optimization, sensitivity speedups,
  GPU support, and code generation remain later decisions.

## ADR 0017: Prepared KernelPlan Backend for Scheduled Residuals

- Status: accepted
- Date: 2026-05-23

### Context

M19 needed one narrow scheduler-backed optimized CPU path. The audit selected
scheduled residual evaluation for the canonical chain dynamics problem, with the
existing Python residual assembly as the reference baseline.

### Decision

Add a dependency-free `prepared-python` backend that prepares supported
`KernelPlan`s into slot-indexed executable data. Use it through a scheduled
residual evaluator for the `evaluate_residuals` stage. The reference Python
backend and existing residual assembly path remain unchanged.

The prepared backend does not use `eval`, `exec`, code generation, GPU support,
or external dependencies. It is attached to `ExecutionSchedule` metadata so
reports can show backend choice, cached prepared kernels, materialized residual
values, and reference-validation status.

### Consequences

- The first optimized path is scheduler-backed and inspectable.
- Output validation against reference residual assembly is required before
  timing.
- The committed benchmark result summary is limited to scheduled chain residual
  evaluation.
- Broader optimized backends, residual/Jacobian fusion, KKT optimization,
  sensitivity optimization, and hardware-specific execution remain later work.
