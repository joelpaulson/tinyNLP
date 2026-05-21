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
