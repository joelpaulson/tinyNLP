# AGENTS

This is the durable control packet for Codex and other automation working in
tinyNLP.

## Project Scope

tinyNLP is a tinygrad-inspired, CPU-first nonlinear programming execution
substrate. Its purpose is to make the path from model expression to IR,
derivatives, sparsity, assembly, KKT systems, solver steps, sensitivities, and
benchmark reports visible, traceable, benchmarkable, and eventually optimizable
for available hardware.

The initial implementation path starts with smooth structured constrained
problems. Inequalities, bounds, additional solver backends, code generation, and
hardware-specific execution are later roadmap items.

## Core Design Principles

- Keep the core small and inspectable.
- Every operation should be traceable.
- Separate symbolic structure from numeric values.
- Prefer explicit data flow over hidden global state.
- Keep public APIs narrow until examples and tests prove the shape.
- Every new feature needs tests.
- Every performance claim needs a benchmark.

## Implementation Boundaries

Codex may:

- Edit documentation, tests, CI, packaging, and project-control files when asked.
- Add minimal package stubs needed for layout or importability.
- Add tests for behavior that already exists or is explicitly requested by the
  active milestone.
- Maintain and test completed M1-M5 features: core expression IR, reference
  evaluator, trace reports, canonical examples, benchmark smoke scaffolding,
  KernelPlan, backend protocol, and registered Python reference backend.
- Refine benchmark policy and benchmark harness structure before benchmark
  result summaries.

Codex may not implement these before a milestone asks for them:

- Autodiff or derivative graph construction.
- Vector Jacobians or derivative verification workflows.
- Sparsity or structure discovery.
- Problem APIs.
- Residual, Jacobian, Hessian, or KKT assembly.
- Solver steps, solver backends, or sensitivities.
- Bridges to external modeling systems.
- Optimized backends.
- Benchmark result summaries.
- Inequalities, bounds, GPU support, hardware-specific code generation, or large
  external solver wrappers.

Do not make speed claims in README or project documentation unless they are
backed by committed benchmark output.

## Required Checks

After each milestone, run:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If a change affects `pyproject.toml`, also run:

```sh
uv sync
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

## Commit Discipline

- Keep one commit per milestone unless the user asks for a different shape.
- Use clear imperative commit messages.
- Do not mix feature implementation with unrelated formatting or policy churn.
- Do not commit generated artifacts unless they are expected project state, such
  as `uv.lock` or committed benchmark summaries.
- Before committing, check `git status --short` and ensure every staged file is
  intentional.

## Benchmark Policy

- Benchmarks are correctness checks plus timing, not demos.
- Validate benchmark outputs before using timing data.
- Record what stage is measured and what problem is used.
- README speed claims require benchmark source, command, environment metadata,
  and committed result summary.
- Do not report speedups without a reproducible baseline.

## Dependency Policy

- Keep runtime dependencies minimal.
- Do not add NumPy, SciPy, CasADi, JAX, PyTorch, solver packages, or bridge
  dependencies until a roadmap milestone explicitly needs them.
- Development dependencies should stay limited to validation and benchmark
  infrastructure.
- Explain any new dependency in the commit or adjacent documentation.

## Stop Conditions

Stop and ask the user before proceeding if:

- The requested change would implement a future roadmap feature.
- A dependency is needed but not explicitly allowed by the active milestone.
- A packaging or naming decision conflicts with `DECISIONS.md`.
- A benchmark result would be needed to support requested documentation.
- Local validation fails for a reason that changes the implementation plan.
