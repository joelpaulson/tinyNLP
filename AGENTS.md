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
- Maintain and test completed M1-M20 features: core expression IR, reference
  evaluator, trace reports, canonical examples, benchmark smoke scaffolding,
  KernelPlan, backend protocol, registered Python reference backend, autodiff,
  Jacobian verification, structural sparsity, problem APIs, residual/Jacobian
  assembly, KKT system objects, the reference linear-solve interface, and the
  simple constrained residual-reduction solver, implicit sensitivity prototype,
  execution schedule metadata, scheduled report helpers, schedule audit
  examples, the optional CasADi correctness bridge, and the M19 prepared
  scheduler-backed residual-evaluation backend, plus the M20 benchmark-claim
  audit and final roadmap closeout.
- Refine benchmark policy and benchmark harness structure before benchmark
  result summaries.

Codex may not implement these before a milestone asks for them:

- Hessian assembly.
- Production solver backends or broad sensitivity workflows beyond the M13
  prototype.
- Hessian-backed nonlinear solver methods beyond the M12 residual-reduction
  prototype.
- Broad bridges to external modeling systems beyond the optional M17 CasADi
  correctness bridge.
- Optimized backends beyond the M19 prepared residual-evaluation path.
- Scheduler-driven optimized execution beyond the M19 scheduled
  `evaluate_residuals` path.
- Benchmark result summaries beyond the M19 scheduled residual-evaluation
  summary.
- Inequalities, bounds, GPU support, hardware-specific code generation, or large
  external solver wrappers.
- New expression operations unless the active milestone explicitly asks for
  them and includes derivative, backend, bridge, and tests.

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

## Autonomous Branch Workflow

For long Codex runs:

- Start from a clean `main` that is up to date with `origin/main`.
- Create a short-lived branch named `codex/<milestone-or-purpose>`.
- Work one milestone at a time and keep one commit per milestone.
- Run required checks before each milestone commit.
- Fast-forward merge back to `main` only when checks pass and `main` has not
  diverged.
- Push `main` after the fast-forward merge if the user requested remote sync.
- Delete the short-lived local branch after a successful merge.
- Stop and ask the user if the merge is not fast-forward, if conflicts appear,
  or if unrelated user changes are present.

## Flagship Roadmap Transition Rules

For the F-series flagship usability and speed roadmap:

- Work one F milestone at a time.
- Do not start the next F milestone until required checks pass and `TASKS.md`
  marks the current milestone complete and the next milestone ready.
- Keep the flagship example honest: it must expose the model/problem, schedule,
  residual/Jacobian assembly, solver trace when applicable, optional CasADi
  correctness report, and benchmark command rather than hiding work in opaque
  utilities.
- Keep correctness, usability, and speed evidence separate.
- Do not add README speed wording unless a committed benchmark result summary
  supports the exact claim.
- Do not add new operations, inequalities, bounds, GPU support, code generation,
  broad bridges, or broad solver claims unless the active milestone explicitly
  asks for them.

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
