# TASKS

This is the execution task board for Codex-driven tinyNLP development. It turns
the high-level roadmap into milestone-sized work while preserving the project
identity: a tinygrad-inspired, CPU-first nonlinear programming execution
substrate whose pipeline is visible, traceable, benchmarkable, and eventually
optimizable for available hardware.

## How Codex Should Use This File

- Work one milestone at a time.
- Use one branch per milestone or milestone group.
- Make one commit per milestone.
- Update this file's milestone status before committing.
- For autonomous runs, create a short-lived branch from clean `main`.
- Keep each milestone in its own commit and update milestone status before
  committing.
- After a milestone commit passes checks, fast-forward merge to `main` only if
  `main` has not diverged.
- Stop on merge conflicts, non-fast-forward merges, unrelated worktree changes,
  or validation failures after two focused repair attempts.
- Keep equality constraints as an initial implementation path, not the package
  identity.
- Do not add inequalities, bounds, GPU support, production IPOPT-style logic, or
  performance claims until a milestone explicitly asks for them.
- With M11 complete, stop before M12 unless explicitly approved.

## Shared Required Checks

Every milestone must end with:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If `pyproject.toml` changes, also run:

```sh
uv sync
python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"
```

## M0 Documentation and Tooling Sanity

- Status: complete.
- Purpose: confirm the control packet, packaging, CI, and import skeleton are
  coherent before runtime feature work begins.
- Allowed scope: documentation cleanup, CI wording, package metadata fixes,
  import smoke tests, and task-board status updates.
- Out-of-scope items: expression IR, evaluator behavior, autodiff, sparsity,
  assembly, KKT objects, solvers, bridges, benchmark implementation, new runtime
  dependencies.
- Files likely touched: `README.md`, `ROADMAP.md`, `AGENTS.md`,
  `BENCHMARKING.md`, `DECISIONS.md`, `TASKS.md`, `docs/`, `pyproject.toml`,
  `.github/workflows/ci.yml`, `tests/test_import.py`.
- Implementation notes: preserve `tinyNLP` as the public name, `tinynlp-opt` as
  the distribution name, and `tinynlp` as the import package.
- Acceptance tests: import smoke test passes; docs do not contain speed claims;
  no runtime feature modules are implemented.
- Benchmark requirements: none; benchmark policy only.
- Required checks: shared required checks, plus the `pyproject.toml` checks if
  packaging changes.
- Commit message: `Sanity check documentation and tooling`.
- Stop conditions: naming decisions conflict with `DECISIONS.md`; tooling
  changes require a dependency not already approved.

## M1 Core IR Data Model

- Status: complete.
- Purpose: introduce the smallest inspectable expression IR data model.
- Allowed scope: immutable node/value identifiers, operation metadata,
  variables, constants, basic scalar operations, graph/container types, and
  structural tests.
- Out-of-scope items: evaluation, autodiff, simplification, code generation,
  sparsity, problem APIs, inequalities, bounds, solvers, bridges, benchmarks.
- Files likely touched: `src/tinynlp/ir/`, `tests/`, `docs/architecture.md`,
  `TASKS.md`.
- Implementation notes: keep symbolic structure separate from numeric values;
  every operation must retain enough metadata to be traced later.
- Acceptance tests: graph construction is deterministic; node ordering is
  stable; variables/constants/basic operations can be represented without
  evaluating them.
- Benchmark requirements: none.
- Required checks: shared required checks.
- Commit message: `Add core IR data model`.
- Stop conditions: the IR design requires hidden global state, eager numeric
  evaluation, or an unapproved dependency.

## M2 Reference Expression Evaluator

- Status: complete.
- Purpose: add a CPU-first reference evaluator for the supported IR.
- Allowed scope: scalar numeric evaluation, environment/value binding,
  deterministic errors for missing values, and evaluator tests.
- Out-of-scope items: autodiff, vector Jacobians, sparsity, assembly, KKT,
  solvers, optimized backends, benchmarks, inequalities, bounds.
- Files likely touched: `src/tinynlp/ir/`, `src/tinynlp/backends/`, `tests/`,
  `docs/architecture.md`, `TASKS.md`.
- Implementation notes: evaluator behavior should be simple and explicit; keep
  evaluation separate from graph construction.
- Acceptance tests: supported expressions evaluate to expected scalar values;
  missing inputs and unsupported operations produce clear errors.
- Benchmark requirements: none.
- Required checks: shared required checks.
- Commit message: `Add reference expression evaluator`.
- Stop conditions: evaluator design mutates IR structure or obscures operation
  traceability.

## M3 Trace Reports and Canonical Examples

- Status: complete.
- Purpose: make the early IR/evaluator path explainable through traces and
  examples.
- Allowed scope: human-readable trace reports, minimal example scripts or docs,
  tests for trace content, and docs explaining the visible path.
- Out-of-scope items: autodiff, sparsity, assembly, KKT, solvers, benchmark
  results, performance claims, inequalities, bounds.
- Files likely touched: `src/tinynlp/profiling/`, `examples/`, `docs/`,
  `tests/`, `TASKS.md`.
- Implementation notes: traces should identify model expression to IR to
  evaluation steps without adding a large logging framework.
- Acceptance tests: canonical examples run in tests or smoke tests; trace output
  contains stable operation identifiers and evaluation order.
- Benchmark requirements: none.
- Required checks: shared required checks.
- Commit message: `Add trace reports and canonical examples`.
- Stop conditions: examples require unsupported math, dependencies, or future
  pipeline stages.

## M4 Benchmark Scaffolding

- Status: complete.
- Purpose: create benchmark infrastructure before any performance claims.
- Allowed scope: benchmark directory structure, benchmark metadata schema,
  correctness-before-timing helpers, pytest-benchmark smoke cases for the
  reference evaluator, and documentation for result summaries.
- Out-of-scope items: optimized backend work, speed claims, committed comparison
  claims, solver benchmarks, external-package baselines, inequalities, bounds.
- Files likely touched: `benchmarks/`, `src/tinynlp/profiling/`, `tests/`,
  `BENCHMARKING.md`, `TASKS.md`.
- Implementation notes: benchmarks must validate expected outputs before timing;
  metadata must capture command, environment, problem, and measured stage.
- Acceptance tests: benchmark smoke cases can be collected; correctness checks
  fail before timing when expected outputs are wrong.
- Benchmark requirements: expected benchmark is expression/residual evaluation
  for M3 canonical examples; baseline is the M2 CPU reference evaluator compared
  only against expected numeric outputs, not against another package or backend.
- Required checks: shared required checks.
- Commit message: `Add benchmark scaffolding`.
- Stop conditions: benchmark output would be presented as a performance claim;
  scaffolding needs unapproved runtime dependencies.

## M5 Kernel Plan and Backend Protocol

- Status: complete.
- Purpose: make expression execution kernels visible before optimized backends.
- Allowed scope: `KernelPlan` construction for supported expression graphs,
  backend protocol/interface, registered reference Python backend, kernel plan
  report formatting, simple operation and temporary counts, tests, and benchmark
  smoke cases through the protocol.
- Out-of-scope items: autodiff, sparsity discovery, problem APIs, KKT systems,
  solvers, bridges, inequalities, bounds, GPU support, Numba, LLVM, C codegen,
  CasADi, SciPy, optimized backends, external solver wrappers, speed claims.
- Files likely touched: `src/tinynlp/backends/`, `src/tinynlp/profiling/`,
  `tests/backends/`, `tests/profiling/`, `benchmarks/`, `TASKS.md`.
- Implementation notes: preserve `evaluate(expr, values)` as a small
  compatibility wrapper around the registered reference backend; keep symbolic
  graph structure separate from runtime values; make plan order and reports
  deterministic.
- Acceptance tests: kernel plans are deterministic; operation counts and
  temporary counts are stable; protocol execution matches current reference
  evaluator behavior; report output is human-readable and contains no object
  addresses.
- Benchmark requirements: benchmark expression/residual evaluation through the
  backend protocol only; baseline is expected numeric output, not another package
  or optimized backend; no result summaries or speed claims.
- Required checks: shared required checks.
- Commit message: `Add kernel plan and backend protocol`.
- Stop conditions: backend interface grows beyond expression execution, requires
  new dependencies, implies optimized performance, or starts implementing future
  milestones.

## M6 Reverse-Mode Autodiff for Scalar Outputs

- Status: complete.
- Purpose: add reverse-mode derivative construction for scalar-output
  expressions.
- Allowed scope: derivative graph or tape representation for supported scalar
  operations, reverse accumulation, gradient evaluation, and derivative tests.
- Out-of-scope items: vector Jacobians, sparsity discovery, Hessian assembly,
  KKT, solvers, optimized backends, bridges, inequalities, bounds.
- Files likely touched: `src/tinynlp/autodiff/`, `src/tinynlp/ir/`, `tests/`,
  `docs/architecture.md`, `TASKS.md`.
- Implementation notes: derivative construction must remain inspectable and tied
  back to IR operations; do not collapse symbolic structure into opaque callables.
- Acceptance tests: gradients match hand-derived results for supported scalar
  expressions; traces identify derivative contributions.
- Benchmark requirements: none; do not publish timing claims.
- Required checks: shared required checks.
- Commit message: `Add reverse-mode autodiff for scalar outputs`.
- Stop conditions: autodiff requires unsupported operations, hidden mutation, or
  external autodiff dependencies.

## M7 Vector Jacobians and Derivative Verification

- Status: complete.
- Purpose: extend derivative support to vector outputs and add verification
  checks.
- Allowed scope: Jacobian construction/evaluation for vector expressions,
  finite-difference or exact-reference verification tests, and derivative trace
  reports.
- Out-of-scope items: sparsity discovery as a separate feature, sparse assembly,
  KKT systems, solvers, optimized backends, bridges, inequalities, bounds.
- Files likely touched: `src/tinynlp/autodiff/`, `src/tinynlp/ir/`, `tests/`,
  `docs/architecture.md`, `TASKS.md`.
- Implementation notes: verification tolerances must be explicit and tests
  should stay small enough to debug by inspection.
- Acceptance tests: Jacobians match verification checks for canonical vector
  examples; unsupported derivative cases fail clearly.
- Benchmark requirements: none; derivative benchmark sources wait for a later
  benchmark milestone unless explicitly requested.
- Required checks: shared required checks.
- Commit message: `Add vector Jacobians and derivative verification`.
- Stop conditions: verification becomes flaky or requires unapproved numeric
  libraries.

## M8 Structural Sparsity Discovery

- Status: complete.
- Purpose: discover reusable structural sparsity separately from numeric values.
- Allowed scope: structural dependency analysis, sparsity patterns for supported
  expressions/Jacobians, trace reports, and stability tests across value changes.
- Out-of-scope items: sparse numeric assembly, KKT systems, solvers, optimized
  backends, bridges, inequalities, bounds.
- Files likely touched: `src/tinynlp/nlp/`, `src/tinynlp/autodiff/`,
  `src/tinynlp/ir/`, `tests/`, `docs/architecture.md`, `TASKS.md`.
- Implementation notes: sparsity is symbolic structure; do not require numeric
  values to discover it.
- Acceptance tests: sparsity patterns are deterministic; changing numeric values
  does not change structural patterns for the same graph.
- Benchmark requirements: none; no timing claims.
- Required checks: shared required checks.
- Commit message: `Add structural sparsity discovery`.
- Stop conditions: sparsity analysis depends on numeric sampling or hides
  dependency provenance.

## M9 Problem API and Assembly Contracts

- Status: complete.
- Purpose: define the first problem-level API and explicit assembly contracts.
- Allowed scope: smooth structured constrained problem containers, objective and
  residual registration, equality residual blocks as the first supported block
  kind, shape metadata, assembly contract types, and tests.
- Out-of-scope items: inequalities, bounds, production solver APIs, sparse
  numeric assembly implementation, KKT systems, external solver wrappers.
- Files likely touched: `src/tinynlp/nlp/`, `src/tinynlp/ir/`, `tests/`,
  `docs/architecture.md`, `TASKS.md`.
- Implementation notes: equality constraints may be the initial implementation
  path, but docs and APIs should not define tinyNLP as equality-only; leave room
  for future inequality and bound blocks.
- Acceptance tests: problem definitions expose dimensions, symbolic structure,
  residual expressions, and assembly plans without solving.
- Benchmark requirements: none.
- Required checks: shared required checks.
- Commit message: `Add problem API and assembly contracts`.
- Stop conditions: API design implies unsupported inequalities/bounds or
  hard-codes a specific solver backend.

## M10 Sparse Residual/Jacobian Assembly

- Status: complete.
- Purpose: assemble residual values and Jacobians using discovered structure.
- Allowed scope: sparse coordinate/structure representation, residual assembly,
  Jacobian assembly, shape checks, trace metadata, and correctness tests.
- Out-of-scope items: Hessian assembly unless explicitly scoped, KKT systems,
  solver steps, optimized backends, external sparse libraries, inequalities,
  bounds.
- Files likely touched: `src/tinynlp/nlp/`, `src/tinynlp/autodiff/`,
  `src/tinynlp/backends/`, `tests/`, `docs/architecture.md`, `TASKS.md`.
- Implementation notes: use simple dependency-free coordinate sparse structures
  first; do not add SciPy unless a later milestone explicitly approves it.
- Acceptance tests: assembled residuals and Jacobians match dense/reference
  expectations for canonical problems; trace metadata identifies each
  contribution.
- Benchmark requirements: none for performance claims; benchmark cases may be
  planned but not cited.
- Required checks: shared required checks.
- Commit message: `Add sparse residual and Jacobian assembly`.
- Stop conditions: implementation requires external sparse dependencies or
  obscures contribution-level traceability.

## M11 KKT System Object and Linear-Solve Interface

- Status: complete.
- Purpose: represent KKT systems explicitly and define a minimal linear-solve
  interface.
- Allowed scope: KKT block metadata, simple dense/reference KKT assembly from
  supported problem contracts, linear-solve protocol, and tests with fake or
  reference solvers.
- Out-of-scope items: production factorization, large external solver wrappers,
  IPOPT-style logic, nonlinear solver globalization, inequalities, bounds.
- Files likely touched: `src/tinynlp/solvers/`, `src/tinynlp/nlp/`,
  `src/tinynlp/backends/`, `tests/`, `docs/architecture.md`, `TASKS.md`.
- Implementation notes: keep the linear-solve interface backend-neutral and
  inspectable; KKT blocks must expose dimensions and provenance; use an
  identity/reference primal block until a later Hessian milestone exists.
- Acceptance tests: KKT object exposes expected block structure; reference
  linear solve interface can solve tiny deterministic systems.
- Benchmark requirements: none; KKT timing waits for benchmark baseline work.
- Required checks: shared required checks.
- Commit message: `Add KKT system object and linear-solve interface`.
- Stop conditions: design commits to a production solver backend or hides KKT
  block construction.

## M12 Simple Constrained Solver Prototype

- Status: ready.
- Purpose: add a small constrained solver prototype for supported smooth
  structured constrained problems.
- Allowed scope: simple step loop, residual/KKT calls, convergence diagnostics,
  failure reporting, and tests on tiny canonical problems.
- Out-of-scope items: production IPOPT-style logic, inequality/bound handling,
  line-search complexity beyond the prototype need, large wrappers, performance
  claims.
- Files likely touched: `src/tinynlp/solvers/`, `src/tinynlp/nlp/`, `tests/`,
  `examples/`, `docs/architecture.md`, `TASKS.md`.
- Implementation notes: prioritize correctness, traceability, and readable
  diagnostics over breadth; keep equality constraints an initial path only. Use
  the chain dynamics problem as the canonical smoke case; objective values are
  tracked metrics until objective-gradient and Hessian support exists.
- Acceptance tests: solver reduces residuals on tiny deterministic problems and
  reports clear failure states on invalid inputs.
- Benchmark requirements: none; no solve-time claims.
- Required checks: shared required checks.
- Commit message: `Add simple constrained solver prototype`.
- Stop conditions: prototype starts growing into production solver policy or
  requires unsupported problem classes.

## M13 Implicit Sensitivity Prototype

- Status: blocked until M12 is complete.
- Purpose: add a first sensitivity workflow using the existing derivative and
  KKT path.
- Allowed scope: sensitivity object/model, implicit solve through the KKT
  interface, reference tests, and trace reports for sensitivity steps.
- Out-of-scope items: inequality/bound sensitivities, production robustness,
  optimized backends, external solver wrappers, performance claims.
- Files likely touched: `src/tinynlp/solvers/`, `src/tinynlp/nlp/`,
  `src/tinynlp/autodiff/`, `tests/`, `docs/architecture.md`, `TASKS.md`.
- Implementation notes: document assumptions and failure modes; keep sensitivity
  calculations tied to inspectable KKT metadata.
- Acceptance tests: sensitivities match reference finite-difference checks on
  tiny canonical problems within explicit tolerances.
- Benchmark requirements: none; no timing claims.
- Required checks: shared required checks.
- Commit message: `Add implicit sensitivity prototype`.
- Stop conditions: sensitivity path requires unsupported solver features or
  produces unverifiable results.

## M14 CasADi Baseline Bridge

- Status: blocked until M13 is complete.
- Purpose: add a baseline bridge for correctness comparison against CasADi where
  explicitly available.
- Allowed scope: optional bridge module, optional dependency wiring if approved,
  small conversion path for canonical supported problems, and skip-safe tests.
- Out-of-scope items: making CasADi a required runtime dependency, IPOPT-style
  production logic, broad modeling coverage, inequalities, bounds, performance
  claims.
- Files likely touched: `src/tinynlp/bridges/`, `tests/`, `examples/`,
  `pyproject.toml`, `docs/architecture.md`, `TASKS.md`.
- Implementation notes: bridge must be optional and isolated; if
  `pyproject.toml` changes, run the additional pyproject checks. M14 can be
  revisited during later planning if an earlier correctness baseline becomes
  valuable, but CasADi must remain optional unless a milestone explicitly
  changes dependency policy.
- Acceptance tests: tests skip cleanly when CasADi is absent; when present, the
  bridge matches tinyNLP results on canonical supported problems.
- Benchmark requirements: none for speed; CasADi may be a correctness baseline
  only in this milestone.
- Required checks: shared required checks, plus `uv sync` and the `tomllib`
  parse check if dependency metadata changes.
- Commit message: `Add CasADi baseline bridge`.
- Stop conditions: bridge requires a mandatory heavy dependency, solver-wrapper
  behavior, or unsupported problem classes.

## M15 First Optimized Backend and Benchmark-Backed Result

- Status: blocked until M14 is complete.
- Purpose: add the first optimized backend and the first benchmark-backed
  result summary.
- Allowed scope: one narrow optimized CPU backend for an already-supported
  pipeline stage, benchmark source, benchmark command, environment metadata,
  committed result summary, and docs linking only to committed evidence.
- Out-of-scope items: GPU support, broad code generation, unsupported problem
  classes, uncommitted speed claims, production solver claims, inequalities,
  bounds.
- Files likely touched: `src/tinynlp/backends/`, `benchmarks/`, `docs/`,
  `BENCHMARKING.md`, `README.md`, `tests/`, `TASKS.md`.
- Implementation notes: optimized behavior must match the CPU reference path
  before timing is reported; keep the measured stage narrow.
- Acceptance tests: optimized backend produces the same outputs as the reference
  path on benchmarked problems; benchmark report validates outputs before timing.
- Benchmark requirements: expected benchmark is one committed expression/residual
  or sparse assembly benchmark from the implemented pipeline; baseline is the
  CPU reference backend for the same problem, command, dependency set, machine
  metadata, and output validation.
- Required checks: shared required checks.
- Commit message: `Add first optimized backend with benchmark result`.
- Stop conditions: result cannot be reproduced, output validation fails, or the
  benchmark would imply a broader performance claim than the evidence supports.
