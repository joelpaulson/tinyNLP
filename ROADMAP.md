# ROADMAP

tinyNLP will grow through small, milestone-based phases. The roadmap stays
high-level on purpose; detailed execution tasks will live in `TASKS.md` later.

## Phase 0: Control Packet and Tooling

- Keep the package skeleton, CI, contributor rules, benchmark policy, and design
  notes coherent.
- Maintain `uv`, `pytest`, `ruff`, permissive `mypy`, and `pytest-benchmark`
  as the initial development stack.
- Avoid runtime feature work in this phase.

## Phase 1: Expression IR and Reference Evaluator

- Introduce a small expression IR for smooth scalar/vector expressions.
- Add a CPU-first reference evaluator.
- Keep every operation traceable from model expression to IR node.

## Phase 2: Trace Reports and Canonical Examples

- Add concise trace reports that explain what was built and evaluated.
- Create canonical examples that exercise the visible pipeline.
- Keep examples focused on clarity rather than coverage.

## Phase 3: Autodiff and Derivative Verification

- Add derivative construction for the supported IR.
- Verify derivatives against reference checks.
- Keep derivative graphs inspectable and tied back to source operations.

## Phase 4: Sparsity and Structure Discovery

- Discover sparsity and reusable structure separately from numeric values.
- Report structural assumptions and discovered patterns.
- Add tests for structural stability across value changes.

## Phase 5: Problem API and Assembly Contracts

- Define a small problem API for smooth structured constrained problems.
- Establish residual, Jacobian, and Hessian assembly contracts.
- Keep assembly traceable to expression and derivative structure.

## Phase 6: KKT Systems and Linear-Solve Interface

- Build explicit KKT system assembly for supported problems.
- Add a minimal linear-solve interface without committing to large wrappers.
- Preserve enough metadata to inspect each KKT block.

## Phase 7: Simple Constrained Solver Prototype

- Add a simple constrained solver prototype for the supported problem class.
- Prioritize correctness, traceability, and debuggability.
- Keep solver behavior covered by tests and trace reports.

## Phase 8: Sensitivity Workflows

- Add sensitivity calculations after the KKT and derivative paths are stable.
- Validate sensitivities against reference problems.
- Document assumptions and failure modes.

## Phase 9: Benchmark Baselines and First Optimized Backend

- Establish benchmark baselines for the implemented pipeline.
- Commit benchmark commands, environment metadata, and result summaries.
- Add the first optimized backend only after reference behavior is stable.

## Later

- Inequalities and bounds.
- Additional solver backends.
- Code generation.
- Hardware-specific execution.
- Bridges to external formats when they preserve the inspectable pipeline.

## Next Control Artifact

`TASKS.md` will be generated next from this roadmap. It should translate each
phase into concrete, testable tasks without expanding scope beyond the active
milestone.
