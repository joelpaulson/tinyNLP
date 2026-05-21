# ROADMAP

tinyNLP will grow in small, benchmarkable milestones. The direction is a
transparent nonlinear programming execution pipeline, not a monolithic modeling
layer.

## Milestone 0: Repository Bootstrap

- Package skeleton with uv-compatible tooling.
- Contributor rules and initial decision log.
- CI for import tests, Ruff linting, Ruff formatting checks, and pytest.
- Benchmark policy without benchmark claims.

## Milestone 1: Minimal Smooth Problem Path

- Introduce a small expression graph IR.
- Build derivative graph construction for a narrow set of smooth operations.
- Keep symbolic structure separate from numeric values.
- Add tests that trace every operation through the pipeline.

## Milestone 2: Structured Assembly

- Add sparse structure discovery.
- Assemble residuals, Jacobians, and Hessians for supported smooth problems.
- Record enough trace data to explain each assembled contribution.
- Add benchmark harnesses before making any performance claims.

## Milestone 3: KKT Systems and Solver Steps

- Build explicit KKT system assembly for the supported problem class.
- Add simple CPU-first solver steps.
- Keep solver internals inspectable and test-covered.
- Compare behavior with committed benchmark and correctness reports.

## Milestone 4: Sensitivities and Reports

- Add sensitivity workflows once KKT and derivative paths are stable.
- Emit benchmark reports with machine, problem, and configuration metadata.
- Keep benchmark output reproducible enough to support README claims.

## Later Milestones

- Add inequalities and bounds when the equality-oriented path is stable.
- Add additional solver backends when the internal contracts are proven.
- Explore hardware-aware CPU optimization before considering GPU support.
- Add bridges only when they preserve tinyNLP's inspectable pipeline.
