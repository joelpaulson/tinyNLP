# AGENTS

This repository is intentionally small and experimental. Agents and contributors
should optimize for clarity, traceability, and testable progress.

## Core Rules

- Keep the core small and inspectable.
- Every operation should be traceable.
- Separate symbolic structure from numeric values.
- Every new feature needs tests.
- Every performance claim needs a benchmark.
- Do not add inequalities, GPU support, or a large external solver wrapper until
  ROADMAP explicitly asks for it.
- Do not make speed claims in README unless backed by committed benchmark
  output.

## Milestone Checks

After each milestone run:

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

## Development Guidance

- Prefer explicit data flow over hidden global state.
- Keep public APIs narrow until the pipeline has real examples.
- Do not implement expression IR, autodiff, solver logic, or bridges as part of
  repository maintenance unless the current roadmap milestone calls for it.
- If a feature changes the observable pipeline, add tests that show how the
  trace changes.
- If a benchmark result is cited in documentation, commit the benchmark source,
  command, environment notes, and output summary together.
