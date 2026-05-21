# Design Principles

tinyNLP should make nonlinear-programming execution visible before making it
fast.

## Principles

- Keep the core small and inspectable.
- Make every operation traceable through the pipeline.
- Separate symbolic structure from numeric values.
- Prefer CPU-first reference behavior before optimized backends.
- Treat benchmark claims as artifacts that require committed evidence.
- Let roadmap milestones decide when scope expands.

## Practical Implications

- New features need tests that show the visible behavior.
- Traces should explain what was built, evaluated, assembled, or solved.
- Dependencies should be added only when they directly serve the active
  milestone.
- Optimizations should preserve the reference path and identify what they
  measure.
