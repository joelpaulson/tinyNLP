# M11 Readiness

M11 is sufficient for the first M12 prototype if that prototype is framed as a
small constrained residual-reduction workflow.

## Ready For M12

- `Problem.variables` provides deterministic variable order.
- Residual values, Jacobian coordinates, KKT blocks, and KKT entry provenance
  are visible.
- `DenseReferenceLinearSolver` can solve tiny KKT systems for reference tests.
- Existing pipeline objects expose enough structure for an iteration trace.

## Current Boundary

- Objective values can be tracked as metrics.
- Objective gradients, Hessian assembly, and true Lagrangian KKT blocks are not
  implemented yet.
- M12 should not claim general NLP optimality.
- Solver traces should stay tied to residual, Jacobian, KKT, and linear-solve
  artifacts.
