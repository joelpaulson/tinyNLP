# Scheduled Residual Evaluation Benchmark

This result summary is limited to one scheduled stage:
`evaluate_residuals` on `chain_dynamics_case(horizon=100)`.

It is not a solver, Jacobian, KKT, sensitivity, or package-wide performance
claim.

## Command

```sh
uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json /private/tmp/tinynlp-scheduled-residual-benchmark-1.json
uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json /private/tmp/tinynlp-scheduled-residual-benchmark-2.json
uv run pytest benchmarks/test_scheduler_backend_benchmark.py --benchmark-json /private/tmp/tinynlp-scheduled-residual-benchmark-final.json
```

## Code

- Evidence commit: `af56b74` (`Add first scheduler-backed optimized backend`).
- Audit commit: this summary was reviewed during M20
  `Audit benchmark claims after optimized backend`.
- Benchmark source: `benchmarks/test_scheduler_backend_benchmark.py`.
- Measured stage: scheduled `evaluate_residuals`.
- Problem: `chain_dynamics_case(horizon=100)`.
- Variables: 201.
- Residuals: 100.
- Reference baseline: `assemble_residuals(contract, values)` using a cached
  `AssemblyContract` and the registered Python backend.
- Optimized path: `ScheduledResidualEvaluator.evaluate(values)` using prepared
  residual `KernelPlan`s and the `prepared-python` backend.
- Output validation: passed before timing, with `max_abs_error=0`.

## Environment

- OS: Darwin 24.6.0 arm64.
- CPU: Apple M4.
- Python: 3.12.13.
- tinyNLP: 0.0.0.
- pytest: 9.0.3.
- pytest-benchmark: 5.2.3.
- Ruff: 0.15.14.
- uv: 0.10.11.

## Timing Summary

Times are pytest-benchmark means in microseconds. Each benchmark used
`rounds=5` and `iterations=10`.

| Run | Reference mean | Prepared mean | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 404.672 us | 213.317 us | 1.90x |
| 2 | 400.324 us | 204.975 us | 1.95x |
| 3 | 400.303 us | 205.576 us | 1.95x |

Allowed wording:

> On Darwin 24.6.0 arm64 with Apple M4, for scheduled `evaluate_residuals` on
> `chain_dynamics_case(horizon=100)`, the prepared KernelPlan backend measured
> 51% to 53% of the Python reference backend mean runtime using
> `benchmarks/test_scheduler_backend_benchmark.py` (reference/prepared ratios
> 1.90x to 1.95x). This is a narrow scheduled-stage result, not a solver,
> Jacobian, KKT, sensitivity, or package-wide speed claim.
