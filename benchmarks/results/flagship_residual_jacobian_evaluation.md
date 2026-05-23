# Flagship Residual+Jacobian Evaluation Benchmark

This result summary is limited to one scheduled stage group: prepared
residual+Jacobian evaluation on `flagship_chain_case(horizon=100)`.

It is not a solver, CasADi, KKT, sensitivity, or package-wide performance
claim.

## Command

```sh
uv run pytest benchmarks/test_scheduler_residual_jacobian_benchmark.py --benchmark-json /private/tmp/tinynlp-flagship-rj-benchmark-1.json
uv run pytest benchmarks/test_scheduler_residual_jacobian_benchmark.py --benchmark-json /private/tmp/tinynlp-flagship-rj-benchmark-2.json
uv run pytest benchmarks/test_scheduler_residual_jacobian_benchmark.py --benchmark-json /private/tmp/tinynlp-flagship-rj-benchmark-3.json
```

## Code

- Benchmark source: `benchmarks/test_scheduler_residual_jacobian_benchmark.py`.
- Runtime path source commit: `c6fdca3`
  (`Add scheduler-backed residual and Jacobian execution`).
- Result-summary commit: this F4 summary is committed as
  `Add flagship benchmark and correctness evidence`.
- Measured stage group: scheduled residual+Jacobian evaluation.
- Problem: `flagship_chain_case(horizon=100)`.
- Variables: 201.
- Residuals: 100.
- Jacobian coordinate entries: 300.
- Reference baseline: `assemble_residuals(contract, values)` plus
  `assemble_jacobian(contract, values)` using a cached `AssemblyContract`.
- Prepared path: `ScheduledResidualJacobianEvaluator.evaluate(values)` using
  prepared residual and Jacobian `KernelPlan`s with the `prepared-python`
  backend.
- Output validation: passed before timing in every run, with residual and
  Jacobian outputs checked again after timing.

## Environment

- OS: Darwin 24.6.0 arm64.
- Platform: macOS-15.6.1-arm64-arm-64bit.
- CPU: Apple M4.
- Python: 3.12.13.
- tinyNLP: 0.0.0.
- pytest: 9.0.3.
- pytest-benchmark: 5.2.3.
- uv: 0.10.11.

## Timing Summary

Times are pytest-benchmark means in microseconds. Each benchmark used
`rounds=5` and `iterations=10`.

| Run | Reference mean | Prepared mean | Ratio |
| --- | ---: | ---: | ---: |
| 1 | 1138.971 us | 788.639 us | 1.44x |
| 2 | 1137.473 us | 803.072 us | 1.42x |
| 3 | 1145.613 us | 784.914 us | 1.46x |

Allowed wording:

> On Darwin 24.6.0 arm64 with Apple M4, for scheduled residual+Jacobian
> evaluation on `flagship_chain_case(horizon=100)`, the prepared
> scheduler-backed path measured 68.5% to 70.6% of the Python reference
> assembly path mean runtime using
> `benchmarks/test_scheduler_residual_jacobian_benchmark.py`
> (reference/prepared ratios 1.42x to 1.46x). This is a narrow scheduled-stage
> result, not a solver, CasADi, KKT, sensitivity, or package-wide speed claim.
