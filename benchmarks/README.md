# Benchmarks

This directory holds benchmark sources for tinyNLP. There are no committed
benchmark result summaries yet, and this repository does not make speed claims.

## Current Scaffold

- `test_reference_evaluator_benchmark.py` measures expression/residual
  evaluation through the backend protocol and registered Python reference
  backend.
- The benchmark validates expected residual outputs before timing.
- The only baseline in this scaffold is the expected numeric output for the
  canonical examples, not another package, method, or optimized backend.

## Running Benchmarks

Use pytest-benchmark through uv:

```sh
uv run pytest benchmarks
```

## Result Summary Requirements

Do not add benchmark result summaries until there is a milestone that asks for
them. Future committed summaries must include:

- Benchmark source and command.
- Measured pipeline stage.
- Problem definition and expected output.
- Dependency versions.
- Machine and environment metadata.
- Result summary tied to committed code.

Do not make README performance claims without benchmark source, command,
environment metadata, and committed result summary.
