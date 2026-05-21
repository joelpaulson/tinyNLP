# DECISIONS

This file records architectural decisions that should shape tinyNLP as it grows.

## 0001: CPU-First Execution

- Status: accepted
- Date: 2026-05-21

tinyNLP starts as a CPU-first project. This keeps the initial execution path
inspectable and makes it easier to validate symbolic structure, numeric values,
assembly, and solver behavior before adding hardware-specific complexity.

## 0002: Inspectable Core

- Status: accepted
- Date: 2026-05-21

The core should stay small enough that contributors can trace how an operation
moves through the pipeline. Abstractions should be added only when they make the
execution path clearer or remove meaningful duplication.

## 0003: Separate Symbolic Structure From Numeric Values

- Status: accepted
- Date: 2026-05-21

Symbolic structure and numeric values should be represented separately. This
lets tinyNLP inspect sparsity, derivative structure, and assembly plans without
requiring every numeric value to be present.

## 0004: uv, Ruff, pytest, and Permissive mypy

- Status: accepted
- Date: 2026-05-21

The repository uses uv for environment management, Ruff for linting and
formatting, pytest for tests, pytest-benchmark for future benchmark work, and a
permissive mypy configuration that can become stricter over time.
