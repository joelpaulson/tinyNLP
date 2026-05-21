"""Backend protocol for executing visible kernel plans."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from tinynlp.backends.kernel_plan import KernelPlan


class Backend(Protocol):
    """Small interface implemented by expression execution backends."""

    name: str

    def execute(self, plan: KernelPlan, values: Mapping[str, float]) -> float:
        """Execute a kernel plan with explicit runtime values."""
