"""Backend interfaces and reference execution."""

from tinynlp.backends.kernel_plan import (
    KernelPlan,
    KernelPlanStats,
    KernelStep,
    build_kernel_plan,
    format_kernel_plan,
)
from tinynlp.backends.prepared import (
    PreparedKernel,
    PreparedKernelBackend,
    PreparedKernelConstant,
    PreparedKernelStep,
    PreparedKernelVariable,
    prepare_kernel,
)
from tinynlp.backends.protocol import Backend
from tinynlp.backends.reference import EvaluationError, PythonReferenceBackend, evaluate
from tinynlp.backends.registry import DEFAULT_BACKEND, get_backend, register_backend

__all__ = [
    "Backend",
    "DEFAULT_BACKEND",
    "EvaluationError",
    "KernelPlan",
    "KernelPlanStats",
    "KernelStep",
    "PreparedKernel",
    "PreparedKernelBackend",
    "PreparedKernelConstant",
    "PreparedKernelStep",
    "PreparedKernelVariable",
    "PythonReferenceBackend",
    "build_kernel_plan",
    "evaluate",
    "format_kernel_plan",
    "get_backend",
    "prepare_kernel",
    "register_backend",
]
