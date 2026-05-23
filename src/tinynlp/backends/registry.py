"""Small backend registry."""

from __future__ import annotations

from tinynlp.backends.protocol import Backend

DEFAULT_BACKEND = "python"
_BACKENDS: dict[str, Backend] = {}


def register_backend(backend: Backend) -> None:
    """Register a backend by name."""

    if not backend.name:
        msg = "backend name must be non-empty"
        raise ValueError(msg)
    _BACKENDS[backend.name] = backend


def get_backend(name: str = DEFAULT_BACKEND) -> Backend:
    """Return a registered backend by name."""

    _ensure_default_backend()
    try:
        return _BACKENDS[name]
    except KeyError as exc:
        available = ", ".join(sorted(_BACKENDS)) or "<none>"
        msg = f"unknown backend {name!r}; available backends: {available}"
        raise KeyError(msg) from exc


def _ensure_default_backend() -> None:
    if DEFAULT_BACKEND not in _BACKENDS:
        from tinynlp.backends.reference import PythonReferenceBackend

        register_backend(PythonReferenceBackend())
    if "prepared-python" not in _BACKENDS:
        from tinynlp.backends.prepared import PreparedKernelBackend

        register_backend(PreparedKernelBackend())
