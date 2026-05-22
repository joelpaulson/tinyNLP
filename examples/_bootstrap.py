"""Helpers for running examples directly from a source checkout."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_src_on_path() -> None:
    """Make ``src/`` importable when an example is run by file path."""

    repo_src = Path(__file__).resolve().parents[1] / "src"
    if repo_src.exists() and str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
