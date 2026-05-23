import subprocess
import sys
from pathlib import Path


def test_flagship_least_squares_trace_example_runs_from_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "flagship_least_squares_trace.py"

    result = subprocess.run(
        [sys.executable, str(module_path)],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "flagship_least_squares" in result.stdout
    assert "status=converged" in result.stdout
    assert "success=True" in result.stdout
    assert "dimensions=variables=7 residuals=3" in result.stdout
    assert "LeastSquaresTrace" in result.stdout
    assert "NormalEquationSystem" in result.stdout
    assert "residual_norm=" in result.stdout
    assert "least_squares_value=" in result.stdout
    assert "accepted_step_length=" in result.stdout
    assert "object at" not in result.stdout
