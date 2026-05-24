import importlib.util
import sys
from pathlib import Path


def test_flagship_start_here_report_is_deterministic_and_honest() -> None:
    example = _load_example("flagship_start_here")

    report = example.flagship_start_here_report()

    assert report == example.flagship_start_here_report()
    assert "FlagshipStartHere" in report
    assert "problem=flagship_chain" in report
    assert "dimensions=variables=7 residuals=3" in report
    assert "residual_blocks=[chain_dynamics]" in report
    assert "residual_values=[0.115, 0.11595, 0.1168]" in report
    assert "jacobian_shape=(3, 7)" in report
    assert "jacobian_entries=9" in report
    assert "backend=prepared-python" in report
    assert "stage_group=residual_plus_jacobian" in report
    assert "validation_passed=True" in report
    assert "residual_max_abs_error=0" in report
    assert "jacobian_max_abs_error=0" in report
    assert "least_squares_reference:" in report
    assert "status=converged" in report
    assert "initial_residual_norm=0.200778" in report
    assert "final_residual_norm=2.08517e-10" in report
    assert "casadi_correctness:" in report
    assert "purpose=correctness_only" in report
    assert "benchmark_evidence:" in report
    assert "benchmarks/results/scheduled_residual_evaluation.md" in report
    assert "benchmarks/results/flagship_residual_jacobian_evaluation.md" in report
    assert "benchmarks/test_scheduler_residual_jacobian_benchmark.py" in report
    assert "next_step:" in report
    assert "write the next roadmap" in report
    assert "object at" not in report

    forbidden = [
        "package-wide speed claim",
        "solver-speed claim",
        "CasADi performance baseline",
        "IPOPT comparison",
        "GPU speed",
        "production solver",
    ]
    for phrase in forbidden:
        assert phrase not in report


def _load_example(name: str):
    examples_path = Path(__file__).resolve().parents[1] / "examples"
    module_path = examples_path / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
