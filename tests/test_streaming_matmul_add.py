import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "streaming_matmul_add.py"


def load_module():
    spec = importlib.util.spec_from_file_location("streaming_matmul_add", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*arguments: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def test_contract_is_deterministic_and_has_fixed_case_sets(tmp_path: Path):
    first = run_cli("--contract")
    output = tmp_path / "contract.json"
    second = run_cli("--contract", "--json-output", str(output))
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout == output.read_text()
    contract = json.loads(first.stdout)
    correctness = [case for case in contract["cases"] if case["kind"] == "correctness"]
    performance = [case for case in contract["cases"] if case["kind"] == "performance"]
    assert len(correctness) == 7
    assert [case["name"] for case in performance] == [
        "performance-small",
        "performance-medium",
        "performance-large",
    ]
    assert contract["kernel_name"] == "streaming_matmul_add_kernel_mix_aic"
    assert contract["seed"] == 20260925
    assert contract["tolerances"] == {"atol": 0.02, "rtol": 0.02}


def test_cases_cover_dimension_tails_and_multiple_k_tiles():
    module = load_module()
    correctness = [case for case in module.CASES if case.kind == "correctness"]
    assert any(case.m % 32 for case in correctness)
    assert any(case.n % 32 for case in correctness)
    assert any(case.k % 32 for case in correctness)
    assert any(case.k > 32 for case in correctness)
    assert len({(case.m, case.n, case.k) for case in correctness}) == 7


def test_exception_classification_is_machine_readable():
    module = load_module()

    class CompilerError(Exception):
        pass

    assert module.classify_exception(CompilerError("bad IR")) == "compilation"
    assert module.classify_exception(RuntimeError("device launch failed")) == "runtime"


def test_kernel_definition_resolves_triton_annotations_from_module_globals(monkeypatch):
    fake_triton = types.ModuleType("triton")
    fake_language = types.ModuleType("triton.language")
    fake_language.constexpr = object()
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_language)

    module = load_module()

    assert module.tl is fake_language
    assert module.streaming_matmul_add_kernel.__name__ == module.PYTHON_KERNEL_NAME


def test_cli_rejects_ambiguous_or_invalid_invocations():
    assert run_cli().returncode == 2
    assert run_cli("--contract", "--case", "correctness-00-tiny").returncode == 2
    assert run_cli("--case", "correctness-00-tiny", "--launches", "0").returncode == 2


def test_cli_serializes_compilation_failure(monkeypatch, capsys):
    module = load_module()

    class CompilationError(Exception):
        pass

    def fail(_case, _launches):
        raise CompilationError("candidate did not compile")

    monkeypatch.setattr(module, "run", fail)
    assert module.main(["--case", "correctness-00-tiny"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failure"
    assert payload["failure"] == {
        "kind": "compilation",
        "message": "candidate did not compile",
    }


def test_cli_persists_same_failure_payload(monkeypatch, capsys, tmp_path: Path):
    module = load_module()

    def fail(_case, _launches):
        return {
            "status": "failure",
            "failure": {"kind": "correctness", "message": "mismatch"},
        }

    monkeypatch.setattr(module, "run", fail)
    output = tmp_path / "result.json"
    assert module.main([
        "--case", "correctness-01-square-tail", "--json-output", str(output)
    ]) == 1
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())


def test_non_finite_error_metrics_are_strict_json(monkeypatch, capsys):
    module = load_module()

    def fail(_case, _launches):
        return {
            "status": "failure",
            "failure": {
                "kind": "correctness",
                **module.error_metrics(float("nan"), float("inf")),
            },
        }

    monkeypatch.setattr(module, "run", fail)
    assert module.main(["--case", "correctness-00-tiny"]) == 1
    encoded = capsys.readouterr().out
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert json.loads(encoded)["failure"] == {
        "kind": "correctness",
        "max_abs_error": None,
        "max_rel_error": None,
        "non_finite": ["max_abs_error", "max_rel_error"],
    }


def test_process_control_exceptions_propagate(monkeypatch):
    module = load_module()

    def interrupt(_case, _launches):
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "run", interrupt)
    try:
        module.main(["--case", "correctness-00-tiny"])
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("KeyboardInterrupt was converted into a candidate failure")
