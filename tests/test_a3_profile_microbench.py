from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmarks" / "a3_profile_microbench.py"
MANIFEST = ROOT / "experiments" / "a3-profile-hypotheses.json"


def load_module():
    spec = importlib.util.spec_from_file_location("a3_profile_microbench", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], text=True, capture_output=True, check=False
    )


def test_manifest_has_ten_hypotheses_and_six_timeline_controls():
    payload = json.loads(MANIFEST.read_text())
    assert payload["schema"] == "profiling-skill/a3-profile-hypotheses/v1"
    assert len(payload["hypotheses"]) == 10
    assert {item["id"] for item in payload["timeline_probes"]} == {
        "ordered-phases",
        "reversed-phases",
        "repeated-phases",
        "duration-scaled-phases",
        "serialized-vs-overlapped",
        "fused-vs-split",
    }
    all_items = payload["hypotheses"] + payload["timeline_probes"]
    for item in all_items:
        assert set(item["variants"]) == {"before", "after"}
        assert item["expected_evidence"]
        assert item["falsifier"]
        assert item["sweeps"]


def test_contract_is_deterministic_and_enumerates_every_manifest_experiment():
    first = run_cli("--contract")
    second = run_cli("--contract")
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    contract = json.loads(first.stdout)
    manifest = json.loads(MANIFEST.read_text())
    expected = [x["id"] for x in manifest["hypotheses"] + manifest["timeline_probes"]]
    assert [x["id"] for x in contract["experiments"]] == expected
    assert contract["seed"] == 20261013
    assert contract["variants"] == ["before", "after"]


def test_deterministic_inputs_are_identical_for_pair_and_change_with_sweep():
    module = load_module()
    before = module.generate_inputs("loop-bound", "short-local")
    after = module.generate_inputs("loop-bound", "short-local")
    longer = module.generate_inputs("loop-bound", "long-local")
    assert before.keys() == after.keys()
    assert all(before[key].equal(after[key]) for key in before)
    assert any(
        before[key].shape != longer[key].shape or not before[key].equal(longer[key])
        for key in before.keys() & longer.keys()
    )


def test_pair_runner_checks_equivalence_and_emits_profile_contract(monkeypatch):
    module = load_module()

    def fake_run(experiment, variant, sweep, launches):
        return {
            "status": "success",
            "variant": variant,
            "output": [1.0, 2.0],
            "kernel_names": [f"probe_{variant}"],
            "host_observed_us": [10.0] * launches,
        }

    monkeypatch.setattr(module, "run_variant", fake_run)
    payload = module.run_pair("loop-bound", "short-local", 2)
    assert payload["status"] == "success"
    assert payload["equivalent"] is True
    assert payload["expected_evidence"]
    assert payload["falsifier"]
    assert payload["profile"]["mode"] == "msprof-op"
    assert payload["profile"]["kernel_names"] == ["probe_before", "probe_after"]


def test_pair_runner_preserves_unsupported_as_inconclusive(monkeypatch):
    module = load_module()

    def fake_run(experiment, variant, sweep, launches):
        if variant == "before":
            return {"status": "unsupported", "reason": "compiler has no primitive"}
        return {"status": "success", "variant": variant, "output": [1.0]}

    monkeypatch.setattr(module, "run_variant", fake_run)
    payload = module.run_pair("row-reduction-vs-dot", "square", 1)
    assert payload["status"] == "inconclusive"
    assert payload["equivalent"] is None
    assert payload["results"]["before"]["reason"] == "compiler has no primitive"


def test_cli_rejects_ambiguous_selection_and_unknown_sweep():
    assert run_cli().returncode == 2
    assert run_cli("--contract", "--experiment", "loop-bound").returncode == 2
    result = run_cli("--experiment", "loop-bound", "--sweep", "not-a-sweep")
    assert result.returncode == 2


def test_cli_pair_writes_machine_readable_result(monkeypatch, tmp_path, capsys):
    module = load_module()
    monkeypatch.setattr(
        module,
        "run_pair",
        lambda experiment, sweep, launches: {
            "schema": "profiling-skill/a3-profile-result/v1",
            "status": "success",
            "experiment": experiment,
            "sweep": sweep,
            "launches": launches,
        },
    )
    output = tmp_path / "result.json"
    assert module.main([
        "--experiment", "loop-bound", "--sweep", "short-local",
        "--pair", "--launches", "2", "--json-output", str(output),
    ]) == 0
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)


def test_reference_models_make_before_after_outputs_equivalent():
    module = load_module()
    for experiment in module.experiments():
        sweep = experiment["sweeps"][0]["id"]
        inputs = module.generate_inputs(experiment["id"], sweep)
        before = module.reference_output(experiment["id"], "before", inputs)
        after = module.reference_output(experiment["id"], "after", inputs)
        assert module.outputs_close(before, after), experiment["id"]
