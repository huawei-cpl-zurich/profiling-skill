"""Functional contracts for the actor/analyzer evidence boundary."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "actor_analyzer_contract", Path(__file__).resolve().parents[1]
    / "scripts/actor_analyzer_contract.py")
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def report():
    return {
        "candidate_commit": "a" * 40,
        "profiling": {"status": "acquired"},
        "evidence": [{"path": "evidence/pipes.json", "kind": "profile",
                      "kernels": ["gather"], "cases": ["strided"]}],
        "observations": [{"finding": "MTE2 activity is high, bandwidth is low",
                          "evidence": ["evidence/pipes.json"], "confidence": "high"}],
        "bottlenecks": [{"finding": "Likely inefficient memory transactions",
                         "evidence": ["evidence/pipes.json"], "confidence": "medium"}],
        "ceilings": [{"resource": "memory", "status": "estimated",
                      "basis": "Product bandwidth with useful workload bytes",
                      "assumptions": ["Useful bytes exclude excess transactions"],
                      "lower_bound_us": 2.0, "measured_us": 10.0}],
        "unknowns": ["Actual transferred bytes are not established"],
    }


def test_valid_analysis_preserves_attribution_and_uncertainty():
    value = report()
    assert contract.validate_analysis(value, "a" * 40) is value


@pytest.mark.parametrize("change", [
    lambda r: r.update(candidate_commit="b" * 40),
    lambda r: r["evidence"][0].update(path="../private/profile.json"),
    lambda r: r["evidence"][0].update(kernels=[]),
    lambda r: r["observations"][0].update(evidence=["missing.json"]),
    lambda r: r["bottlenecks"][0].update(confidence="certain"),
    lambda r: r["ceilings"][0].update(lower_bound_us=float("nan")),
    lambda r: r["ceilings"][0].update(assumptions=[]),
    lambda r: r.update(unknowns="none"),
    lambda r: r["profiling"].update(status="not_attempted"),
])
def test_rejects_misattribution_or_unsupported_claims(change):
    value = report()
    change(value)
    with pytest.raises(ValueError):
        contract.validate_analysis(value, "a" * 40)


def test_unavailable_profile_and_unknown_ceiling_remain_honest_reports():
    value = report()
    value.update(profiling={"status": "unavailable", "reason": "Retained job timed out"},
                 evidence=[], observations=[], bottlenecks=[])
    value["ceilings"] = [{"resource": "dependency chain", "status": "not_established",
                          "basis": "No defensible peak for this recurrence",
                          "assumptions": []}]
    assert contract.validate_analysis(value, "a" * 40) is value
    value["profiling"].pop("reason")
    with pytest.raises(ValueError):
        contract.validate_analysis(value, "a" * 40)


def test_unknown_ceiling_cannot_smuggle_in_a_numeric_estimate():
    value = report()
    value["ceilings"][0].update(status="not_established")
    with pytest.raises(ValueError):
        contract.validate_analysis(value, "a" * 40)


def test_acquired_requires_real_profile_references_not_only_timing():
    value = report()
    value["evidence"][0]["kind"] = "timing"
    with pytest.raises(ValueError):
        contract.validate_analysis(value, "a" * 40)


@pytest.mark.parametrize("change", [
    lambda r: r.update(evidence="file.json"),
    lambda r: r["evidence"].append(dict(r["evidence"][0])),
    lambda r: r["evidence"][0].update(path="/private/profile.json"),
    lambda r: r["evidence"][0].update(cases=[]),
    lambda r: r["evidence"][0].update(kind="invented"),
    lambda r: r["observations"][0].update(finding=""),
    lambda r: r.update(profiling={"status": "unavailable", "reason": "offline"}),
    lambda r: r["ceilings"][0].update(measured_us=-1),
    lambda r: r["ceilings"][0].update(basis=""),
])
def test_malformed_or_contradictory_reports_are_rejected(change):
    value = report()
    change(value)
    with pytest.raises(ValueError):
        contract.validate_analysis(value, "a" * 40)


def test_extra_provenance_is_preserved_not_discarded():
    value = report()
    value["remote_handle"] = "retained handle"
    assert contract.validate_analysis(value, "a" * 40)["remote_handle"] == "retained handle"


def test_roofline_uses_units_and_does_not_hide_inconsistent_assumptions():
    result = contract.roofline(2e9, 1e6, 1e15, 1e12, measured_us=1)
    assert result == {"compute_limit_us": 2.0, "memory_limit_us": 1.0,
                      "lower_bound_us": 2.0, "ideal_fraction": 2.0}


@pytest.mark.parametrize("inputs", [(-1, 1, 1, 1), (1, 1, 0, 1),
                                    (1, float("inf"), 1, 1), (True, 1, 1, 1)])
def test_roofline_rejects_invalid_physical_inputs(inputs):
    with pytest.raises(ValueError):
        contract.roofline(*inputs)


def test_prompt_renderer_formats_round_budget_task_and_frozen_revision():
    task = "Optimize foo{bar}; preserve its public API."
    actor = contract.build_prompt("actor", 1200, 2, task)
    analyzer = contract.build_prompt("analyzer", 600, 2, task, "a" * 40)
    assert task in actor and task in analyzer
    assert "1200" in actor and "600" in analyzer
    assert "a" * 40 in analyzer


@pytest.mark.parametrize("arguments", [
    ("unknown", 600, 1, "task"), ("actor", 0, 1, "task"),
    ("actor", True, 1, "task"), ("actor", 600, 0, "task"),
    ("actor", 600, 1, ""), ("analyzer", 600, 1, "task"),
    ("actor", 600, 1, "task", "not-a-commit"),
])
def test_prompt_renderer_rejects_invalid_invocations(arguments):
    with pytest.raises(ValueError):
        contract.build_prompt(*arguments)
