from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "analyze_pipe_saturation.py"


def model(product: str = "a5") -> dict:
    return {
        "schema_version": 1,
        "model_id": f"fixture-{product}-pipes-v1",
        "product": product,
        "provenance": {
            "source": "functional-test-fixture",
            "revision": "fixture-revision-1",
        },
        "resources": [
            {
                "resource": "cube",
                "numerator_metric": "cube_busy_cycles",
                "denominator_metric": "cube_available_cycles",
                "saturation_threshold": 0.8,
                "capacity_validation": {
                    "state": "validated",
                    "source": "fixture://cube-capacity",
                },
            },
            {
                "resource": "mte2",
                "numerator_metric": "mte2_busy_cycles",
                "denominator_metric": "mte2_available_cycles",
                "saturation_threshold": 0.75,
                "capacity_validation": {
                    "state": "validated",
                    "source": "fixture://mte2-capacity",
                },
            },
        ],
    }


def evidence(product: str = "a5", *, phases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "product": product,
        "provenance": {
            "capture_id": "capture-007",
            "source_sha256": "a" * 64,
        },
        "phases": phases,
    }


def phase(name: str, metrics: dict, **extra: object) -> dict:
    return {
        "phase_id": name,
        "start_ns": 10,
        "end_ns": 20,
        "metrics": metrics,
        **extra,
    }


def run_cli(tmp_path: Path, selected_model: dict, captured: dict):
    model_path = tmp_path / "model.json"
    evidence_path = tmp_path / "evidence.json"
    model_path.write_text(json.dumps(selected_model))
    evidence_path.write_text(json.dumps(captured))
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--product",
            captured["product"],
            "--model",
            str(model_path),
            "--evidence",
            str(evidence_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_cli_reports_phase_local_saturated_and_unsaturated_states(tmp_path: Path):
    captured = evidence(
        phases=[
            phase(
                "load",
                {
                    "cube_busy_cycles": 20,
                    "cube_available_cycles": 100,
                    "mte2_busy_cycles": 90,
                    "mte2_available_cycles": 100,
                },
            ),
            phase(
                "compute",
                {
                    "cube_busy_cycles": 90,
                    "cube_available_cycles": 100,
                    "mte2_busy_cycles": 20,
                    "mte2_available_cycles": 100,
                },
            ),
            phase(
                "tail",
                {
                    "cube_busy_cycles": 20,
                    "cube_available_cycles": 100,
                    "mte2_busy_cycles": 30,
                    "mte2_available_cycles": 100,
                },
            ),
        ]
    )

    first = run_cli(tmp_path, model(), captured)
    second = run_cli(tmp_path, model(), captured)

    assert first.returncode == 0, first.stderr
    assert first.stdout == second.stdout
    result = json.loads(first.stdout)
    assert result["schema_version"] == 1
    assert result["product"] == "a5"
    assert result["model"] == {
        "model_id": "fixture-a5-pipes-v1",
        "provenance": model()["provenance"],
    }
    assert result["evidence_provenance"] == captured["provenance"]
    assert [item["state"] for item in result["phases"]] == [
        "saturated",
        "saturated",
        "unsaturated",
    ]
    assert result["phases"][0]["saturated_resources"] == ["mte2"]
    assert result["phases"][1]["saturated_resources"] == ["cube"]
    assert result["phases"][0]["interval_ns"] == {"start": 10, "end": 20}


def test_missing_and_na_metrics_are_unknown_not_zero(tmp_path: Path):
    captured = evidence(
        phases=[
            phase(
                "missing",
                {
                    "cube_busy_cycles": None,
                    "cube_available_cycles": 100,
                    "mte2_busy_cycles": "N/A",
                    "mte2_available_cycles": 100,
                },
            )
        ]
    )

    run = run_cli(tmp_path, model(), captured)

    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["phases"][0]["state"] == "unknown"
    resources = result["phases"][0]["resources"]
    assert [item["state"] for item in resources] == ["unknown", "unknown"]
    assert all(item["ratio"] is None for item in resources)
    assert "missing" in resources[0]["reason"]
    assert "missing" in resources[1]["reason"]


def test_activity_and_composition_without_capacity_stay_unknown(tmp_path: Path):
    selected_model = model()
    selected_model["resources"] = [
        {
            "resource": "mte3",
            "numerator_metric": "mte3_busy_cycles",
            "denominator_metric": "mte3_available_cycles",
            "saturation_threshold": 0.8,
            "capacity_validation": {
                "state": "unvalidated",
                "source": "fixture://activity-only",
            },
        }
    ]
    captured = evidence(
        phases=[
            phase(
                "active",
                {},
                activity={"mte3": 1.0},
                composition={"mte3": 0.99},
            )
        ]
    )

    run = run_cli(tmp_path, selected_model, captured)

    assert run.returncode == 0, run.stderr
    resource = json.loads(run.stdout)["phases"][0]["resources"][0]
    assert resource["state"] == "unknown"
    assert resource["ratio"] is None
    assert "not validated" in resource["reason"]


def test_known_unsaturated_resource_does_not_hide_unknown_resource(tmp_path: Path):
    captured = evidence(
        phases=[
            phase(
                "partial",
                {
                    "cube_busy_cycles": 10,
                    "cube_available_cycles": 100,
                    "mte2_busy_cycles": "NA",
                    "mte2_available_cycles": 100,
                },
            )
        ]
    )

    run = run_cli(tmp_path, model(), captured)

    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["phases"][0]["state"] == "unknown"


def test_cli_rejects_incompatible_product_model(tmp_path: Path):
    captured = evidence(product="a3", phases=[phase("all", {})])

    run = run_cli(tmp_path, model(product="a5"), captured)

    assert run.returncode == 2
    assert run.stdout == ""
    assert "does not match model product" in run.stderr


def test_cli_rejects_non_positive_denominator(tmp_path: Path):
    captured = evidence(
        phases=[
            phase(
                "bad-capacity",
                {
                    "cube_busy_cycles": 1,
                    "cube_available_cycles": 0,
                    "mte2_busy_cycles": 1,
                    "mte2_available_cycles": 1,
                },
            )
        ]
    )

    run = run_cli(tmp_path, model(), captured)

    assert run.returncode == 0, run.stderr
    resource = json.loads(run.stdout)["phases"][0]["resources"][0]
    assert resource["state"] == "unknown"
    assert resource["ratio"] is None
    assert "positive" in resource["reason"]


def test_cli_rejects_invalid_phase_boundaries(tmp_path: Path):
    captured = evidence(
        phases=[phase("reversed", {}, start_ns=20, end_ns=10)]
    )

    run = run_cli(tmp_path, model(), captured)

    assert run.returncode == 2
    assert "end_ns must be greater than start_ns" in run.stderr

    captured = evidence(
        phases=[phase("float-ns", {}, start_ns=10.0, end_ns=11)]
    )
    run = run_cli(tmp_path, model(), captured)
    assert run.returncode == 2 and run.stdout == ""
    assert "start_ns must be a non-negative integer" in run.stderr


def test_cli_rejects_empty_provenance(tmp_path: Path):
    selected_model = model()
    selected_model["provenance"] = {}
    run = run_cli(tmp_path, selected_model, evidence(phases=[phase("all", {})]))
    assert run.returncode == 2
    assert "model.provenance must not be empty" in run.stderr

    captured = evidence(phases=[phase("all", {})])
    captured["provenance"] = {}
    run = run_cli(tmp_path, model(), captured)
    assert run.returncode == 2
    assert "evidence.provenance must not be empty" in run.stderr


def test_cli_rejects_malformed_provenance_and_boolean_schema(tmp_path: Path):
    selected_model = model()
    selected_model["provenance"] = {"source": "", "revision": "rev"}
    run = run_cli(tmp_path, selected_model, evidence(phases=[phase("all", {})]))
    assert run.returncode == 2 and run.stdout == ""
    assert "model.provenance.source must be a non-empty string" in run.stderr

    captured = evidence(phases=[phase("all", {})])
    captured["provenance"]["capture_id"] = " "
    run = run_cli(tmp_path, model(), captured)
    assert run.returncode == 2 and run.stdout == ""
    assert "evidence.provenance.capture_id must be a non-empty string" in run.stderr

    captured = evidence(phases=[phase("all", {})])
    captured["provenance"]["source_sha256"] = "not-a-sha256"
    run = run_cli(tmp_path, model(), captured)
    assert run.returncode == 2 and run.stdout == ""
    assert "evidence.provenance.source_sha256" in run.stderr

    selected_model = model()
    selected_model["schema_version"] = True
    run = run_cli(tmp_path, selected_model, evidence(phases=[phase("all", {})]))
    assert run.returncode == 2 and run.stdout == ""
    assert "model schema_version must be integer 1" in run.stderr

    captured = evidence(phases=[phase("all", {})])
    captured["schema_version"] = True
    run = run_cli(tmp_path, model(), captured)
    assert run.returncode == 2 and run.stdout == ""
    assert "evidence schema_version must be integer 1" in run.stderr


def test_cli_preserves_large_one_nanosecond_interval(tmp_path: Path):
    start = 2**53 + 1
    captured = evidence(
        phases=[phase("one-ns", {}, start_ns=start, end_ns=start + 1)]
    )

    run = run_cli(tmp_path, model(), captured)

    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["phases"][0]["interval_ns"] == {
        "start": start,
        "end": start + 1,
    }


def test_capacity_numerator_above_denominator_is_unknown(tmp_path: Path):
    captured = evidence(
        phases=[
            phase(
                "invalid-ratio",
                {
                    "cube_busy_cycles": 101,
                    "cube_available_cycles": 100,
                    "mte2_busy_cycles": 10,
                    "mte2_available_cycles": 100,
                },
            )
        ]
    )

    run = run_cli(tmp_path, model(), captured)

    assert run.returncode == 0, run.stderr
    cube = json.loads(run.stdout)["phases"][0]["resources"][0]
    assert cube["state"] == "unknown"
    assert cube["ratio"] is None
    assert "cannot exceed" in cube["reason"]
