from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "build_profile_reverse_report.py"
PLANNED_HYPOTHESES = ROOT / "references" / "a3-profiling-hypotheses.json"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = {
        "schema_version": 1,
        "campaign_id": "fixture-campaign",
        "starter_candidate_id": "starter",
        "references": [
            {
                "query_id": "8dfb07b37dc34532974f00418ea65306",
                "citation": "ref://profiling-skill/common/profiling/repos/ascend-msprof-pipe-memory-timeline-semantics",
            }
        ],
        "candidates": [
            {
                "candidate_id": "starter",
                "treatment": "starter",
                "round": 0,
                "option": 0,
                "source_hash": "aaa",
                "parent_candidate_id": None,
                "duplicate_of": None,
                "source_features": {"loop_bound": "total_k"},
                "code_delta": ["frozen starter"],
            },
            {
                "candidate_id": "fast",
                "treatment": "guarded-new",
                "round": 1,
                "option": 2,
                "source_hash": "bbb",
                "parent_candidate_id": "starter",
                "duplicate_of": None,
                "source_features": {"loop_bound": "sequence_k"},
                "code_delta": ["bound the loop by sequence length"],
            },
            {
                "candidate_id": "fast-copy",
                "treatment": "guarded-new",
                "round": 2,
                "option": 1,
                "source_hash": "bbb",
                "parent_candidate_id": "fast",
                "duplicate_of": "fast",
                "source_features": {"loop_bound": "sequence_k"},
                "code_delta": ["no source change"],
            },
            {
                "candidate_id": "compile-fail",
                "treatment": "guarded-old",
                "round": 2,
                "option": 9,
                "source_hash": "ccc",
                "parent_candidate_id": "starter",
                "duplicate_of": None,
                "source_features": {"block_n": 512},
                "code_delta": ["increase block size"],
            },
            {
                "candidate_id": "cannbot-incomplete",
                "treatment": "cannbot-only",
                "round": 1,
                "option": 0,
                "source_hash": None,
                "parent_candidate_id": "starter",
                "duplicate_of": None,
                "lineage_state": "incomplete",
                "lineage_note": "attempt directory exists but produced source is missing",
                "source_features": {},
                "code_delta": [],
            },
        ],
    }
    captures = {
        "schema_version": 1,
        "captures": [
            {"candidate_id": "starter", "case_id": "43", "status": "valid", "latency_us": 100.0, "profile": {"metrics": {"aiv_vector": 0.7}, "saturation": {"state": "unknown", "reason": "no validated denominator"}}},
            {"candidate_id": "starter", "case_id": "43", "status": "valid", "latency_us": 102.0},
            {"candidate_id": "starter", "case_id": "44", "status": "valid", "latency_us": 200.0},
            {"candidate_id": "fast", "case_id": "43", "status": "valid", "latency_us": 50.0, "profile": {"metrics": {"aiv_vector": 0.5}}},
            {"candidate_id": "fast", "case_id": "43", "status": "valid", "latency_us": 52.0},
            {"candidate_id": "fast", "case_id": "44", "status": "valid", "latency_us": 100.0},
            {"candidate_id": "compile-fail", "case_id": "43", "status": "candidate_failure", "failure": {"kind": "compilation", "message": "resource limit"}},
            {"candidate_id": "compile-fail", "case_id": "44", "status": "infrastructure_failure", "failure": {"kind": "transport", "message": "observer timeout"}},
        ],
    }
    hypotheses = {
        "schema_version": 1,
        "results": [
            {
                "hypothesis_id": "local-loop-bound",
                "title": "Local loop bounds remove control work",
                "expected": {"direction": "decrease", "minimum_relative_change": 0.1},
                "measurements": [{"status": "valid", "before": 100.0, "after": 70.0}],
                "signals": [
                    {"name": "scalar_activity", "relationship": "tracks_outcome", "summary": "fell with latency"},
                    {"name": "mte_activity", "relationship": "context_only", "summary": "changed without capacity semantics"},
                ],
            },
            {
                "hypothesis_id": "large-tile",
                "title": "A very large tile is faster",
                "expected": {"direction": "decrease", "minimum_relative_change": 0.1},
                "measurements": [{"status": "valid", "before": 100.0, "after": 125.0}],
                "signals": [{"name": "activity_ratio", "relationship": "contradicts_outcome", "summary": "looked better while latency regressed"}],
            },
            {
                "hypothesis_id": "schedule",
                "title": "Schedule choice has a large effect",
                "expected": {"direction": "decrease", "minimum_relative_change": 0.1},
                "measurements": [{"status": "valid", "before": 100.0, "after": 97.0}],
                "signals": [],
            },
            {
                "hypothesis_id": "timeline-order",
                "title": "Timeline exposes phase order",
                "expected": {"direction": "increase", "minimum_relative_change": 0.1},
                "measurements": [{"status": "infrastructure_failure"}],
                "signals": [{"name": "timeline_phase_order", "relationship": "not_exported", "summary": "not present"}],
            },
        ],
    }
    paths = tuple(tmp_path / name for name in ("manifest.json", "captures.json", "hypotheses.json"))
    for path, value in zip(paths, (manifest, captures, hypotheses)):
        write_json(path, value)
    return paths  # type: ignore[return-value]


def test_builds_joined_outputs_and_preserves_failures_and_duplicates(tmp_path: Path):
    manifest, captures, hypotheses = fixtures(tmp_path)
    output = tmp_path / "report"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--captures", str(captures), "--hypotheses", str(hypotheses), "--output-dir", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr

    report = json.loads((output / "report.json").read_text())
    by_id = {item["candidate_id"]: item for item in report["candidates"]}
    assert by_id["fast"]["cases"]["43"]["median_latency_us"] == 51.0
    assert by_id["fast"]["cases"]["43"]["latency_samples_us"] == [50.0, 52.0]
    assert by_id["fast"]["cases"]["43"]["profile_metrics"] == {"aiv_vector": 0.5}
    assert by_id["fast"]["cases"]["43"]["starter_speedup"] == 101.0 / 51.0
    assert by_id["fast"]["geometric_mean"]["starter_speedup"] > 1.98
    assert by_id["fast"]["geometric_mean"]["parent_speedup"] > 1.98
    assert by_id["fast-copy"]["duplicate_of"] == "fast"
    assert by_id["fast-copy"]["evidence_source_candidate_id"] == "fast"
    assert by_id["compile-fail"]["failures"] == [
        {"case_id": "43", "status": "candidate_failure", "kind": "compilation", "message": "resource limit"},
        {"case_id": "44", "status": "infrastructure_failure", "kind": "transport", "message": "observer timeout"},
    ]
    assert by_id["compile-fail"]["geometric_mean"]["starter_speedup"] is None
    assert by_id["cannbot-incomplete"]["lineage_state"] == "incomplete"
    assert by_id["cannbot-incomplete"]["lineage_note"] == "attempt directory exists but produced source is missing"
    assert by_id["cannbot-incomplete"]["cases"] == {}
    assert by_id["fast"]["saturation"] == {
        "state": "unknown",
        "reason": "no validated capacity evidence",
    }

    verdicts = {item["hypothesis_id"]: item["classification"] for item in report["hypotheses"]}
    assert verdicts == {
        "local-loop-bound": "supported",
        "large-tile": "refuted",
        "schedule": "inconclusive",
        "timeline-order": "unavailable",
    }
    signals = {item["signal"]: item["classification"] for item in report["signals"]}
    assert signals == {
        "activity_ratio": "misleading",
        "mte_activity": "contextual",
        "scalar_activity": "useful",
        "timeline_phase_order": "unavailable",
    }

    markdown = (output / "report.md").read_text()
    assert "Reproducible scaffold" in markdown
    assert "bound the loop by sequence length" in markdown
    assert "ref://profiling-skill/common/profiling/repos/ascend-msprof-pipe-memory-timeline-semantics" in markdown
    assert "Saturation: unknown" in markdown
    assert "unavailablex" not in markdown

    with (output / "candidates.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["candidate_id"] for row in rows] == [
        "cannbot-incomplete", "compile-fail", "fast", "fast-copy", "starter"
    ]


def test_rejects_unknown_candidate_and_does_not_infer_saturation(tmp_path: Path):
    manifest, captures, hypotheses = fixtures(tmp_path)
    payload = json.loads(captures.read_text())
    payload["captures"].append({"candidate_id": "missing", "case_id": "43", "status": "valid", "latency_us": 1})
    write_json(captures, payload)
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--captures", str(captures), "--hypotheses", str(hypotheses), "--output-dir", str(tmp_path / "out")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert "unknown candidate" in run.stderr


def test_rejects_unvalidated_saturation_claim(tmp_path: Path):
    manifest, captures, hypotheses = fixtures(tmp_path)
    payload = json.loads(captures.read_text())
    payload["captures"][0]["profile"]["saturation"] = {"state": "saturated", "reason": "ratio was high"}
    write_json(captures, payload)
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--captures", str(captures), "--hypotheses", str(hypotheses), "--output-dir", str(tmp_path / "out")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 2
    assert "validated capacity evidence" in run.stderr


def test_checked_in_hypotheses_are_valid_unavailable_scaffold(tmp_path: Path):
    manifest, captures, _ = fixtures(tmp_path)
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--captures", str(captures), "--hypotheses", str(PLANNED_HYPOTHESES), "--output-dir", str(tmp_path / "planned")],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads((tmp_path / "planned" / "report.json").read_text())
    assert len(report["hypotheses"]) == 11
    assert {item["classification"] for item in report["hypotheses"]} == {"unavailable"}
