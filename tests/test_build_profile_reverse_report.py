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
    assert "Partial evidence" in markdown
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
    assert report["report_state"] == "partial"


def test_consumes_producer_manifest_and_nested_results_without_inventing_lineage(
    tmp_path: Path,
):
    source_a, source_b, source_c = "SOURCE_A", "SOURCE_B", "SOURCE_C"
    manifest = {
        "schema": "profiling-skill/bsa-candidate-manifest/v1",
        "campaign_root": "/retained/campaign",
        "environment": {"product": "a3", "runtime": "named"},
        "occurrences": [
            {"occurrence_id": "starter-occ", "execution_id": "exec-a", "treatment": "starter", "round": 0, "position": 0, "option_id": "starter", "recoverability": "recoverable", "source_sha256": "a", "is_starter": True},
            {"occurrence_id": "fast-occ", "execution_id": "exec-b", "treatment": "guarded", "round": 1, "position": 0, "option_id": "fast", "recoverability": "recoverable", "source_sha256": "b", "parent_occurrence_id": "starter-occ", "lineage_note": "request-declared incumbent", "parent_feature_deltas": {"loop_bound": {"before": "total_packed_k", "after": "per_sequence_k"}, "block_n": {"before": 64, "after": 128}}, "hypothesis": "local K bound"},
            {"occurrence_id": "fast-duplicate", "execution_id": "exec-b", "treatment": "guarded", "round": 2, "position": 1, "option_id": "same", "recoverability": "recoverable", "source_sha256": "b"},
            {"occurrence_id": "failed-occ", "execution_id": "exec-c", "treatment": "cannbot", "round": 1, "position": 0, "option_id": "bad", "recoverability": "recoverable", "source_sha256": "c"},
            {"occurrence_id": "lost-occ", "treatment": "cannbot", "round": 1, "position": 1, "option_id": "lost", "recoverability": "source_unavailable", "artifact_diagnostics": "sanitized artifact omitted source"},
        ],
        "executions": [
            {"execution_id": "exec-a", "source": source_a, "source_sha256": "a", "parameters": {"BLOCK_N": 64}, "features": {"loop_bound": "total_packed_k", "block_n": 64}, "occurrence_ids": ["starter-occ"]},
            {"execution_id": "exec-b", "source": source_b, "source_sha256": "b", "parameters": {"BLOCK_N": 128}, "features": {"loop_bound": "per_sequence_k", "block_n": 128}, "occurrence_ids": ["fast-occ", "fast-duplicate"]},
            {"execution_id": "exec-c", "source": source_c, "source_sha256": "c", "parameters": {"BLOCK_N": 512}, "features": {"loop_bound": "total_packed_k", "block_n": 512}, "occurrence_ids": ["failed-occ"]},
        ],
    }
    provenance = {"target": "bz-a3-2", "runtime": "named", "physical_device": 1, "command": ["run"], "handle": "remote:job-1"}
    results = {
        "schema": "profiling-skill/bsa-candidate-results/v1",
        "cases": [47],
        "executions": {
            "exec-a": {"status": "completed", "kernel_name": "kernel", "cases": {"47": {"status": "completed", "median_us": 101.0, "relative_range": 0.02, "accepted_sample_indexes": [0, 1], "samples": [
                {"duration_us": 100.0, "pipe_utilization": {"aiv_vec_ratio": 0.7}, "evidence_sha256": "1" * 64, "provenance": provenance},
                {"duration_us": 102.0, "pipe_utilization": {"aiv_vec_ratio": 0.72}, "evidence_sha256": "2" * 64, "provenance": provenance},
                {"duration_us": 150.0, "pipe_utilization": {"aiv_vec_ratio": 0.99}, "evidence_sha256": "9" * 64, "provenance": provenance},
            ]}}, "infrastructure_failures": []},
            "exec-b": {"status": "completed", "kernel_name": "kernel", "cases": {"47": {"status": "completed", "median_us": 50.0, "relative_range": 0.04, "accepted_sample_indexes": [0, 1], "samples": [
                {"duration_us": 49.0, "pipe_utilization": {"aiv_vec_ratio": 0.5}, "evidence_sha256": "3" * 64, "provenance": provenance},
                {"duration_us": 51.0, "pipe_utilization": {"aiv_vec_ratio": 0.52}, "evidence_sha256": "4" * 64, "provenance": provenance},
            ]}}, "infrastructure_failures": [{"kind": "transport", "diagnostics": "recovered observer"}]},
            "exec-c": {"status": "candidate_failure", "kernel_name": None, "cases": {"47": {"status": "pending", "samples": []}}, "failure": {"kind": "compile_error", "category": "candidate", "diagnostics": "resource limit"}, "infrastructure_failures": []},
        },
    }
    manifest_path, results_path, hypotheses_path = (
        tmp_path / "producer-manifest.json",
        tmp_path / "producer-results.json",
        tmp_path / "hypotheses.json",
    )
    write_json(manifest_path, manifest)
    write_json(results_path, results)
    (tmp_path / "legacy").mkdir()
    _, _, legacy_hypotheses = fixtures(tmp_path / "legacy")
    hypotheses_path.write_text(legacy_hypotheses.read_text())

    output = tmp_path / "producer-report"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest_path), "--results", str(results_path), "--hypotheses", str(hypotheses_path), "--output-dir", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    report = json.loads((output / "report.json").read_text())
    by_id = {item["candidate_id"]: item for item in report["candidates"]}
    assert set(by_id) == {"starter-occ", "fast-occ", "fast-duplicate", "failed-occ", "lost-occ"}
    assert by_id["fast-duplicate"]["duplicate_of"] == "fast-occ"
    assert by_id["fast-duplicate"]["evidence_source_candidate_id"] == "fast-occ"
    assert by_id["fast-occ"]["kernel_name"] == "kernel"
    assert by_id["fast-occ"]["source_features"] == {"loop_bound": "per_sequence_k", "block_n": 128}
    assert by_id["fast-occ"]["code_delta"] == {
        "loop_bound": {"before": "total_packed_k", "after": "per_sequence_k"},
        "block_n": {"before": 64, "after": 128},
    }
    assert by_id["fast-occ"]["hypothesis"] == "local K bound"
    assert by_id["fast-occ"]["lineage_note"] == "request-declared incumbent"
    assert by_id["failed-occ"]["lineage_state"] == "unlinked"
    assert by_id["fast-occ"]["cases"]["47"]["starter_speedup"] == 101.0 / 50.0
    assert by_id["fast-occ"]["cases"]["47"]["parent_speedup"] == 101.0 / 50.0
    assert by_id["fast-occ"]["cases"]["47"]["profile_metrics"] == {"aiv_vec_ratio": 0.51}
    assert by_id["starter-occ"]["cases"]["47"]["latency_samples_us"] == [100.0, 102.0]
    assert len(by_id["starter-occ"]["cases"]["47"]["samples"]) == 3
    assert by_id["starter-occ"]["cases"]["47"]["samples"][2]["accepted"] is False
    sample = by_id["fast-occ"]["cases"]["47"]["samples"][0]
    assert sample["evidence_sha256"] == "3" * 64
    assert sample["provenance"]["handle"] == "remote:job-1"
    assert by_id["fast-occ"]["infrastructure_failures"][0]["message"] == "recovered observer"
    assert by_id["fast-duplicate"]["infrastructure_failures"][0]["message"] == "recovered observer"
    assert by_id["failed-occ"]["failures"][0]["kind"] == "compile_error"
    assert by_id["lost-occ"]["lineage_state"] == "unrecoverable"
    assert by_id["lost-occ"]["parent_candidate_id"] is None
    assert by_id["lost-occ"]["geometric_mean"]["starter_speedup"] is None
    assert report["report_state"] == "partial"


def test_producer_schema_without_explicit_starter_or_parent_has_no_speedups(tmp_path: Path):
    manifest = {
        "schema": "profiling-skill/bsa-candidate-manifest/v1",
        "occurrences": [{"occurrence_id": "only", "execution_id": "exec", "recoverability": "recoverable"}],
        "executions": [{"execution_id": "exec", "occurrence_ids": ["only"], "parameters": {}}],
    }
    results = {
        "schema": "profiling-skill/bsa-candidate-results/v1",
        "executions": {"exec": {"status": "completed", "cases": {"47": {"status": "completed", "samples": [{"duration_us": 9.0, "pipe_utilization": {}, "provenance": {}}]}}, "infrastructure_failures": []}},
    }
    manifest_path, results_path = tmp_path / "manifest.json", tmp_path / "results.json"
    write_json(manifest_path, manifest)
    write_json(results_path, results)
    output = tmp_path / "out"
    run = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest_path), "--captures", str(results_path), "--hypotheses", str(PLANNED_HYPOTHESES), "--output-dir", str(output)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    candidate = json.loads((output / "report.json").read_text())["candidates"][0]
    assert candidate["parent_candidate_id"] is None
    assert candidate["cases"]["47"]["starter_speedup"] is None
    assert candidate["cases"]["47"]["parent_speedup"] is None


def test_accepts_only_paired_microcampaign_hypothesis_results(tmp_path: Path):
    manifest, captures, _ = fixtures(tmp_path)
    paired = {
        "schema_version": 1,
        "manifest_sha256": "a" * 64,
        "raw_cells": [
            {"id": "loop/small/before", "variant": "before"},
            {"id": "loop/small/after", "variant": "after"},
        ],
        "results": [{
            "hypothesis_id": "loop",
            "title": "Local loop bounds reduce time",
            "expected": {"direction": "decrease", "minimum_relative_change": 0.05},
            "measurements": [{
                "status": "valid", "before": 100.0, "after": 80.0,
                "sweep": "small", "metric": "core-time proxy", "unit": "us",
                "before_cell_id": "loop/small/before", "after_cell_id": "loop/small/after",
            }],
            "signals": [],
            "falsification": "after does not improve",
        }],
    }
    paired_path = tmp_path / "paired.json"
    write_json(paired_path, paired)
    output = tmp_path / "paired-out"
    accepted = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--captures", str(captures), "--hypotheses", str(paired_path), "--output-dir", str(output)],
        text=True, capture_output=True, check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert json.loads((output / "report.json").read_text())["hypotheses"][0]["classification"] == "supported"

    raw_path = tmp_path / "raw.json"
    write_json(raw_path, {"schema_version": 1, "results": [
        {"id": "loop/small/before", "experiment": "loop", "sweep": "small", "variant": "before", "status": "success", "measurements": {"median": 100.0}}
    ]})
    rejected = subprocess.run(
        [sys.executable, str(SCRIPT), "--manifest", str(manifest), "--captures", str(captures), "--hypotheses", str(raw_path), "--output-dir", str(tmp_path / "raw-out")],
        text=True, capture_output=True, check=False,
    )
    assert rejected.returncode == 2
    assert "raw unpaired microcampaign cells" in rejected.stderr
