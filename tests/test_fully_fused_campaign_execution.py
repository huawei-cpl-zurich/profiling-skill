from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "fully_fused_campaign_execution",
    ROOT / "scripts" / "fully_fused_campaign_execution.py",
)
execution = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = execution
SPEC.loader.exec_module(execution)


def _manifest() -> dict:
    cells = []
    for product in execution.PRODUCTS:
        for task in execution.TASKS:
            for treatment in execution.TREATMENTS:
                cell_id = f"{product}-{task}-{treatment}"
                cells.append({
                    "cell_id": cell_id, "product": product, "task": task,
                    "treatment": treatment,
                    "branch": f"experiment/fused/{cell_id}", "round_count": 4,
                })
    return {
        "schema_version": 4, "run_id": "fused",
        "dimensions": {"products": list(execution.PRODUCTS),
                       "tasks": list(execution.TASKS),
                       "treatments": list(execution.TREATMENTS),
                       "round_count": 4},
        "cells": cells, "order": [cell["cell_id"] for cell in cells],
        "manifest_sha256": "f" * 64,
    }


def _rankings() -> dict:
    return {
        product: {task: list(range(offset, offset + 5))
                  for offset, task in enumerate(execution.TASKS)}
        for product in execution.PRODUCTS
    }


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_archive_seal_verifies_every_frozen_root_and_completion_file(
    tmp_path: Path, monkeypatch,
):
    first, second = tmp_path / "old-one", tmp_path / "old-two"
    first.mkdir(); second.mkdir()
    (first / "result.json").write_text("one\n")
    (second / "result.json").write_text("two\n")
    completion = tmp_path / "completion.json"
    completion.write_text("{}\n")
    digests = {str(first.resolve()): "1" * 64, str(second.resolve()): "2" * 64}
    monkeypatch.setattr(execution, "deterministic_tree_digest",
                        lambda path: digests[str(path.resolve())])
    seal = {
        "schema": execution.ARCHIVE_SCHEMA,
        "roots": [{"path": str(path.resolve()), "sha256": digests[str(path.resolve())]}
                  for path in (first, second)],
        "completion_files": [{"path": str(completion.resolve()),
                              "sha256": _sha(completion)}],
    }
    seal["seal_sha256"] = execution.receipt_sha256(seal, "seal_sha256")

    assert execution.verify_archive_seal(seal)["root_count"] == 2
    (completion).write_text("drift\n")
    with pytest.raises(execution.ExecutionError, match="completion file drifted"):
        execution.verify_archive_seal(seal)


def test_remote_validation_bundle_contains_runtime_and_campaign_fixtures(tmp_path: Path):
    bundle = execution.build_validation_bundle(ROOT, tmp_path / "validation.tar")
    with tarfile.open(bundle) as archive:
        members = set(archive.getnames())

    assert "scripts/fully_fused_campaign_execution.py" in members
    assert "tests/test_fully_fused_campaign_execution.py" in members
    assert "experiments/audited-repair-canaries.json" in members


def test_prepare_requires_exact_24_cell_matrix_and_product_rankings():
    prepared = execution.prepare_campaign(_manifest(), _rankings(),
                                          archive_attestation="a" * 64)

    assert prepared["schema"] == execution.PREPARED_SCHEMA
    assert prepared["cell_count"] == 24
    assert prepared["expected_round_commits"] == 96
    assert len(set(prepared["branches"])) == 24
    assert prepared["ranked_cases"]["a3"]["gdn"] == [1, 2, 3, 4, 5]
    execution.verify_prepared_receipt(prepared)


def test_receipt_hashes_are_self_verifying_and_reject_tampering():
    prepared = execution.prepare_campaign(
        _manifest(), _rankings(), archive_attestation="a" * 64,
    )
    assert execution.verify_prepared_receipt(prepared) == prepared
    tampered = json.loads(json.dumps(prepared))
    tampered["cell_count"] = 23
    with pytest.raises(execution.ExecutionError, match="prepared receipt digest"):
        execution.verify_prepared_receipt(tampered)


@pytest.mark.parametrize("mutation,message", [
    ("missing-cell", "exact 24-cell"),
    ("duplicate-branch", "unique unmerged branch"),
    ("wrong-rounds", "four rounds"),
    ("missing-ranking", "rankings"),
])
def test_prepare_fails_closed_on_matrix_or_ranking_drift(mutation: str, message: str):
    manifest, rankings = _manifest(), _rankings()
    if mutation == "missing-cell":
        manifest["cells"].pop()
    elif mutation == "duplicate-branch":
        manifest["cells"][1]["branch"] = manifest["cells"][0]["branch"]
    elif mutation == "wrong-rounds":
        manifest["cells"][0]["round_count"] = 3
    else:
        rankings["a5"].pop("bsa")
    with pytest.raises(execution.ExecutionError, match=message):
        execution.prepare_campaign(manifest, rankings, archive_attestation="a" * 64)


def test_known_good_matmul_gate_requires_three_samples_fusion_and_both_products():
    calls = []

    def probe(product: str, ranked_cases: list[int]) -> dict:
        calls.append((product, ranked_cases))
        return {
            "status": "ok", "product": product, "task": "matmul",
            "handle": f"remote:{product}:job:gate", "samples_us": [9.0, 10.0, 11.0],
            "median_us": 10.0, "variability_ratio": 0.2,
            "fusion_gate": {"cases": list(range(10)), "logical_launches_per_case": 1,
                            "entrypoint": "matmul", "kernel_name": "matmul"},
        }

    receipt = execution.run_matmul_gates(_rankings(), probe, prepared_sha256="b" * 64)
    assert receipt["status"] == "passed"
    assert [item[0] for item in calls] == ["a3", "a5"]
    assert {row["product"] for row in receipt["products"]} == {"a3", "a5"}
    execution.verify_gate_receipt(receipt, _rankings(), prepared_sha256="b" * 64)


def test_authentic_controller_full_domain_fusion_proof_covers_ranked_gate():
    ranked = _rankings()["a3"]["matmul"]
    receipt = {
        "status": "ok", "product": "a3", "task": "matmul", "handle": "remote:h",
        "samples_us": [9.0, 10.0, 11.0], "median_us": 10.0,
        "variability_ratio": 0.2,
        "fusion_gate": {"cases": list(range(10)), "logical_launches_per_case": 1,
                        "entrypoint": "matmul", "kernel_name": "matmul"},
    }
    assert execution.validate_matmul_gate("a3", ranked, receipt)["fusion_gate"][
        "cases"
    ] == list(range(10))


def test_known_good_gate_rejects_nonfused_or_infrastructure_outcome():
    rankings = _rankings()
    with pytest.raises(execution.ExecutionError, match="infrastructure pending"):
        execution.validate_matmul_gate(
            "a3", rankings["a3"]["matmul"],
            {"status": "infrastructure_error", "handle": "remote:a3:job:1"},
        )
    with pytest.raises(execution.ExecutionError, match="fully fused"):
        execution.validate_matmul_gate(
            "a3", rankings["a3"]["matmul"], {
                "status": "ok", "product": "a3", "task": "matmul", "handle": "h",
                "samples_us": [1.0, 1.0, 1.0], "median_us": 1.0,
                "variability_ratio": 0.0,
                "fusion_gate": {"cases": list(range(10)),
                                "logical_launches_per_case": 2,
                                "entrypoint": "matmul", "kernel_name": "matmul"},
            },
        )


def _report() -> dict:
    rows = []
    for cell in _manifest()["cells"]:
        rows.append({
            **{key: cell[key] for key in ("cell_id", "product", "task", "treatment", "branch")},
            "status": "complete", "baseline_median_us": 20.0,
            "comparison_median_us": 10.0, "speedup_vs_baseline": 2.0,
            "best_round": 4,
            "commits": [hashlib.sha1(f"{cell['cell_id']}-{number}".encode()).hexdigest()
                        for number in range(4)],
            "variability_ratio": 0.03,
            "fusion_gate": {"logical_launches_per_case": 1},
            "profiler_evidence": ["remote:/compact/profile.json"],
            "bottleneck": {"status": "not-collected",
                           "reason": "timing evidence has no pipe metrics"},
        })
    return {"schema_version": 2, "run_id": "fused", "manifest_sha256": "f" * 64,
            "summary": {"complete": 24, "candidate_failed": 0,
                        "infrastructure_pending": 0}, "cells": rows,
            "discarded_infrastructure_attempts": [{"cell_id": "x", "error": "timeout"}]}


def test_compact_report_requires_96_commits_and_emits_product_tables():
    compact = execution.compact_report(_manifest(), _report())

    assert compact["round_commit_count"] == 96
    assert compact["infrastructure_exclusions"] == 1
    assert len(compact["tables"]["a3"]) == 12
    row = compact["tables"]["a5"][0]
    assert row["raw_candidate_us"] == 10.0
    assert row["speedup"] == 2.0
    assert row["fusion"] == "1 logical launch/case"
    rendered = execution.render_markdown(compact)
    assert "| Product | Task | Treatment |" in rendered
    assert "Profiler evidence" in rendered and "Round commits" in rendered
    execution.verify_compact_report(compact)


def test_compact_report_distinguishes_candidate_failure_from_infra_exclusion():
    report = _report()
    report["cells"][0].update(
        status="candidate_failed", comparison_median_us=None,
        speedup_vs_baseline=None,
        commits=[hashlib.sha1(f"failed-{n}".encode()).hexdigest() for n in range(4)],
        failure=[{"failure_type": "compile_error", "reason": "bad constexpr"}],
        fusion_gate=None,
    )
    compact = execution.compact_report(_manifest(), report)

    failed = compact["tables"]["a3"][0]
    assert failed["outcome"] == "candidate_failed:compile_error"
    assert failed["raw_candidate_us"] is None
    assert compact["infrastructure_exclusions"] == 1


def test_compact_report_rejects_missing_round_lineage():
    report = _report()
    report["cells"][0]["commits"].pop()
    with pytest.raises(execution.ExecutionError, match="four unique round commits"):
        execution.compact_report(_manifest(), report)


@pytest.mark.parametrize("mutation,message", [
    ("schema", "schema"), ("run", "run identity"),
    ("manifest", "manifest identity"), ("commit", "commit"),
    ("fusion", "successful fusion proof"),
])
def test_compact_report_fails_closed_on_identity_lineage_and_fusion(
    mutation: str, message: str,
):
    manifest, report = _manifest(), _report()
    if mutation == "schema":
        report["schema_version"] = 1
    elif mutation == "run":
        report["run_id"] = "other"
    elif mutation == "manifest":
        report["manifest_sha256"] = "e" * 64
    elif mutation == "commit":
        report["cells"][0]["commits"][0] = "not-a-commit"
    else:
        report["cells"][0]["fusion_gate"] = None
    with pytest.raises(execution.ExecutionError, match=message):
        execution.compact_report(manifest, report)


def test_compact_report_digest_rejects_tampering():
    compact = execution.compact_report(_manifest(), _report())
    compact["tables"]["a3"][0]["speedup"] = 999
    with pytest.raises(execution.ExecutionError, match="report.*digest"):
        execution.verify_compact_report(compact)
