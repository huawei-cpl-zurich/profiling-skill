from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gate", ROOT / "scripts/profile_behavioral_gate.py")
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(gate)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(root: Path, name: str, value, **metadata) -> dict:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n" if isinstance(value, (dict, list)) else value)
    return {"path": name, "sha256": sha(path), **metadata}


def fixture(tmp_path: Path):
    root = tmp_path / "artifacts"; root.mkdir()
    cases = []
    for number, product in enumerate(("a3", "a3", "a5", "a5"), 1):
        provenance = {"product": product, "handle": f"frozen:{product}:{number}"}
        evidence = artifact(root, f"cases/{number}.json",
            {"schema": "compact/v1", "provenance": provenance, "value": number},
            schema="compact/v1", provenance=provenance)
        claims = ([{"state": "saturated", "resource": "MTE2",
                    "interval": "whole-kernel", "provenance": "capacity/v1"}]
                  if number == 4 else [])
        cases.append({"id": f"case-{number}", "product": product, "evidence": evidence,
                      "rubric": {"conclusions": [f"fact-{number}"],
                                 "saturation_claims": claims}})
    manifest = {
        "schema": gate.MANIFEST_SCHEMA, "prompt_sha256": "1" * 64,
        "skills": {"current": "2" * 64, "candidate": "3" * 64},
        "launcher": {"identity": "audited-launcher/v1", "sha256": "4" * 64},
        "model": {"identity": "test-model", "config_sha256": "5" * 64},
        "reviewer": {"identity": "reviewer/v1", "config_sha256": "6" * 64},
        "allowed_targets": {"a3": ["bz-a3-1", "bz-a3-2"], "a5": ["bz-a5"]},
        "cases": cases,
    }
    manifest_path = tmp_path / "manifest.json"; manifest_path.write_text(json.dumps(manifest))

    def common(session: str, arm: str, classification="success"):
        row = {"session_id": session, "classification": classification,
               "manifest_sha256": sha(manifest_path), "launcher": deepcopy(manifest["launcher"]),
               "model": deepcopy(manifest["model"]), "skill_sha256": manifest["skills"][arm]}
        if classification != "success":
            failure = "device_busy" if classification == "discarded_infrastructure" else "compile"
            row.update(failure_type=failure, receipt=artifact(root, f"receipts/{session}.json", {
                "schema": gate.RECEIPT_SCHEMA, "session_id": session,
                "classification": classification, "failure_type": failure}))
        return row

    acquisitions = []
    for agent in range(1, 4):
        session = f"acq-{agent}"; outcomes = []
        for product, target in (("a3", "bz-a3-1"), ("a5", "bz-a5")):
            handle = f"remote:{target}:job:{session}-{product}"
            provenance = {"product": product, "target": target, "handle": handle}
            evidence = artifact(root, f"evidence/{session}-{product}.json",
                {"schema": "profile/v1", "provenance": provenance},
                schema="profile/v1", provenance=provenance)
            outcomes.append({"session_id": session, "product": product, "target": target,
                "handle": handle, "evidence": evidence,
                "command": artifact(root, f"commands/{session}-{product}.txt", "run\n"),
                "log": artifact(root, f"logs/{session}-{product}.txt", "ok\n"),
                "reasoning": artifact(root, f"reason/{session}-{product}.txt", "why\n"),
                "agent_payload": {"product": product,
                                  "prompt_sha256": manifest["prompt_sha256"]},
                "manual_review": {"passed": True, "reviewer": manifest["reviewer"],
                    "notes": artifact(root, f"review/{session}-{product}.txt", "sound\n")}})
        acquisitions.append({"kind": "acquisition", **common(session, "candidate"),
                             "outcomes": outcomes})

    interpretations = []
    for arm in ("current", "candidate"):
        for agent in range(1, 4):
            session = f"{arm}-{agent}"; answers = []
            for case in cases:
                conclusions = list(case["rubric"]["conclusions"])
                if arm == "current" and agent == 1 and case["id"] == "case-1":
                    conclusions = ["wrong"]
                answers.append({"case_id": case["id"],
                    "evidence_sha256": case["evidence"]["sha256"],
                    "conclusions": conclusions,
                    "saturation_claims": deepcopy(case["rubric"]["saturation_claims"]),
                    "log": artifact(root, f"logs/{session}-{case['id']}.txt", "used evidence\n"),
                    "reasoning": artifact(root, f"reason/{session}-{case['id']}.txt", "because\n"),
                    "agent_payload": {"case_id": case["id"], "product": case["product"],
                        "prompt_sha256": manifest["prompt_sha256"],
                        "evidence_sha256": case["evidence"]["sha256"]},
                    "manual_review": {"passed": True, "reviewer": manifest["reviewer"], "notes": artifact(
                        root, f"review/{session}-{case['id']}.txt", "sound\n")}})
            interpretations.append({"kind": "interpretation", "arm": arm,
                                    **common(session, arm), "answers": answers})
    records = {"schema": gate.RECORDS_SCHEMA,
               "acquisition": acquisitions, "interpretation": interpretations}
    records_path = tmp_path / "records.json"; records_path.write_text(json.dumps(records))
    return manifest_path, records_path, root, manifest, records


def test_accepts_12_of_12_candidate_and_blinded_payload(tmp_path: Path):
    manifest, records_path, root, _, records = fixture(tmp_path)
    report = gate.evaluate(manifest, records_path, root)
    assert report["acquisition"]["successful_pairs"] == 3
    assert report["interpretation"] == {"current": {"passed": 11, "total": 12},
                                        "candidate": {"passed": 12, "total": 12}}
    assert report["acceptance"]["passed"] is True and len(report["scores"]) == 24
    assert len(report["attempts"]) == 9
    payloads = [a["agent_payload"] for row in records["interpretation"] for a in row["answers"]]
    assert all("arm" not in p and "workspace" not in p for p in payloads)


def test_acquisition_requires_paired_attested_session(tmp_path: Path):
    manifest, path, root, _, records = fixture(tmp_path)
    records["acquisition"][0]["outcomes"][1]["session_id"] = "other"
    path.write_text(json.dumps(records))
    with pytest.raises(gate.GateError, match="paired session"):
        gate.evaluate(manifest, path, root)


def test_agent_payload_cannot_classify_infrastructure(tmp_path: Path):
    manifest, path, root, _, records = fixture(tmp_path)
    records["acquisition"][0]["outcomes"][0]["agent_payload"]["failure_type"] = "device_busy"
    path.write_text(json.dumps(records))
    with pytest.raises(gate.GateError, match="agent payload"):
        gate.evaluate(manifest, path, root)


@pytest.mark.parametrize("damage", ["hash", "symlink", "escape", "json", "target", "handle"])
def test_rejects_fabricated_evidence_or_remote_identity(tmp_path: Path, damage: str):
    manifest, path, root, _, records = fixture(tmp_path)
    outcome = records["acquisition"][0]["outcomes"][0]
    if damage == "hash": outcome["evidence"]["sha256"] = "0" * 64
    elif damage == "symlink":
        target = root / outcome["evidence"]["path"]; other = root / "other.json"
        other.write_text("{}"); target.unlink(); target.symlink_to(other)
    elif damage == "escape": outcome["evidence"]["path"] = "../manifest.json"
    elif damage == "json":
        evidence_path = root / outcome["evidence"]["path"]
        evidence_path.write_text("not json")
        outcome["evidence"]["sha256"] = sha(evidence_path)
    elif damage == "target": outcome["target"] = "unapproved"
    else: outcome["handle"] = "remote:bz-a3-2:job:wrong"
    path.write_text(json.dumps(records))
    with pytest.raises(gate.GateError): gate.evaluate(manifest, path, root)


@pytest.mark.parametrize("field,value", [
    ("conclusions", ["fact-1", "contradiction"]),
    ("saturation_claims", [{"state": "saturated", "resource": "MTE2",
                            "interval": "whole-kernel", "provenance": "activity-only"}]),
])
def test_exact_rubric_fails_extra_or_contradictory_claim(tmp_path: Path, field: str, value):
    manifest, path, root, _, records = fixture(tmp_path)
    records["interpretation"][3]["answers"][0][field] = value
    path.write_text(json.dumps(records)); report = gate.evaluate(manifest, path, root)
    assert report["interpretation"]["candidate"]["passed"] == 11
    assert report["acceptance"]["passed"] is False


@pytest.mark.parametrize("identity", ["manifest_sha256", "launcher", "model", "skill_sha256"])
def test_rejects_identity_drift(tmp_path: Path, identity: str):
    manifest, path, root, _, records = fixture(tmp_path); row = records["interpretation"][0]
    row[identity] = ({"identity": "wrong", "sha256": "9" * 64} if identity == "launcher"
                     else {"identity": "wrong", "config_sha256": "9" * 64}
                     if identity == "model" else "9" * 64)
    path.write_text(json.dumps(records))
    with pytest.raises(gate.GateError, match="identity"): gate.evaluate(manifest, path, root)


def test_rejects_duplicate_session(tmp_path: Path):
    manifest, path, root, _, records = fixture(tmp_path)
    records["interpretation"][1]["session_id"] = records["interpretation"][0]["session_id"]
    path.write_text(json.dumps(records))
    with pytest.raises(gate.GateError, match="duplicate session"): gate.evaluate(manifest, path, root)


def test_counts_infra_replacement_and_counted_failure(tmp_path: Path):
    manifest_path, path, root, manifest, records = fixture(tmp_path)
    for classification, session in (("discarded_infrastructure", "infra-1"),
                                    ("counted_failure", "failed-1")):
        failure = "device_busy" if classification.startswith("discarded") else "compile"
        receipt = artifact(root, f"receipts/{session}.json", {"schema": gate.RECEIPT_SCHEMA,
            "session_id": session, "classification": classification, "failure_type": failure})
        records["acquisition"].append({"kind": "acquisition", "session_id": session,
            "classification": classification, "failure_type": failure, "receipt": receipt,
            "manifest_sha256": sha(manifest_path), "launcher": manifest["launcher"],
            "model": manifest["model"], "skill_sha256": manifest["skills"]["candidate"]})
    path.write_text(json.dumps(records)); report = gate.evaluate(manifest_path, path, root)
    assert report["counts"] == {"success": 9, "discarded_infrastructure": 1,
                                "counted_failure": 1}


def test_candidate_requires_manual_review_pass_and_notes(tmp_path: Path):
    manifest, path, root, _, records = fixture(tmp_path)
    records["interpretation"][3]["answers"][0]["manual_review"]["passed"] = False
    path.write_text(json.dumps(records)); report = gate.evaluate(manifest, path, root)
    assert report["interpretation"]["candidate"]["passed"] == 12
    assert report["acceptance"]["manual_review"] is False
    assert report["acceptance"]["passed"] is False


@pytest.mark.parametrize("location", ["acquisition", "current", "reviewer"])
def test_every_reasoning_log_requires_attested_manual_review(tmp_path: Path, location: str):
    manifest, path, root, _, records = fixture(tmp_path)
    if location == "acquisition": records["acquisition"][0]["outcomes"][0]["manual_review"]["passed"] = False
    elif location == "current": records["interpretation"][0]["answers"][0]["manual_review"]["passed"] = False
    else: records["interpretation"][0]["answers"][0]["manual_review"]["reviewer"]["identity"] = "other"
    path.write_text(json.dumps(records))
    if location == "reviewer":
        with pytest.raises(gate.GateError, match="review identity"): gate.evaluate(manifest, path, root)
    else:
        assert gate.evaluate(manifest, path, root)["acceptance"]["manual_review"] is False


def test_rejects_nontext_log(tmp_path: Path):
    manifest, path, root, _, records = fixture(tmp_path)
    reference = records["interpretation"][0]["answers"][0]["log"]
    log = root / reference["path"]; log.write_bytes(b"\xff"); reference["sha256"] = sha(log)
    path.write_text(json.dumps(records))
    with pytest.raises(gate.GateError, match="log text"): gate.evaluate(manifest, path, root)


def test_accepts_gz_a3_durable_handle(tmp_path: Path):
    manifest_path, path, root, manifest, records = fixture(tmp_path)
    manifest["allowed_targets"]["a3"].append("gz-a3")
    manifest_path.write_text(json.dumps(manifest)); manifest_hash = sha(manifest_path)
    for row in records["acquisition"] + records["interpretation"]:
        row["manifest_sha256"] = manifest_hash
    outcome = records["acquisition"][0]["outcomes"][0]
    outcome.update(target="gz-a3", handle="gz-a3:job-123")
    provenance = {"product": "a3", "target": "gz-a3", "handle": "gz-a3:job-123"}
    outcome["evidence"]["provenance"] = provenance
    evidence = root / outcome["evidence"]["path"]
    evidence.write_text(json.dumps({"schema": "profile/v1", "provenance": provenance}))
    outcome["evidence"]["sha256"] = sha(evidence)
    path.write_text(json.dumps(records))
    assert gate.evaluate(manifest_path, path, root)["acceptance"]["passed"] is True


def test_case_evidence_product_must_match_case(tmp_path: Path):
    manifest, path, root, document, _ = fixture(tmp_path)
    reference = document["cases"][0]["evidence"]
    reference["provenance"]["product"] = "a5"
    evidence = root / reference["path"]
    evidence.write_text(json.dumps({"schema": reference["schema"],
                                    "provenance": reference["provenance"]}))
    reference["sha256"] = sha(evidence)
    manifest.write_text(json.dumps(document))
    with pytest.raises(gate.GateError, match="product mismatch"): gate.evaluate(manifest, path, root)


def test_cli_only_writes_atomic_report(tmp_path: Path):
    manifest, records, root, _, _ = fixture(tmp_path); output = tmp_path / "result/report.json"
    assert gate.main(["--manifest", str(manifest), "--records", str(records),
                      "--artifact-root", str(root), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["acceptance"]["passed"] is True
    assert [p.name for p in output.parent.iterdir()] == ["report.json"]
