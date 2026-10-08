#!/usr/bin/env python3
"""Validate and aggregate trusted profiling behavioral-gate records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Sequence

MANIFEST_SCHEMA = "profiling-skill/behavioral-manifest/v2"
RECORDS_SCHEMA = "profiling-skill/behavioral-records/v2"
REPORT_SCHEMA = "profiling-skill/behavioral-report/v2"
RECEIPT_SCHEMA = "profiling-skill/launcher-receipt/v1"
MAX_ARTIFACT = 65_536
HEX = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
INFRA = {"transport", "device_busy", "target_unavailable", "observer"}
COUNTED = {"compile", "runtime", "profiler_command", "evidence", "interpretation", "launcher"}
STAGES = {"launcher", "compile", "runtime", "profiler", "evidence", "interpretation",
          "observer", "transport"}


class GateError(RuntimeError):
    pass


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GateError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise GateError(f"expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact(value, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise GateError(f"invalid {label} fields")
    return value


def _hex(value, label: str) -> str:
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise GateError(f"invalid {label} identity")
    return value


def _identity(value, hash_key: str, label: str) -> dict:
    value = _exact(value, {"identity", hash_key}, label)
    if not isinstance(value["identity"], str) or not value["identity"] or len(value["identity"]) > 128:
        raise GateError(f"invalid {label} identity")
    _hex(value[hash_key], label)
    return value


def _handle(target, handle, label: str) -> str:
    if not isinstance(target, str) or not isinstance(handle, str) or len(handle) > 256:
        raise GateError(f"invalid {label} handle")
    prefix = f"remote:{target}:job:"
    handle_id = (handle.removeprefix(prefix) if handle.startswith(prefix) else
                 handle.removeprefix("gz-a3:")
                 if target == "gz-a3" and handle.startswith("gz-a3:") else "")
    if not IDENTIFIER.fullmatch(handle_id):
        raise GateError(f"invalid {label} handle")
    return handle


def _semantic(values, label: str, validate) -> set[str]:
    if not isinstance(values, list):
        raise GateError(f"invalid {label}")
    canonical = []
    for value in values:
        validate(value)
        canonical.append(json.dumps(value, sort_keys=True, separators=(",", ":")))
    if len(canonical) != len(set(canonical)):
        raise GateError(f"duplicate {label}")
    return set(canonical)


def _conclusion(value) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise GateError("invalid conclusion")


def _artifact(reference, root: Path, label: str, *, metadata=False,
              text=False) -> tuple[dict, dict | None]:
    keys = {"path", "sha256", "schema", "provenance"} if metadata else {"path", "sha256"}
    reference = _exact(reference, keys, f"{label} reference")
    relative = Path(reference["path"]) if isinstance(reference["path"], str) else Path("..")
    path = root / relative
    parents = [root.joinpath(*relative.parts[:index])
               for index in range(1, len(relative.parts) + 1)]
    if (relative.is_absolute() or ".." in relative.parts or len(str(relative)) > 256
            or any(candidate.is_symlink() for candidate in parents) or not path.exists()
            or path.is_symlink() or not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())):
        raise GateError(f"invalid {label} path")
    size = path.stat().st_size
    if not 0 < size <= MAX_ARTIFACT or _hex(reference["sha256"], label) != _hash(path):
        raise GateError(f"invalid {label} hash or size")
    if text:
        try: content = path.read_text()
        except UnicodeDecodeError as error: raise GateError(f"invalid {label} text") from error
        if not content.strip(): raise GateError(f"empty {label} text")
    document = _json(path) if metadata or label == "receipt" else None
    if metadata:
        if (not isinstance(reference["schema"], str) or not reference["schema"]
                or not isinstance(reference["provenance"], dict)
                or document.get("schema") != reference["schema"]
                or document.get("provenance") != reference["provenance"]):
            raise GateError(f"invalid {label} metadata")
    return reference, document


def _claim(claim) -> dict:
    claim = _exact(claim, {"state", "resource", "interval", "provenance"}, "saturation claim")
    if not isinstance(claim["state"], str) or claim["state"] not in {
            "saturated", "unsaturated", "unknown"}:
        raise GateError("invalid saturation claim state")
    if any(not isinstance(claim[key], str) or not claim[key] or len(claim[key]) > 256
           for key in ("resource", "interval", "provenance")):
        raise GateError("invalid saturation claim value")
    return claim


def _manifest(path: Path, root: Path) -> tuple[dict, dict[str, dict], str]:
    value = _exact(_json(path), {"schema", "prompt", "prompt_sha256", "skills", "launcher",
                                 "model", "reviewer", "allowed_targets", "cases"}, "manifest")
    if value["schema"] != MANIFEST_SCHEMA:
        raise GateError("unsupported manifest schema")
    _hex(value["prompt_sha256"], "prompt")
    _artifact(value["prompt"], root, "prompt", text=True)
    if value["prompt"]["sha256"] != value["prompt_sha256"]:
        raise GateError("prompt identity mismatch")
    skills = _exact(value["skills"], {"current", "candidate"}, "skills")
    for arm in skills: _hex(skills[arm], f"{arm} skill")
    if skills["current"] == skills["candidate"]:
        raise GateError("current and candidate skills must be distinct")
    _identity(value["launcher"], "sha256", "launcher")
    _identity(value["model"], "config_sha256", "model")
    _identity(value["reviewer"], "config_sha256", "reviewer")
    targets = _exact(value["allowed_targets"], {"a3", "a5"}, "allowed targets")
    for product, names in targets.items():
        if (not isinstance(names, list) or not names
                or any(not isinstance(name, str) or not IDENTIFIER.fullmatch(name) for name in names)
                or len(set(names)) != len(names)):
            raise GateError(f"invalid {product} targets")
    if not isinstance(value["cases"], list) or len(value["cases"]) != 4:
        raise GateError("exactly four cases are required")
    cases = {}
    for case in value["cases"]:
        case = _exact(case, {"id", "product", "evidence", "capture", "rubric"}, "case")
        if (not isinstance(case["id"], str) or not IDENTIFIER.fullmatch(case["id"])
                or case["id"] in cases or not isinstance(case["product"], str)
                or case["product"] not in {"a3", "a5"}):
            raise GateError("invalid or duplicate case")
        _artifact(case["evidence"], root, "case evidence", metadata=True)
        capture = _exact(case["capture"], {"target", "handle", "command", "log"},
                         "case capture")
        product, target = case["product"], capture["target"]
        if not isinstance(target, str) or target not in targets[product]:
            raise GateError("invalid case capture target")
        _handle(target, capture["handle"], "case capture")
        for label in ("command", "log"):
            _artifact(capture[label], root, f"case capture {label}", text=True)
        expected = {"product": product, "target": target, "handle": capture["handle"]}
        if case["evidence"]["provenance"] != expected:
            raise GateError("case evidence product mismatch or invalid capture provenance")
        rubric = _exact(case["rubric"], {"conclusions", "saturation_claims"}, "rubric")
        _semantic(rubric["conclusions"], "rubric conclusions", _conclusion)
        _semantic(rubric["saturation_claims"], "rubric saturation claims", _claim)
        cases[case["id"]] = case
    if Counter(case["product"] for case in cases.values()) != {"a3": 2, "a5": 2}:
        raise GateError("cases must contain two A3 and two A5 cases")
    return value, cases, _hash(path)


def _common(record: dict, expected_keys: set[str], manifest: dict, manifest_hash: str,
            sessions: set[str], expected_skill: str) -> tuple[str, str]:
    record = _exact(record, expected_keys, "trusted record")
    session, classification = record["session_id"], record["classification"]
    if not isinstance(session, str) or not IDENTIFIER.fullmatch(session):
        raise GateError("invalid session identity")
    if session in sessions: raise GateError("duplicate session identity")
    sessions.add(session)
    if not isinstance(classification, str):
        raise GateError("invalid failure classification")
    if (record["manifest_sha256"] != manifest_hash or record["launcher"] != manifest["launcher"]
            or record["model"] != manifest["model"]
            or record["skill_sha256"] != manifest["skills"][expected_skill]):
        raise GateError("record identity drift")
    return session, classification


def _failure(record: dict, base: set[str], manifest: dict, manifest_hash: str,
             sessions: set[str], skill: str, root: Path) -> str:
    fields = {"failure_type", "stage", "product", "target", "handle", "command", "log",
              "diagnostic", "receipt"}
    session, classification = _common(
        record, base | fields, manifest, manifest_hash, sessions, skill)
    failure_type = record["failure_type"]
    allowed = (INFRA if classification == "discarded_infrastructure" else COUNTED
               if classification == "counted_failure" else set())
    if not isinstance(failure_type, str) or failure_type not in allowed:
        raise GateError("invalid failure classification")
    if not isinstance(record["stage"], str) or record["stage"] not in STAGES:
        raise GateError("invalid failure stage")
    product, target, handle = record["product"], record["target"], record["handle"]
    if product is not None:
        if (not isinstance(product, str) or product not in {"a3", "a5"}
                or not isinstance(target, str) or target not in manifest["allowed_targets"][product]):
            raise GateError("invalid failure product or target")
        if handle is not None: _handle(target, handle, "failure")
    elif target is not None or handle is not None:
        raise GateError("failure target/handle requires product")
    for label in ("command", "log", "diagnostic"):
        _artifact(record[label], root, f"failure {label}", text=True)
    _, receipt = _artifact(record["receipt"], root, "receipt")
    expected = {"schema": RECEIPT_SCHEMA, "session_id": session, "kind": record["kind"],
                "arm": record.get("arm"), "classification": classification,
                "failure_type": failure_type, "stage": record["stage"], "product": product,
                "target": target, "handle": handle,
                "artifacts": {label: record[label]["sha256"]
                              for label in ("command", "log", "diagnostic")}}
    if receipt != expected: raise GateError("invalid launcher receipt")
    return classification


def _payload(value, expected: dict) -> None:
    if value != expected:
        raise GateError("invalid agent payload or arm leak")


def _review(value, manifest: dict, root: Path) -> bool:
    value = _exact(value, {"passed", "reviewer", "notes"}, "manual review")
    if not isinstance(value["passed"], bool) or value["reviewer"] != manifest["reviewer"]:
        raise GateError("invalid manual review identity")
    _artifact(value["notes"], root, "manual review notes", text=True)
    return value["passed"]


def _outcome(value, session: str, manifest: dict, root: Path,
             artifacts: list[dict], handles: set[str], evidence_hashes: set[str]) -> bool:
    value = _exact(value, {"session_id", "product", "target", "handle", "evidence", "command",
                           "log", "reasoning", "agent_payload", "manual_review"},
                   "acquisition outcome")
    product, target, handle = value["product"], value["target"], value["handle"]
    if value["session_id"] != session: raise GateError("outcomes do not share paired session")
    if (not isinstance(product, str) or product not in {"a3", "a5"}
            or not isinstance(target, str) or target not in manifest["allowed_targets"][product]):
        raise GateError("unapproved acquisition target")
    _handle(target, handle, "durable")
    _artifact(value["evidence"], root, "profile evidence", metadata=True)
    expected_provenance = {"product": product, "target": target, "handle": handle}
    if value["evidence"]["provenance"] != expected_provenance:
        raise GateError("invalid evidence provenance")
    evidence_hash = value["evidence"]["sha256"]
    if handle in handles or evidence_hash in evidence_hashes:
        raise GateError("reused acquisition handle or evidence")
    handles.add(handle); evidence_hashes.add(evidence_hash)
    for label in ("command", "log", "reasoning"): _artifact(value[label], root, label, text=True)
    _payload(value["agent_payload"], {"product": product,
                                     "prompt_sha256": manifest["prompt_sha256"]})
    review_passed = _review(value["manual_review"], manifest, root)
    artifacts.append({"session_id": session, "product": product,
                      **{key: value[key] for key in ("evidence", "command", "log", "reasoning")}})
    artifacts[-1]["manual_review"] = value["manual_review"]
    return review_passed


def evaluate(manifest_path: Path, records_path: Path, artifact_root: Path) -> dict:
    if artifact_root.is_symlink() or not artifact_root.is_dir():
        raise GateError("artifact root must be a regular directory")
    root = artifact_root.resolve()
    manifest, cases, manifest_hash = _manifest(manifest_path.resolve(), root)
    records = _exact(_json(records_path.resolve()), {"schema", "acquisition", "interpretation"},
                     "records")
    if records["schema"] != RECORDS_SCHEMA or not all(
            isinstance(records[key], list) for key in ("acquisition", "interpretation")):
        raise GateError("invalid records schema")
    sessions: set[str] = set(); artifacts = [{"prompt": manifest["prompt"]}]
    artifacts.extend({"case_id": case_id, "evidence": case["evidence"],
                      "capture": case["capture"]} for case_id, case in cases.items())
    scores = []; attempts = []
    counts = Counter(); acquisition_success = 0; acquisition_terminal = 0; manual_ok = True
    handles: set[str] = set(); evidence_hashes: set[str] = set()
    base = {"kind", "session_id", "classification", "manifest_sha256", "launcher", "model",
            "skill_sha256"}
    for record in records["acquisition"]:
        if not isinstance(record, dict) or record.get("kind") != "acquisition":
            raise GateError("invalid acquisition record")
        if record.get("classification") != "success":
            classification = _failure(record, base, manifest, manifest_hash, sessions,
                                      "candidate", root)
            counts[classification] += 1
            artifacts.append({"session_id": record["session_id"], "receipt": record["receipt"]})
            attempts.append({"kind": "acquisition", "session_id": record["session_id"],
                             "classification": classification,
                             **{key: record[key] for key in ("failure_type", "stage", "product",
                                "target", "handle", "command", "log", "diagnostic", "receipt")}})
            acquisition_terminal += classification == "counted_failure"
            continue
        session, classification = _common(record, base | {"outcomes"}, manifest,
                                          manifest_hash, sessions, "candidate")
        outcomes = record["outcomes"]
        if (classification != "success" or not isinstance(outcomes, list) or len(outcomes) != 2
                or any(not isinstance(item, dict) for item in outcomes)
                or sorted(item.get("product") for item in outcomes
                          if isinstance(item.get("product"), str)) != ["a3", "a5"]):
            raise GateError("successful acquisition must pair A3 and A5")
        for outcome in outcomes:
            manual_ok &= _outcome(outcome, session, manifest, root, artifacts,
                                  handles, evidence_hashes)
        acquisition_success += 1; acquisition_terminal += 1; counts["success"] += 1
        attempts.append({"kind": "acquisition", "session_id": session,
                         "classification": "success", "failure_type": None})
    terminal = Counter(); passed = Counter()
    interp_base = base | {"arm"}
    for record in records["interpretation"]:
        if not isinstance(record, dict) or record.get("kind") != "interpretation" \
                or not isinstance(record.get("arm"), str) \
                or record["arm"] not in {"current", "candidate"}:
            raise GateError("invalid interpretation record")
        arm = record["arm"]
        if record.get("classification") != "success":
            classification = _failure(record, interp_base, manifest, manifest_hash,
                                      sessions, arm, root)
            counts[classification] += 1
            artifacts.append({"session_id": record["session_id"], "receipt": record["receipt"]})
            attempts.append({"kind": "interpretation", "arm": arm,
                             "session_id": record["session_id"],
                             "classification": classification,
                             **{key: record[key] for key in ("failure_type", "stage", "product",
                                "target", "handle", "command", "log", "diagnostic", "receipt")}})
            if classification == "counted_failure":
                terminal[arm] += 1
                for case_id in cases:
                    scores.append({"session_id": record["session_id"], "arm": arm,
                                   "case_id": case_id, "passed": False,
                                   "failure_type": record["failure_type"]})
            continue
        session, classification = _common(record, interp_base | {"answers"}, manifest,
                                          manifest_hash, sessions, arm)
        if classification != "success" or not isinstance(record["answers"], list) \
                or len(record["answers"]) != 4:
            raise GateError("successful interpretation requires four answers")
        if any(not isinstance(answer, dict) or not isinstance(answer.get("case_id"), str)
               for answer in record["answers"]):
            raise GateError("interpretation case set mismatch")
        by_case = {answer["case_id"]: answer for answer in record["answers"]}
        if len(by_case) != 4 or set(by_case) != set(cases):
            raise GateError("interpretation case set mismatch")
        terminal[arm] += 1; counts["success"] += 1
        attempts.append({"kind": "interpretation", "arm": arm, "session_id": session,
                         "classification": "success", "failure_type": None})
        for case_id, case in cases.items():
            answer = _exact(by_case[case_id], {"case_id", "evidence_sha256", "conclusions",
                "saturation_claims", "log", "reasoning", "agent_payload", "manual_review"}, "answer")
            if answer["evidence_sha256"] != case["evidence"]["sha256"]:
                raise GateError("answer evidence identity drift")
            _payload(answer["agent_payload"], {"case_id": case_id, "product": case["product"],
                "prompt_sha256": manifest["prompt_sha256"],
                "evidence_sha256": case["evidence"]["sha256"]})
            for label in ("log", "reasoning"): _artifact(answer[label], root, label, text=True)
            review = answer["manual_review"]
            review_passed = _review(review, manifest, root)
            conclusions = _semantic(answer["conclusions"], "answer conclusions", _conclusion)
            claims = _semantic(answer["saturation_claims"], "answer saturation claims", _claim)
            score = (conclusions == _semantic(case["rubric"]["conclusions"],
                                              "rubric conclusions", _conclusion)
                     and claims == _semantic(case["rubric"]["saturation_claims"],
                                             "rubric saturation claims", _claim))
            passed[arm] += score
            manual_ok &= review_passed
            scores.append({"session_id": session, "arm": arm, "case_id": case_id,
                           "passed": score, "evidence": case["evidence"],
                           "log": answer["log"], "reasoning": answer["reasoning"],
                           "manual_review": review})
    if acquisition_terminal != 3:
        raise GateError("exactly three terminal acquisition sessions required")
    if terminal != {"current": 3, "candidate": 3}:
        raise GateError("exactly three terminal interpretation sessions per arm required")
    interpretation = {arm: {"passed": passed[arm], "total": 12}
                      for arm in ("current", "candidate")}
    acquisition = {"terminal_sessions": acquisition_terminal,
                   "successful_pairs": acquisition_success,
                   "score": {"passed": acquisition_success, "total": 3}}
    acceptance = {"acquisition_3_of_3": acquisition_success == 3,
                  "candidate_12_of_12": passed["candidate"] == 12,
                  "candidate_beats_current": passed["candidate"] > passed["current"],
                  "manual_review": manual_ok}
    acceptance["passed"] = all(acceptance.values())
    return {"schema": REPORT_SCHEMA, "manifest_sha256": manifest_hash,
            "counts": dict(counts), "acquisition": acquisition,
            "interpretation": interpretation, "scores": scores,
            "artifacts": artifacts, "attempts": attempts, "acceptance": acceptance}


def _write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        json.dump(report, output, sort_keys=True, indent=2); output.write("\n")
        temporary = output.name
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try: report = evaluate(args.manifest, args.records, args.artifact_root)
    except GateError as error: parser.error(str(error))
    _write(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["acceptance"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
