#!/usr/bin/env python3
"""Run profiling behavioral agents behind the audited Bubblewrap boundary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from production_launcher import LaunchError, ProductionLauncher  # noqa: E402

REQUEST_SCHEMA = "profiling-skill/behavioral-launch-request/v1"
RECEIPT_SCHEMA = "profiling-skill/launcher-receipt/v1"
MAX_ARTIFACT = 65_536
RETAINED_REMOTE_FIELD = 4_096
HEX = re.compile(r"[0-9a-f]{64}")
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
HANDLE = re.compile(r"(?:remote:[A-Za-z0-9_.-]+:job:|gz-a3:)[A-Za-z0-9_.:-]+")
TERMINAL = {"completed", "failed", "cancelled"}
INFRA_STATES = {"reconnecting", "observation-unavailable", "target-unavailable",
                "device-busy", "transport-error"}


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class LaunchPaths:
    root: Path
    workspace: Path
    prompt: Path
    skill: Path
    state: Path
    socket_dir: Path
    wrapper: Path
    journal: Path


def _sha(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    value = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        value.update(item.relative_to(path).as_posix().encode() + b"\0")
        value.update(hashlib.sha256(item.read_bytes()).digest())
    return value.hexdigest()


def launcher_digest() -> str:
    """Pin this adapter and the audited boundary implementation it inherits."""
    value = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(__file__).with_name("production_launcher.py")):
        value.update(path.name.encode() + b"\0")
        value.update(hashlib.sha256(path.read_bytes()).digest())
    return value.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _bounded(value: object) -> str:
    text = str(value)
    # Logs are audit evidence, but credential-looking values are never evidence.
    text = re.sub(r"(?i)(authorization|token|password|secret)(\s*[:=]\s*)\S+",
                  r"\1\2<redacted>", text)
    encoded = text.encode(errors="replace")
    if len(encoded) > MAX_ARTIFACT:
        encoded = encoded[:MAX_ARTIFACT - 18] + b"\n...[truncated]\n"
    return encoded.decode(errors="replace")


def _compact_remote_text(value: str, limit: int,
                         result_failure_type: str | None = None) -> str:
    """Replace a large remote field with bounded, classification-safe evidence."""
    encoded = value.encode(errors="replace")
    if len(encoded) <= limit:
        return value
    digest = hashlib.sha256(encoded).hexdigest()
    content_hashes = list(dict.fromkeys(re.findall(
        r"(?m)^REMOTE_CONTENT_SHA256=([0-9a-f]{64})\s*$", value)))
    parsed_hash = _parse_remote(value).get("content_sha256")
    authoritative_hash = (parsed_hash if HEX.fullmatch(str(parsed_hash))
                          else content_hashes[-1] if content_hashes else None)
    if authoritative_hash and authoritative_hash not in content_hashes:
        content_hashes.append(authoritative_hash)

    failure_type = result_failure_type or classify_failure(
        {"stdout": value, "stderr": ""})[1]
    decisive_markers = {
        "profiler_command": ("profile_command=msprof ", "profiler failure",
                             "msprof op --help", "msprof op simulator --help"),
        "evidence": ("no exported kernel row", "expected one exported", "selector",
                     "evidence"),
        "compile": ("compile",),
        "runtime": ("runtime",),
    }.get(failure_type, ())
    decisive = next((line for line in value.splitlines()
                     if any(marker in line.lower() for marker in decisive_markers)), None)

    lines = [f"[compacted bytes={len(encoded)} sha256={digest}]"]
    classification_marker = {
        "profiler_command": "profiler failure",
        "evidence": "evidence failure",
        "compile": "compile failure",
    }.get(failure_type)
    if classification_marker:
        lines.append(
            f"[compacted-classification={classification_marker} source-sha256={digest}]")
    historical = [item for item in content_hashes if item != authoritative_hash]
    if historical:
        lines.append(
            f"[remote-content-history count={len(historical)} "
            f"sha256={_canonical_sha256(historical)}]")
    required_hash = (f"REMOTE_CONTENT_SHA256={authoritative_hash}"
                     if authoritative_hash else None)
    reserved = len((required_hash + "\n").encode()) if required_hash else 0
    if decisive:
        bounded_decisive = _bounded(decisive)
        available = limit - len(("\n".join(lines) + "\n").encode()) - reserved - 1
        if available > 0:
            positions = [bounded_decisive.lower().find(marker)
                         for marker in decisive_markers
                         if marker in bounded_decisive.lower()]
            start = max(0, min(positions, default=0) - min(64, available // 4))
            excerpt = bounded_decisive[start:].encode(errors="replace")[:available]
            lines.append(excerpt.decode(errors="ignore"))
    if required_hash:
        # Keep the authoritative value last so line-oriented replay selects it.
        lines.append(required_hash)
    return "\n".join(lines) + "\n"


def _compact_remote_journal(journal: dict, field_limit: int) -> dict:
    retained = deepcopy(journal)
    for collection in ("dispatches", "calls"):
        for entry in retained.get(collection, []):
            if not isinstance(entry, dict) or not isinstance(entry.get("result"), dict):
                continue
            result = entry["result"]
            result_failure_type = classify_failure(result)[1]
            for field in ("stdout", "stderr", "content"):
                if isinstance(result.get(field), str):
                    result[field] = _compact_remote_text(
                        result[field], field_limit,
                        result_failure_type if field in {"stdout", "stderr"} else None)
    return retained


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _summary_value(value: object, limit: int = 256) -> object:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = str(value)
    if len(text.encode(errors="replace")) <= limit:
        return text
    return {"bytes": len(text.encode(errors="replace")),
            "sha256": hashlib.sha256(text.encode(errors="replace")).hexdigest()}


def _summary_arguments(arguments: object) -> object:
    if isinstance(arguments, list) and all(isinstance(item, str) for item in arguments):
        encoded = json.dumps(arguments, separators=(",", ":")).encode()
        if len(encoded) <= 2_048:
            return arguments
        return {"count": len(arguments), "sha256": hashlib.sha256(encoded).hexdigest()}
    return {"sha256": _canonical_sha256(arguments)}


def _result_summary(result: object) -> object:
    if not isinstance(result, dict):
        return None
    summary = {key: _summary_value(result.get(key)) for key in (
        "returncode", "state", "handle", "exit", "reason", "failure_type")
        if result.get(key) is not None}
    fields = {}
    content_hashes = []
    for key in ("stdout", "stderr", "content"):
        value = result.get(key)
        if not isinstance(value, str):
            continue
        encoded = value.encode(errors="replace")
        fields[key] = {"bytes": len(encoded),
                       "sha256": hashlib.sha256(encoded).hexdigest()}
        hashes = re.findall(r"(?m)^REMOTE_CONTENT_SHA256=([0-9a-f]{64})\s*$", value)
        parsed = _parse_remote(value).get("content_sha256")
        if HEX.fullmatch(str(parsed)):
            hashes.append(parsed)
        content_hashes.extend(item for item in hashes if item not in content_hashes)
    if fields:
        summary["fields"] = fields
    if content_hashes:
        unique = list(dict.fromkeys(content_hashes))
        summary["remote_content_sha256"] = unique[:4] + unique[-4:] \
            if len(unique) > 8 else unique
        if len(unique) > 8:
            summary["remote_content_sha256_count"] = len(unique)
            summary["remote_content_sha256_set_digest"] = _canonical_sha256(unique)
    return summary


def _representatives(entries: object, keys: tuple[str, ...]) -> list[dict]:
    if not isinstance(entries, list):
        return []
    selected = entries[:2] + entries[-2:] if len(entries) > 4 else entries
    rows = []
    for entry in selected:
        if not isinstance(entry, dict):
            rows.append({"entry_sha256": _canonical_sha256(entry)})
            continue
        row = {key: _summary_value(entry.get(key)) for key in keys
               if entry.get(key) is not None}
        if "arguments" in entry:
            row["arguments"] = _summary_arguments(entry["arguments"])
        row["result"] = _result_summary(entry.get("result"))
        rows.append(row)
    return rows


def _journal_summary(journal: dict, source_sha256: str) -> dict:
    structural = deepcopy(journal)
    for collection in ("dispatches", "calls"):
        for entry in structural.get(collection, []):
            if isinstance(entry, dict) and isinstance(entry.get("result"), dict):
                for field in ("stdout", "stderr", "content"):
                    entry["result"].pop(field, None)
    dispatches, calls = journal.get("dispatches", []), journal.get("calls", [])
    return {
        "schema": _summary_value(journal.get("schema")),
        "source_canonical_sha256": source_sha256,
        "retention": {
            "schema": "profiling-skill/compact-remote-journal/v1",
            "mode": "representative-summary",
            "dispatch_count": len(dispatches) if isinstance(dispatches, list) else None,
            "call_count": len(calls) if isinstance(calls, list) else None,
            "structural_metadata_sha256": _canonical_sha256(structural),
        },
        "dispatches": _representatives(
            dispatches, ("request_sha256", "target", "dispatch_key", "file_sha256",
                         "state", "handle")),
        "calls": _representatives(
            calls, ("operation", "target", "request_sha256", "handle")),
    }


def retain_remote_journal(store: "ArtifactStore", relative: str, journal: dict) -> dict:
    """Retain an audit journal without copying unbounded remote output fields."""
    source_sha256 = _canonical_sha256(journal)
    for field_limit in (RETAINED_REMOTE_FIELD, 2_048, 1_024, 512):
        retained = _compact_remote_journal(journal, field_limit)
        retained["source_canonical_sha256"] = source_sha256
        retained["retention"] = {
            "schema": "profiling-skill/compact-remote-journal/v1",
            "result_field_limit": field_limit,
        }
        encoded = (json.dumps(retained, sort_keys=True) + "\n").encode()
        if len(encoded) <= MAX_ARTIFACT:
            return store.exact(relative, encoded)
    summary = (json.dumps(_journal_summary(journal, source_sha256), sort_keys=True) + "\n").encode()
    if len(summary) > MAX_ARTIFACT:  # A fixed digest-only last resort cannot overflow.
        dispatches, calls = journal.get("dispatches", []), journal.get("calls", [])
        structural = deepcopy(journal)
        for collection in ("dispatches", "calls"):
            for entry in structural.get(collection, []):
                if isinstance(entry, dict) and isinstance(entry.get("result"), dict):
                    for field in ("stdout", "stderr", "content"):
                        entry["result"].pop(field, None)
        summary = (json.dumps({
            "schema": "profiling-skill/compact-remote-journal/v1",
            "source_canonical_sha256": source_sha256,
            "retention": {
                "mode": "digest-only-summary",
                "dispatch_count": len(dispatches) if isinstance(dispatches, list) else None,
                "call_count": len(calls) if isinstance(calls, list) else None,
                "structural_metadata_sha256": _canonical_sha256(structural),
            },
        }, sort_keys=True) + "\n").encode()
    return store.exact(relative, summary)


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def text(self, relative: str, value: object) -> dict:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise LaunchError("artifact path escaped retained root")
        text = str(value)
        if not text or len(text.encode()) > MAX_ARTIFACT:
            raise LaunchError("artifact exceeds bounded retention limit or is empty")
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_text(text)
        os.replace(temporary, target)
        return {"path": path.as_posix(), "sha256": _sha(target)}

    def json(self, relative: str, value: object) -> dict:
        return self.text(relative, json.dumps(value, sort_keys=True) + "\n")

    def exact(self, relative: str, value: bytes) -> dict:
        """Retain already-validated bounded bytes without reserialization."""
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise LaunchError("artifact path escaped retained root")
        if not value or len(value) > MAX_ARTIFACT:
            raise LaunchError("artifact exceeds bounded retention limit or is empty")
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        temporary.write_bytes(value)
        os.replace(temporary, target)
        return {"path": path.as_posix(), "sha256": _sha(target)}


def classify_failure(document: dict) -> tuple[str, str, str]:
    """Map trusted process/controller evidence; unknown failures count.

    ``trusted_stdout`` and ``trusted_stderr`` scope text classification to the
    selected remote dispatch.  The untrusted aggregate remains useful as a
    retained diagnostic, but agent commentary in it cannot relabel a concrete
    remote failure.
    """
    state = str(document.get("state", "")).lower().replace("_", "-")
    failure = str(document.get("failure_type", "")).lower().replace("_", "-")
    if state in INFRA_STATES or failure in {
            "transport", "device-busy", "target-unavailable", "observer"}:
        normalized = ("device_busy" if "device-busy" in {state, failure}
                      else "target_unavailable" if "target-unavailable" in {state, failure}
                      else "transport" if "transport" in {state, failure}
                      else "observer")
        return "discarded_infrastructure", normalized, (
            "observer" if normalized == "observer" else "transport")
    if failure in {"compile", "runtime", "profiler-command", "evidence"}:
        normalized = failure.replace("-", "_")
        return "counted_failure", normalized, (
            "profiler" if normalized == "profiler_command" else normalized)
    trusted = "trusted_stdout" in document or "trusted_stderr" in document
    stdout_key = "trusted_stdout" if trusted else "stdout"
    stderr_key = "trusted_stderr" if trusted else "stderr"
    text = f"{document.get(stdout_key, '')}\n{document.get(stderr_key, '')}".lower()
    profiler_diagnostic = any(marker in text for marker in (
        "profile_command=msprof ", "profiler failure", "msprof op --help",
        "msprof op simulator --help",
    ))
    if profiler_diagnostic:
        return "counted_failure", "profiler_command", "profiler"
    missing_selector = (
        "no exported kernel row" in text
        or ("expected one exported" in text and "selector" in text)
        or any(marker in text for marker in (
            "missing selector", "missing kernel selector",
            "missing exported selector", "missing exported kernel selector",
        ))
    )
    if missing_selector:
        return "counted_failure", "evidence", "evidence"
    if "compile" in text:
        return "counted_failure", "compile", "compile"
    if "evidence" in text:
        return "counted_failure", "evidence", "evidence"
    if document.get("returncode") not in (None, 0):
        return "counted_failure", "launcher", "launcher"
    return "counted_failure", "runtime", "runtime"


def counted_dispatch_failure(journal: dict, journal_start: tuple[int, int],
                             target: str) -> dict | None:
    """Return the first current-turn counted remote failure, even after a retry."""
    dispatches = journal.get("dispatches", [])[journal_start[0]:]
    calls = journal.get("calls", [])[journal_start[1]:]
    for dispatch in dispatches:
        result = dispatch.get("result")
        if dispatch.get("target") != target or not isinstance(result, dict) \
                or (dispatch.get("state") not in {"failed", "cancelled"}
                    and result.get("returncode") in (None, 0)):
            continue
        evidence = dispatch_failure_evidence(
            {"dispatches": dispatches, "calls": calls}, dispatch, 0)
        if classify_failure(evidence)[0] == "counted_failure":
            return dispatch
    return None


def dispatch_failure_evidence(journal: dict, dispatch: dict, calls_start: int) -> dict:
    """Build failure evidence from one dispatch and only calls for its retained handle."""
    result = dispatch.get("result") if isinstance(dispatch.get("result"), dict) else {}
    handle = dispatch.get("handle")
    related = [entry.get("result") for entry in journal.get("calls", [])[calls_start:]
               if handle and entry.get("handle") == handle
               and entry.get("operation") in {"observe", "logs", "result"}]
    related = [entry for entry in related if isinstance(entry, dict)]
    records = [result, *related]
    structured = next((entry.get("failure_type") for entry in records
                       if entry.get("failure_type")), None)
    return {
        "returncode": result.get("returncode"),
        "state": result.get("state") or dispatch.get("state"),
        "failure_type": structured,
        "stdout": "\n".join(str(entry.get("stdout", "")) for entry in records),
        "stderr": "\n".join(str(entry.get("stderr", "")) for entry in records),
    }


def retained_content_sha256(journal: dict, dispatch: dict) -> str | None:
    """Read one unambiguous content digest from a dispatch and its own retained calls."""
    handle = dispatch.get("handle")
    results = [dispatch.get("result")]
    results += [call.get("result") for call in journal.get("calls", [])
                if handle and call.get("handle") == handle
                and call.get("operation") in {"observe", "logs", "result"}]
    hashes = {_parse_remote(str(result.get("stdout", ""))).get("content_sha256")
              for result in results if isinstance(result, dict)}
    hashes = {value for value in hashes if HEX.fullmatch(str(value))}
    return next(iter(hashes)) if len(hashes) == 1 else None


def failure_record(*, store: ArtifactStore, session_id: str, kind: str, arm: str | None,
                   classification: str, failure_type: str, stage: str,
                   product: str | None, target: str | None, handle: str | None,
                   command: dict, log: dict, diagnostic: dict, manifest_sha256: str,
                   launcher_identity: dict, model_identity: dict,
                   skill_sha256: str) -> dict:
    receipt = {
        "schema": RECEIPT_SCHEMA, "session_id": session_id, "kind": kind, "arm": arm,
        "classification": classification, "failure_type": failure_type, "stage": stage,
        "product": product, "target": target, "handle": handle,
        "artifacts": {"command": command["sha256"], "log": log["sha256"],
                      "diagnostic": diagnostic["sha256"]},
    }
    row = {
        "kind": kind, "session_id": session_id, "classification": classification,
        "failure_type": failure_type, "stage": stage, "product": product,
        "target": target, "handle": handle, "command": command, "log": log,
        "diagnostic": diagnostic,
        "receipt": store.json(f"receipts/{session_id}.json", receipt),
        "manifest_sha256": manifest_sha256, "launcher": launcher_identity,
        "model": model_identity, "skill_sha256": skill_sha256,
    }
    if arm is not None:
        row["arm"] = arm
    return row


def review_queue(items: list[dict], reviewer: dict) -> dict:
    if set(reviewer) != {"identity", "config_sha256"} \
            or not HEX.fullmatch(str(reviewer["config_sha256"])):
        raise LaunchError("invalid pinned reviewer")
    return {"status": "review_pending", "reviewer": reviewer, "items": items}


def finalize_reviews(draft: dict, decisions: dict, reviewer: dict,
                     store: ArtifactStore) -> dict:
    """Attach external human/agent review decisions without trusting test agents."""
    review_queue([], reviewer)
    result = deepcopy(draft)
    rows = result.get("outcomes") if result.get("kind") == "acquisition" else result.get("answers")
    if not isinstance(rows, list):
        raise LaunchError("draft has no reviewable outcomes")
    for row in rows:
        unit = row.get("product") if result["kind"] == "acquisition" else row.get("case_id")
        decision = decisions.get(unit)
        if not isinstance(decision, dict) or set(decision) != {"passed", "notes"} \
                or not isinstance(decision["passed"], bool) \
                or not isinstance(decision["notes"], str) or not decision["notes"].strip():
            raise LaunchError(f"missing valid manual review for {unit}")
        row["manual_review"] = {
            "passed": decision["passed"], "reviewer": reviewer,
            "notes": store.text(f"review/{result['session_id']}-{unit}.txt",
                                decision["notes"].strip() + "\n"),
        }
    if set(decisions) != {row.get("product") if result["kind"] == "acquisition"
                         else row.get("case_id") for row in rows}:
        raise LaunchError("manual review contains unknown units")
    return result


def write_request_example(root: Path) -> Path:
    """Create a self-contained, byte-pinned interpretation request example."""
    if root.exists() and any(root.iterdir()):
        raise LaunchError("request example output directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    prompt = root / "prompt.md"
    prompt.write_text("Interpret the supplied compact profiling evidence.\n")
    skill = root / "ascend-profiling"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: ascend-profiling\ndescription: Example profiling skill.\n---\n")
    units = []
    for index, product in enumerate(("a3", "a3", "a5", "a5"), 1):
        evidence = root / f"case-{index}.json"
        evidence.write_text(json.dumps({"schema": "compact-example/v1", "value": index}) + "\n")
        units.append({"case_id": f"case-{index}", "product": product,
                      "evidence": str(evidence.resolve()), "evidence_sha256": _sha(evidence)})
    model = {"name": "gpt-5.6-sol", "reasoning_effort": "low"}
    model_hash = hashlib.sha256(
        json.dumps(model, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request = {
        "schema": REQUEST_SCHEMA, "kind": "interpretation", "session_id": "example-session",
        "arm": "candidate", "manifest_sha256": "1" * 64,
        "prompt": str(prompt.resolve()), "prompt_sha256": _sha(prompt),
        "skill": str(skill.resolve()), "skill_sha256": _sha(skill),
        "launcher": {"identity": "profiling-behavioral-launcher/v1",
                     "sha256": launcher_digest()},
        "model": {"identity": "example-model", "config_sha256": model_hash, **model},
        "reviewer": {"identity": "example-reviewer/v1", "config_sha256": "2" * 64},
        "allowed_targets": {"a3": ["bz-a3-1", "bz-a3-2"], "a5": ["bz-a5"]},
        "units": units,
    }
    path = root / "request.json"
    _atomic_json(path, request)
    return path


def _parse_remote(output: str) -> dict:
    result: dict[str, str] = {}
    try:
        value = json.loads(output)
    except json.JSONDecodeError:
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            if separator and key.startswith("REMOTE_"):
                result[key.removeprefix("REMOTE_").lower()] = value
    else:
        if isinstance(value, dict):
            result = {str(key).removeprefix("REMOTE_").lower(): str(item)
                      for key, item in value.items() if item is not None}
    return result


def _remote_failure_type(operation: str, parsed: dict, raw: CommandResult) -> str | None:
    supplied = str(parsed.get("failure_type", "")).lower().replace("-", "_")
    if supplied in {"transport", "device_busy", "target_unavailable", "observer",
                    "compile", "runtime", "profiler_command", "evidence"}:
        return supplied
    state = str(parsed.get("state", "")).lower().replace("_", "-")
    if state in {"observation-unavailable", "reconnecting"}:
        return "observer"
    if state == "device-busy":
        return "device_busy"
    if state in {"target-unavailable", "missing"}:
        return "target_unavailable"
    if raw.returncode == 124:
        return "observer" if operation in {"observe", "logs", "result"} else "transport"
    diagnostic = raw.stderr.lower()
    if any(marker in diagnostic for marker in (
            "remote transport unavailable", "connection timed out", "connection refused",
            "no route to host", "could not resolve hostname", "network is unreachable",
            "connection reset", "broken pipe")):
        return "transport"
    return None


class RemoteBroker:
    """Allowlisted host-side cpl-remote bridge with durable dispatch checkpoints."""

    def __init__(self, command: Sequence[str], journal: Path, workspace: Path,
                 allowed_targets: set[str] | None = None, timeout: int = 900):
        if not command:
            raise LaunchError("remote-access command is required")
        self.command = list(command)
        self.journal = journal
        self.workspace = workspace.resolve()
        self.configured_targets = None if allowed_targets is None else set(allowed_targets)
        self.allowed_targets = None if allowed_targets is None else set(allowed_targets)
        self.timeout = timeout
        self.lock = threading.Lock()
        if not journal.exists():
            _atomic_json(journal, {"schema": "profiling-skill/remote-journal/v1",
                                   "dispatches": [], "calls": []})

    def _load(self) -> dict:
        value = json.loads(self.journal.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("dispatches"), list):
            raise LaunchError("invalid remote checkpoint journal")
        return value

    def restrict_targets(self, targets: set[str]) -> None:
        """Restrict subsequent calls to the target assigned to the active turn."""
        with self.lock:
            if self.configured_targets is not None and not targets <= self.configured_targets:
                raise LaunchError("active target is outside the launch allowlist")
            self.allowed_targets = set(targets)

    def _validate(self, arguments: Sequence[str]) -> tuple[
            str, str | None, list[str], str | None, str | None, bytes | None]:
        if not arguments or arguments[0] not in {
                "capabilities", "preflight", "run", "observe", "logs", "result"}:
            raise LaunchError("unsupported remote operation")
        operation = arguments[0]
        forbidden_options = {"--token", "--password", "--identity-file", "--private-key"}
        if any(item.split("=", 1)[0].lower() in forbidden_options for item in arguments):
            raise LaunchError("credentials are forbidden in remote command arguments")
        target = None
        if operation in {"capabilities", "preflight", "run"}:
            if len(arguments) < 2:
                raise LaunchError("remote target is required")
            target = arguments[1]
        elif len(arguments) < 2 or not HANDLE.fullmatch(arguments[1]):
            raise LaunchError("valid retained handle is required")
        if operation in {"observe", "logs", "result"} and self.allowed_targets is not None:
            handle_target = ("gz-a3" if arguments[1].startswith("gz-a3:")
                             else arguments[1].split(":", 3)[1])
            if handle_target not in self.allowed_targets:
                raise LaunchError("unapproved target in retained handle")
        if target and self.allowed_targets is not None and target not in self.allowed_targets:
            raise LaunchError("unapproved target")
        agent_arguments = list(arguments)
        dispatch_key = None
        file_sha256 = None
        file_bytes = None
        if operation == "run" and ("--file" not in arguments or "--" in arguments):
            raise LaunchError("remote run must use a file-backed payload")
        if operation == "run":
            if agent_arguments.count("--dispatch-key") != 1:
                raise LaunchError("one explicit dispatch key is required for remote run")
            key_index = agent_arguments.index("--dispatch-key")
            if key_index + 1 >= len(agent_arguments) \
                    or not IDENTIFIER.fullmatch(agent_arguments[key_index + 1]):
                raise LaunchError("valid dispatch key is required for remote run")
            dispatch_key = agent_arguments[key_index + 1]
            del agent_arguments[key_index:key_index + 2]
            file_index = agent_arguments.index("--file") + 1
            if (file_index >= len(agent_arguments)
                    or not agent_arguments[file_index].startswith("/workspace/")):
                raise LaunchError("workspace path is required for local inputs")
        elif "--dispatch-key" in agent_arguments:
            raise LaunchError("dispatch key is valid only for remote run")
        mapped = list(agent_arguments)
        local_indices = ({agent_arguments.index("--file") + 1} if operation == "run" else set())
        for index in local_indices:
            item = mapped[index]
            if item == "/workspace" or item.startswith("/workspace/"):
                relative = Path(item).relative_to("/workspace")
                host = (self.workspace / relative).resolve()
                if not host.is_relative_to(self.workspace):
                    raise LaunchError("workspace path escaped")
                if host.is_symlink() or not host.is_file():
                    raise LaunchError("remote payload must be a regular workspace file")
                mapped[index] = str(host)
                file_bytes = host.read_bytes()
                file_sha256 = hashlib.sha256(file_bytes).hexdigest()
        return operation, target, mapped, dispatch_key, file_sha256, file_bytes

    def execute(self, arguments: Sequence[str]) -> dict:
        operation, target, mapped, dispatch_key, file_sha256, file_bytes = \
            self._validate(arguments)
        cpl_arguments = list(arguments)
        if operation == "run":
            key_index = cpl_arguments.index("--dispatch-key")
            del cpl_arguments[key_index:key_index + 2]
            identity = {"dispatch_key": dispatch_key, "arguments": cpl_arguments,
                        "file_sha256": file_sha256}
        else:
            identity = {"arguments": cpl_arguments}
        request_hash = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        with self.lock:
            journal = self._load()
            if operation in {"observe", "logs", "result"}:
                known = {entry.get("handle") for entry in journal["dispatches"]}
                if arguments[1] not in known:
                    raise LaunchError("retained handle was not dispatched by this session")
            prior = next((entry for entry in journal["dispatches"]
                          if entry.get("dispatch_key") == dispatch_key), None)
            observed = None
            if operation == "run" and prior:
                if (prior.get("arguments") != cpl_arguments
                        or prior.get("file_sha256") != file_sha256):
                    raise LaunchError(
                        "dispatch key was reused after argv or payload bytes changed")
                if not prior.get("handle"):
                    raise LaunchError(
                        "uncertain dispatch has no retained handle; never resubmit this request; "
                        f"inspect journal entry {request_hash}")
                if prior.get("state") in TERMINAL:
                    return dict(prior["result"])
                mapped = ["observe", prior["handle"], "--wait"]
            elif operation == "run":
                payload_dir = self.journal.parent / "broker-payloads"
                payload_dir.mkdir(mode=0o700, exist_ok=True)
                payload = payload_dir / f"{dispatch_key}-{file_sha256}.sh"
                if not payload.exists():
                    temporary = payload.with_name(f".{payload.name}.tmp-{os.getpid()}")
                    temporary.write_bytes(file_bytes or b"")
                    temporary.chmod(0o600)
                    os.replace(temporary, payload)
                if payload.is_symlink() or _sha(payload) != file_sha256:
                    raise LaunchError("broker payload snapshot identity mismatch")
                mapped[mapped.index("--file") + 1] = str(payload)
                prior = {"request_sha256": request_hash, "target": target,
                         "dispatch_key": dispatch_key, "file_sha256": file_sha256,
                         "arguments": cpl_arguments,
                         "state": "dispatching", "handle": None, "result": None}
                journal["dispatches"].append(prior)
                _atomic_json(self.journal, journal)
            elif operation in {"observe", "result"}:
                observed = next(entry for entry in journal["dispatches"]
                                if entry.get("handle") == arguments[1])
            call_target = target
            if call_target is None and operation in {"observe", "logs", "result"}:
                call_target = ("gz-a3" if arguments[1].startswith("gz-a3:")
                               else arguments[1].split(":", 3)[1])
            call = {"operation": mapped[0], "target": call_target,
                    "arguments": list(arguments),
                    "request_sha256": request_hash,
                    "handle": mapped[1] if mapped[0] in {"observe", "logs", "result"} else None,
                    "result": None}
            journal["calls"].append(call)
            _atomic_json(self.journal, journal)
            try:
                run = subprocess.run([*self.command, *mapped], text=True, capture_output=True,
                                     check=False, timeout=self.timeout, cwd=self.workspace)
                raw = CommandResult(run.returncode, run.stdout, run.stderr)
            except subprocess.TimeoutExpired as error:
                raw = CommandResult(124, str(error.stdout or ""), str(error.stderr or ""))
            parsed = _parse_remote(raw.stdout)
            result = {"returncode": raw.returncode, "stdout": _bounded(raw.stdout),
                      "stderr": _bounded(raw.stderr), "state": parsed.get("state", ""),
                      "handle": parsed.get("handle"), "exit": parsed.get("exit"),
                      "reason": parsed.get("reason"),
                      "failure_type": _remote_failure_type(mapped[0], parsed, raw)}
            dispatch = prior if operation == "run" else observed
            if dispatch is not None:
                handle = result["handle"] or dispatch.get("handle")
                if handle and not HANDLE.fullmatch(handle):
                    raise LaunchError("remote-access returned an invalid handle")
                if observed is not None and handle != observed["handle"]:
                    raise LaunchError("observation returned a different retained handle")
                state = result["state"] or dispatch["state"]
                if raw.returncode == 124 and not handle:
                    state = "uncertain"
                dispatch.update(handle=handle, state=state, result=result)
            call["result"] = result
            _atomic_json(self.journal, journal)
            return result


class _BrokerHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        broker: RemoteBroker = self.server.broker  # type: ignore[attr-defined]
        try:
            request = json.loads(self.rfile.readline())
            arguments = request["arguments"]
            if not isinstance(arguments, list) or not all(isinstance(item, str) for item in arguments):
                raise LaunchError("arguments must be strings")
            response = broker.execute(arguments)
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, LaunchError) as error:
            response = {"returncode": 4, "stdout": "", "stderr": _bounded(error),
                        "state": "", "handle": None}
        self.wfile.write(json.dumps(response).encode() + b"\n")


def broker_client(socket_path: Path, arguments: Sequence[str]) -> int:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(str(socket_path))
        client.sendall(json.dumps({"arguments": list(arguments)}).encode() + b"\n")
        response = json.loads(client.makefile("rb").readline())
    print(response.get("stdout", ""), end="")
    print(response.get("stderr", ""), end="", file=sys.stderr)
    if response.get("handle"):
        print(f"REMOTE_HANDLE={response['handle']}")
    if response.get("state"):
        print(f"REMOTE_STATE={response['state']}")
    return int(response["returncode"])


class BehavioralLauncher(ProductionLauncher):
    """Specialize the audited production boundary for blinded behavioral gates."""

    def __init__(self, remote_command: Sequence[str], **kwargs):
        super().__init__(["unused"], **kwargs)
        self.remote_command = list(remote_command)

    def validate_request(self, request: dict) -> dict:
        keys = {"schema", "kind", "session_id", "arm", "manifest_sha256", "prompt",
                "prompt_sha256", "skill", "skill_sha256", "launcher", "model",
                "reviewer", "allowed_targets", "units"}
        if not isinstance(request, dict) or set(request) != keys or request["schema"] != REQUEST_SCHEMA:
            raise LaunchError("invalid behavioral launch request")
        if request["kind"] not in {"acquisition", "interpretation"} \
                or not isinstance(request["session_id"], str) \
                or not IDENTIFIER.fullmatch(request["session_id"]):
            raise LaunchError("invalid behavioral session")
        for name in ("manifest_sha256", "prompt_sha256", "skill_sha256"):
            if not HEX.fullmatch(str(request[name])):
                raise LaunchError(f"invalid {name}")
        prompt, skill = Path(request["prompt"]), Path(request["skill"])
        if prompt.is_symlink() or not prompt.is_file() or _sha(prompt) != request["prompt_sha256"]:
            raise LaunchError("frozen prompt changed")
        if (not skill.is_dir() or skill.is_symlink()
                or any(path.is_symlink() for path in skill.rglob("*"))
                or _sha(skill) != request["skill_sha256"]):
            raise LaunchError("selected skill changed")
        if set(request["launcher"]) != {"identity", "sha256"} \
                or request["launcher"]["identity"] != "profiling-behavioral-launcher/v1" \
                or request["launcher"]["sha256"] != launcher_digest():
            raise LaunchError("invalid launcher identity")
        model_config = json.dumps({"name": request["model"].get("name"),
                                   "reasoning_effort": request["model"].get("reasoning_effort")},
                                  sort_keys=True, separators=(",", ":")).encode()
        if set(request["model"]) != {"identity", "config_sha256", "name", "reasoning_effort"} \
                or request["model"]["config_sha256"] != hashlib.sha256(model_config).hexdigest() \
                or not all(isinstance(request["model"][key], str) and request["model"][key]
                           for key in ("identity", "name", "reasoning_effort")):
            raise LaunchError("invalid model identity")
        targets = request["allowed_targets"]
        if not isinstance(targets, dict) or set(targets) != {"a3", "a5"} \
                or any(not isinstance(value, list) or not value for value in targets.values()):
            raise LaunchError("invalid allowed targets")
        flattened = [target for product_targets in targets.values()
                     for target in product_targets]
        if any(not isinstance(target, str) or not IDENTIFIER.fullmatch(target)
               for target in flattened):
            raise LaunchError("allowed targets require valid target IDs")
        if any(len(product_targets) != len(set(product_targets))
               for product_targets in targets.values()):
            raise LaunchError("allowed target IDs must be unique within each product")
        units = request["units"]
        if request["kind"] == "acquisition" and isinstance(units, list) \
                and len(units) == 2 and all(isinstance(unit, dict) for unit in units) \
                and len({unit.get("target") for unit in units}) != len(units):
            raise LaunchError("acquisition requires distinct assigned targets")
        if set(targets["a3"]) & set(targets["a5"]):
            raise LaunchError("allowed target IDs must be product-scoped")
        review_queue([], request["reviewer"])
        if request["kind"] == "acquisition":
            if request["arm"] is not None or not isinstance(units, list) or len(units) != 2 \
                    or [unit.get("product") for unit in units if isinstance(unit, dict)] != ["a3", "a5"]:
                raise LaunchError("acquisition requires paired A3 and A5 units")
            for unit in units:
                if set(unit) != {"product", "target"} \
                        or unit["target"] not in targets[unit["product"]]:
                    raise LaunchError("unapproved acquisition target")
        else:
            if request["arm"] not in {"current", "candidate"} or not isinstance(units, list) \
                    or len(units) != 4:
                raise LaunchError("interpretation requires one arm and four cases")
            for unit in units:
                if set(unit) != {"case_id", "product", "evidence", "evidence_sha256"}:
                    raise LaunchError("invalid interpretation case")
                evidence = Path(unit["evidence"])
                if evidence.is_symlink() or not evidence.is_file() \
                        or _sha(evidence) != unit["evidence_sha256"]:
                    raise LaunchError("frozen evidence changed")
            case_ids = [unit["case_id"] for unit in units]
            if any(not isinstance(case_id, str) or not IDENTIFIER.fullmatch(case_id)
                   for case_id in case_ids) \
                    or len(set(case_ids)) != len(case_ids):
                raise LaunchError("interpretation requires unique case IDs")
            if [unit["product"] for unit in units].count("a3") != 2 \
                    or [unit["product"] for unit in units].count("a5") != 2:
                raise LaunchError("interpretation requires exactly two A3 and two A5 cases")
        return request

    def turn_plan(self, request: dict) -> list[dict]:
        turns = []
        for index, unit in enumerate(request["units"]):
            if request["kind"] == "acquisition":
                payload = {"product": unit["product"], "target": unit["target"],
                           "prompt_sha256": request["prompt_sha256"]}
            else:
                payload = {"case_id": unit["case_id"], "product": unit["product"],
                           "prompt_sha256": request["prompt_sha256"],
                           "evidence_sha256": unit["evidence_sha256"]}
            turns.append({"resume": index > 0, "payload": payload, "unit": unit})
        return turns

    def prepare(self, root: Path, request: dict) -> LaunchPaths:
        workspace = root / "workspace"
        workspace.mkdir(parents=True)
        frozen = root / "frozen"
        frozen.mkdir()
        prompt = frozen / "prompt.md"
        shutil.copy2(request["prompt"], prompt)
        skill = frozen / "ascend-profiling"
        shutil.copytree(request["skill"], skill, copy_function=shutil.copy2)
        if _sha(prompt) != request["prompt_sha256"] or _sha(skill) != request["skill_sha256"]:
            raise LaunchError("frozen prompt or selected skill changed during snapshot")
        skill_target = workspace / ".agents" / "skills" / "ascend-profiling"
        skill_target.mkdir(parents=True)
        state = root / "codex-state"
        state.mkdir(mode=0o700)
        (state / "auth.json").touch(mode=0o600)
        socket_dir = root / "broker"
        socket_dir.mkdir(mode=0o700)
        wrapper = root / "cpl-remote"
        wrapper.write_text("#!/bin/sh\nexec python3 /experiment/launcher.py broker-client "
                           "/experiment-state/broker.sock \"$@\"\n")
        wrapper.chmod(0o755)
        public = {"schema": "profiling-skill/agent-turns/v1",
                  "turns": [turn["payload"] for turn in self.turn_plan(request)]}
        _atomic_json(workspace / "request.json", public)
        for unit in request["units"]:
            if "evidence" in unit:
                copied = workspace / f"{unit['case_id']}.json"
                shutil.copy2(unit["evidence"], copied)
                if _sha(copied) != unit["evidence_sha256"]:
                    raise LaunchError("frozen evidence changed during snapshot")
        return LaunchPaths(root, workspace, prompt, skill, state, socket_dir, wrapper,
                           root / "remote-journal.json")

    def base_command(self, paths: LaunchPaths, request: dict) -> list[str]:
        runtime = self._runtime_mount()
        script = Path(__file__).resolve()
        production = script.with_name("production_launcher.py")
        command = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-all",
                   "--share-net", "--clearenv", "--tmpfs", "/", "--proc", "/proc",
                   "--dev", "/dev", "--tmpfs", "/tmp"]
        for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(source).exists():
                command += ["--ro-bind", source, source]
        command += self._resolver_mounts()
        command += ["--dir", "/home", "--dir", "/home/agent", "--dir", "/runtime",
                    "--dir", "/experiment", "--dir", "/tools",
                    "--bind", str(paths.workspace), "/workspace",
                    "--ro-bind", str(paths.skill),
                    "/workspace/.agents/skills/ascend-profiling",
                    "--bind", str(paths.state), "/codex-home",
                    "--ro-bind", str(self.auth_home / "auth.json"), "/codex-home/auth.json",
                    "--ro-bind", str(runtime), "/runtime/node",
                    "--ro-bind", str(script), "/experiment/launcher.py",
                    "--ro-bind", str(production), "/experiment/production_launcher.py",
                    "--ro-bind", str(paths.wrapper), "/tools/cpl-remote",
                    "--bind", str(paths.socket_dir), "/experiment-state",
                    "--chdir", "/workspace", "--setenv", "HOME", "/home/agent",
                    "--setenv", "CODEX_HOME", "/codex-home", "--setenv", "PATH",
                    "/tools:/runtime/node/bin:/usr/bin:/bin", "--setenv",
                    "PROFILE_GATE_REQUEST", "/workspace/request.json", "--setenv",
                    "PROFILE_GATE_OUTPUT", "/workspace/agent-output.json", "--setenv",
                    "CPL_REMOTE_MODE", "retained-broker"]
        return command

    @staticmethod
    def sandbox_preflight(base: Sequence[str]) -> None:
        checks = ["test -r /codex-home/auth.json",
                  "test -r /workspace/.agents/skills/ascend-profiling/SKILL.md",
                  "test ! -e /codex-home/skills", "test ! -e /codex-home/plugins",
                  "test ! -e /home/agent/.agents/skills", "command -v cpl-remote",
                  "python3 -c \"import socket; assert socket.getaddrinfo('api.openai.com',443)\""]
        run = subprocess.run([*base, "sh", "-ceu", ";".join(checks)], text=True,
                             capture_output=True, check=False)
        if run.returncode:
            raise LaunchError(f"outer isolation preflight failed: {_bounded(run.stderr)}")

    @staticmethod
    def _answer(stdout: str) -> dict:
        answer = None
        for line in stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = event.get("item")
            if event.get("type") == "item.completed" and isinstance(item, dict) \
                    and item.get("type") == "agent_message":
                try:
                    candidate = json.loads(item.get("text", ""))
                except json.JSONDecodeError:
                    continue
                if isinstance(candidate, dict):
                    answer = candidate
        if answer is None:
            raise LaunchError("Codex did not return one structured answer")
        return answer

    @staticmethod
    def _agent_file(workspace: Path, value: object) -> Path:
        if not isinstance(value, str):
            raise LaunchError("agent evidence path must be text")
        relative = Path(value.removeprefix("/workspace/"))
        path = (workspace / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(workspace) \
                or path.is_symlink() or not path.is_file():
            raise LaunchError("agent evidence escaped its workspace")
        return path

    def _success_draft(self, request: dict, turns: list[dict], paths: LaunchPaths,
                       store: ArtifactStore, codex_thread_id: str) -> tuple[dict, list[dict]]:
        common = {"kind": request["kind"], "session_id": codex_thread_id,
                  "classification": "success", "manifest_sha256": request["manifest_sha256"],
                  "launcher": request["launcher"], "model": {
                      key: request["model"][key] for key in ("identity", "config_sha256")},
                  "skill_sha256": request["skill_sha256"]}
        review_items = []
        journal = json.loads(paths.journal.read_text())
        retained = {entry.get("handle"): entry for entry in journal["dispatches"]
                    if entry.get("handle")}
        if request["kind"] == "acquisition":
            outcomes = []
            for index, turn in enumerate(turns, 1):
                answer, unit = turn["answer"], request["units"][index - 1]
                if (set(answer) != {"product", "target", "handle", "evidence", "reasoning"}
                        or answer["product"] != unit["product"]
                        or answer["target"] != unit["target"]
                        or answer["handle"] not in retained
                        or retained[answer["handle"]].get("target") != unit["target"]):
                    raise LaunchError("acquisition answer does not match retained remote evidence")
                source = self._agent_file(paths.workspace, answer["evidence"])
                evidence = json.loads(source.read_text())
                provenance = {"product": unit["product"], "target": unit["target"]}
                if not isinstance(evidence, dict) or not isinstance(evidence.get("schema"), str) \
                        or evidence.get("provenance") != provenance:
                    raise LaunchError("acquisition evidence lacks exact provenance")
                reasoning = store.text(
                    f"reasoning/{request['session_id']}-{unit['product']}.txt",
                    _bounded(answer["reasoning"]))
                dispatch = retained[answer["handle"]]
                remote_result = dispatch.get("result")
                if (dispatch.get("state") != "completed" or not isinstance(remote_result, dict)
                        or remote_result.get("state") != "completed"
                        or remote_result.get("returncode") != 0):
                    raise LaunchError(
                        "acquisition evidence requires a terminal successful dispatch")
                emitted_hash = retained_content_sha256(journal, dispatch)
                if not HEX.fullmatch(str(emitted_hash)) or _sha(source) != emitted_hash:
                    raise LaunchError(
                        "acquisition remote evidence bytes do not match retained job hash")
                evidence_ref = store.exact(
                    f"evidence/{request['session_id']}-{unit['product']}.json",
                    source.read_bytes())
                evidence_ref.update(schema=evidence["schema"], provenance=provenance,
                                    remote_sha256=emitted_hash)
                remote_command = store.text(
                    f"commands/{request['session_id']}-{unit['product']}.txt",
                    shlex.join(["cpl-remote", *dispatch["arguments"]]) + "\n")
                combined_log = (f"remote stdout:\n{remote_result.get('stdout', '')}\n"
                                f"remote stderr:\n{remote_result.get('stderr', '')}\n"
                                f"agent log sha256: {turn['log']['sha256']}\n")
                remote_log = store.text(
                    f"logs/{request['session_id']}-{unit['product']}-remote.txt",
                    _bounded(combined_log))
                outcome = {"session_id": codex_thread_id, "product": unit["product"],
                           "target": unit["target"], "handle": answer["handle"],
                           "evidence": evidence_ref, "command": remote_command,
                           "log": remote_log, "reasoning": reasoning,
                           "agent_payload": turn["payload"], "manual_review": None}
                outcomes.append(outcome)
                review_items.append({"session_id": codex_thread_id,
                                     "unit": unit["product"], "reasoning": reasoning})
            return {**common, "outcomes": outcomes}, review_items
        answers = []
        for index, turn in enumerate(turns, 1):
            answer, unit = turn["answer"], request["units"][index - 1]
            if set(answer) != {"case_id", "conclusions", "saturation_claims", "reasoning"} \
                    or answer["case_id"] != unit["case_id"] \
                    or not isinstance(answer["conclusions"], list) \
                    or not isinstance(answer["saturation_claims"], list):
                raise LaunchError("interpretation answer has invalid shape or case identity")
            reasoning = store.text(f"reasoning/{request['session_id']}-{unit['case_id']}.txt",
                                   _bounded(answer["reasoning"]))
            answers.append({"case_id": unit["case_id"],
                            "evidence_sha256": unit["evidence_sha256"],
                            "conclusions": answer["conclusions"],
                            "saturation_claims": answer["saturation_claims"],
                            "log": turn["log"], "reasoning": reasoning,
                            "agent_payload": turn["payload"], "manual_review": None})
            review_items.append({"session_id": codex_thread_id,
                                 "unit": unit["case_id"], "reasoning": reasoning})
        return {**common, "arm": request["arm"], "answers": answers}, review_items

    def _failure_result(self, request: dict, paths: LaunchPaths, store: ArtifactStore,
                        active_unit: dict | None, error: object, *, codex_command: str = "",
                        codex_stdout: str = "", codex_stderr: str = "",
                        codex_returncode: int | None = None,
                        journal_start: tuple[int, int] = (0, 0),
                        failed_handle: str | None = None) -> dict:
        journal = json.loads(paths.journal.read_text())
        acquisition = request["kind"] == "acquisition" and isinstance(active_unit, dict)
        target = active_unit.get("target") if acquisition else None
        product = active_unit.get("product") if acquisition else None
        dispatches = journal.get("dispatches", [])[journal_start[0]:] if acquisition else []
        dispatches = [entry for entry in dispatches if entry.get("target") == target]
        handles = {entry.get("handle") for entry in dispatches if entry.get("handle")}
        calls = journal.get("calls", [])[journal_start[1]:] if acquisition else []
        calls = [entry for entry in calls
                 if entry.get("target") == target or entry.get("handle") in handles]
        dispatch = (next((entry for entry in dispatches
                          if entry.get("handle") == failed_handle), {})
                    if failed_handle else dispatches[-1] if dispatches else {})
        handle = dispatch.get("handle")
        selected = dispatch_failure_evidence(journal, dispatch, journal_start[1]) \
            if dispatch else {"returncode": None, "state": "", "failure_type": None,
                              "stdout": "", "stderr": ""}
        remote_text = (f"remote result:\nstdout:\n{selected['stdout']}\n"
                       f"stderr:\n{selected['stderr']}")
        combined_stdout = f"{remote_text}\ncodex stdout:\n{codex_stdout}"
        combined_stderr = f"codex stderr:\n{codex_stderr}\nlauncher diagnostic:\n{error}"
        evidence = {
            "returncode": (codex_returncode if codex_returncode not in (None, 0)
                           else selected.get("returncode", 1)),
            "state": selected.get("state", ""),
            "failure_type": selected.get("failure_type"),
            "stdout": combined_stdout, "stderr": combined_stderr,
        }
        if dispatch:
            evidence.update(trusted_stdout=selected.get("stdout", ""),
                            trusted_stderr=selected.get("stderr", ""))
        classification, failure_type, stage = classify_failure(evidence)
        label = request["session_id"]
        command = store.text(
            f"failures/{label}-command.txt",
            json.dumps({"codex": codex_command,
                        "dispatches": [entry.get("arguments") for entry in dispatches],
                        "calls": [entry.get("arguments") for entry in calls]},
                       sort_keys=True) + "\n")
        log = store.text(f"failures/{label}-log.txt",
                         _bounded(combined_stdout + "\n" + combined_stderr)
                         or "empty launcher and remote logs\n")
        diagnostic = store.text(
            f"failures/{label}-diagnostic.txt",
            _bounded(f"classification={classification}/{failure_type}/{stage}\n{error}\n"))
        record = failure_record(
            store=store, session_id=request["session_id"], kind=request["kind"],
            arm=request["arm"], classification=classification, failure_type=failure_type,
            stage=stage, product=product, target=target, handle=handle if acquisition else None,
            command=command, log=log, diagnostic=diagnostic,
            manifest_sha256=request["manifest_sha256"], launcher_identity=request["launcher"],
            model_identity={key: request["model"][key]
                            for key in ("identity", "config_sha256")},
            skill_sha256=request["skill_sha256"])
        retained_journal = retain_remote_journal(
            store, f"remote/{label}-failure.json", journal)
        return {"status": classification, "record": record,
                "remote_journal": retained_journal}

    def run(self, request: dict, output: Path, artifact_root: Path, timeout: int = 3600) -> dict:
        request = self.validate_request(request)
        output = output.resolve()
        artifact_root = artifact_root.resolve()
        root = Path(tempfile.mkdtemp(prefix="profile-behavioral-"))
        store = ArtifactStore(artifact_root)
        paths = self.prepare(root, request)
        broker = RemoteBroker(self.remote_command, paths.journal, paths.workspace,
                              set(sum(request["allowed_targets"].values(), [])))
        server = socketserver.UnixStreamServer(str(paths.socket_dir / "broker.sock"), _BrokerHandler)
        server.broker = broker  # type: ignore[attr-defined]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        turns = []
        thread_id = ""
        active_unit = None
        journal_start = (0, 0)
        try:
            base = self.base_command(paths, request)
            self.sandbox_preflight(base)
            for index, planned in enumerate(self.turn_plan(request), 1):
                active_unit = planned["unit"]
                broker.restrict_targets(
                    {active_unit["target"]} if request["kind"] == "acquisition" else set())
                current_journal = json.loads(paths.journal.read_text())
                journal_start = (len(current_journal.get("dispatches", [])),
                                 len(current_journal.get("calls", [])))
                prompt = paths.prompt.read_text() + "\n\nAgent payload:\n" \
                    + json.dumps(planned["payload"], sort_keys=True) + \
                    ("\nEvery `cpl-remote run` must include a new explicit broker-only "
                     "`--dispatch-key <opaque-id>` before `--file`; reuse that key only to "
                     "observe the exact same argv and file bytes.\n"
                     "Transfers are unavailable; have the retained file-backed job emit only "
                     "compact evidence on stdout, then use its handle for observe/result/logs.\n"
                     "Return exactly one JSON object as the final answer.\n")
                if request["kind"] == "acquisition":
                    prompt += (
                        "The compact evidence JSON must have provenance containing exactly "
                        "the assigned `product` and `target`; do not put a durable handle in "
                        "that provenance because the broker assigns it after dispatch. "
                        "The retained terminal remote output must include "
                        "`REMOTE_CONTENT_SHA256=<hex>` containing the SHA-256 of the exact "
                        "evidence file bytes named in your final answer.\n")
                if planned["resume"]:
                    codex = ["/runtime/node/bin/codex", "exec", "resume", "--json",
                             "--ignore-user-config", "--dangerously-bypass-approvals-and-sandbox",
                             "-m", request["model"]["name"], "-c",
                             f'model_reasoning_effort="{request["model"]["reasoning_effort"]}"',
                             thread_id, "-"]
                else:
                    codex = ["/runtime/node/bin/codex", "exec", "--json", "--ignore-user-config",
                             "--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox",
                             "-m", request["model"]["name"], "-c",
                             f'model_reasoning_effort="{request["model"]["reasoning_effort"]}"',
                             "-C", "/workspace", "-"]
                run = subprocess.run([*base, *codex], input=prompt, text=True,
                                     capture_output=True, check=False, timeout=timeout)
                command_ref = store.text(f"commands/{request['session_id']}-{index}.txt",
                                         "codex exec" + (" resume" if planned["resume"] else "") + "\n")
                log_ref = store.text(f"logs/{request['session_id']}-{index}.txt",
                                     _bounded(run.stdout + "\n" + run.stderr) or "empty log\n")
                if run.returncode:
                    result = self._failure_result(
                        request, paths, store, active_unit,
                        f"Codex process exited {run.returncode}",
                        codex_command="codex exec" + (" resume" if planned["resume"] else ""),
                        codex_stdout=run.stdout, codex_stderr=run.stderr,
                        codex_returncode=run.returncode, journal_start=journal_start)
                    _atomic_json(output, result)
                    return result
                journal = json.loads(paths.journal.read_text())
                counted = counted_dispatch_failure(journal, journal_start,
                                                   active_unit.get("target", ""))
                if counted is not None:
                    result = self._failure_result(
                        request, paths, store, active_unit,
                        "counted remote failure cannot be replaced within a launcher turn",
                        codex_command="codex exec" + (" resume" if planned["resume"] else ""),
                        codex_stdout=run.stdout, codex_stderr=run.stderr,
                        journal_start=journal_start, failed_handle=counted.get("handle"))
                    _atomic_json(output, result)
                    return result
                if not thread_id:
                    thread_id = self._session_id(run.stdout)
                turns.append({"payload": planned["payload"], "answer": self._answer(run.stdout),
                              "command": command_ref, "log": log_ref})
            draft, reasoning_items = self._success_draft(
                request, turns, paths, store, thread_id)
            result = {"status": "review_pending", "session_id": thread_id,
                      "draft_record": draft,
                      "remote_journal": retain_remote_journal(
                          store, f"remote/{request['session_id']}.json",
                          json.loads(paths.journal.read_text())),
                      "review": review_queue(reasoning_items, request["reviewer"])}
            _atomic_json(output, result)
            return result
        except (OSError, KeyError, TypeError, ValueError, subprocess.TimeoutExpired,
                json.JSONDecodeError, LaunchError) as error:
            result = self._failure_result(
                request, paths, store, active_unit, error, journal_start=journal_start)
            _atomic_json(output, result)
            return result
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
            shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    client = sub.add_parser("broker-client")
    client.add_argument("socket", type=Path)
    client.add_argument("arguments", nargs=argparse.REMAINDER)
    finalize = sub.add_parser("finalize")
    finalize.add_argument("--launch-output", type=Path, required=True)
    finalize.add_argument("--decisions", type=Path, required=True)
    finalize.add_argument("--artifact-root", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    example = sub.add_parser("request-example")
    example.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "broker-client":
        return broker_client(args.socket, args.arguments)
    if args.command == "request-example":
        print(write_request_example(args.output_dir))
        return 0
    if args.command == "finalize":
        launch = json.loads(args.launch_output.read_text())
        decisions = json.loads(args.decisions.read_text())
        if launch.get("status") != "review_pending" or not isinstance(decisions, dict):
            raise LaunchError("finalization requires review-pending output and decision object")
        reviewer = launch.get("review", {}).get("reviewer")
        record = finalize_reviews(launch.get("draft_record"), decisions, reviewer,
                                  ArtifactStore(args.artifact_root))
        _atomic_json(args.output, record)
        return 0
    request_path = os.environ.get("PROFILE_GATE_REQUEST")
    output_path = os.environ.get("PROFILE_GATE_OUTPUT")
    if not request_path or not output_path:
        parser.error("PROFILE_GATE_REQUEST and PROFILE_GATE_OUTPUT are required")
    request = json.loads(Path(request_path).read_text())
    remote = json.loads(os.environ.get("PROFILE_GATE_REMOTE", '["cpl-remote"]'))
    launcher = BehavioralLauncher(remote)
    result = launcher.run(request, Path(output_path), Path(output_path).parent / "artifacts")
    return 0 if result["status"] == "review_pending" else 1


if __name__ == "__main__":
    raise SystemExit(main())
