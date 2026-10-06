#!/usr/bin/env python3
"""Reusable evidence contract for audited agent experiments.

This module deliberately has no Git, agent-runtime, controller-execution, or
campaign lifecycle dependencies.  Runners create evidence; this module checks
that the evidence is compact, attributable, and policy compliant.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Sequence

MAX_EXCERPT = 512
MAX_REPORT_FIELD = 8192
CONTROLLER_POLICY_SCHEMA = "profiling-skill/controller-policy/v1"
REPORT_FIELDS = (
    "hypothesis", "expected_result", "change", "evidence", "observed_result",
    "decision", "postmortem", "next_experiment",
)
SOURCE_FIELDS = ("query", "locator", "content_sha256", "summary", "influence")
HASH_FIELDS = (
    "candidate_sha256", "manifest_sha256", "controller_receipt_sha256",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_SECRET_KEY = (
    r"(?:[a-z0-9]+[-_])*(?:api[-_]?key|auth[-_]?token|access[-_]?token|"
    r"token|password|secret)"
)
_SECRET_KEY_PATTERN = re.compile(_SECRET_KEY, re.IGNORECASE)
_SECRET_OPTION = re.compile(
    rf"(?i)(?P<prefix>(?<!\S)--?{_SECRET_KEY}(?![a-z0-9_-])(?:=|\s+))"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^\s]+)"
)
_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(?P<prefix>(?<![a-z0-9_-]){_SECRET_KEY}(?![a-z0-9_-])"
    r"[\"']?\s*[=:]\s*)(?P<value>\"[^\"]*\"|'[^']*'|[^\s,;}]+)"
)


class AuditError(RuntimeError):
    """Retained experiment evidence violates the audit contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(document: object) -> str:
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def _valid_hash(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def validate_sources(sources: object, no_sources_reason: object, context: str = "report") -> list:
    if not isinstance(sources, list):
        raise AuditError(f"{context} sources must be an array")
    if not sources and (
        not isinstance(no_sources_reason, str) or not no_sources_reason.strip()
    ):
        raise AuditError(f"empty {context} sources require a no_sources_reason")
    for index, source in enumerate(sources):
        if (
            not isinstance(source, dict)
            or set(source) != set(SOURCE_FIELDS)
            or any(
                not isinstance(source.get(field), str) or not source[field].strip()
                for field in SOURCE_FIELDS
            )
            or not _valid_hash(source.get("content_sha256"))
        ):
            raise AuditError(f"{context} source {index} is invalid")
    return sources


def validate_report(document: object) -> dict:
    """Validate the agent's compact experiment report."""
    if not isinstance(document, dict):
        raise AuditError("final report must be a JSON object")
    for field in REPORT_FIELDS:
        value = document.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AuditError(f"final report is missing {field}")
        if len(value) > MAX_REPORT_FIELD:
            raise AuditError(f"final report field {field} exceeds the compact evidence limit")
    if document["decision"] not in {"retain", "revert"}:
        raise AuditError("decision must be retain or revert")
    for field in HASH_FIELDS:
        if not _valid_hash(document.get(field)):
            raise AuditError(f"final report has invalid {field}")
    if not isinstance(document.get("controller_handle"), str) or not document["controller_handle"]:
        raise AuditError("final report is missing controller_handle")
    validate_sources(document.get("sources"), document.get("no_sources_reason"))
    return document


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _timing_summary(samples: object, sample_count: object, median: object,
                    threshold: object, variability: object, context: str) -> float:
    if (
        not isinstance(samples, list)
        or type(sample_count) is not int
        or sample_count < 3
        or len(samples) != sample_count
        or not all(_is_number(value) and value > 0 for value in samples)
        or not _is_number(median)
        or median <= 0
        or not math.isclose(statistics.median(samples), median, rel_tol=1e-12)
        or not _is_number(threshold)
        or threshold < 0
        or not _is_number(variability)
        or variability < 0
    ):
        raise AuditError(f"{context} fixed-sample timing proof is invalid")
    computed = (max(samples) - min(samples)) / median
    if not math.isclose(computed, variability, rel_tol=1e-12, abs_tol=1e-15):
        raise AuditError(f"{context} variability proof is invalid")
    return computed


def _validate_performance_evidence(receipt: dict) -> None:
    fields = {"baseline_median_us", "baseline", "calibration",
              "normalized_samples_us", "normalized_median_us",
              "speedup_vs_baseline"}
    present = fields & receipt.keys()
    if not present:
        return
    if present != fields:
        raise AuditError("normalized performance evidence is incomplete")
    baseline = receipt["baseline"]
    if (not isinstance(baseline, dict)
            or set(baseline) != {"schema", "benchmark", "case_medians_us",
                                 "control_median_us", "sha256"}
            or baseline.get("schema") != "profiling-skill/baseline-timing/v1"):
        raise AuditError("baseline timing evidence is invalid")
    bound = {key: value for key, value in baseline.items() if key != "sha256"}
    if baseline.get("sha256") != sha256_json(bound):
        raise AuditError("baseline timing hash does not bind its values")
    rows = baseline["case_medians_us"]
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict)
                   or set(row) != {"case", "median_us"}
                   or type(row["case"]) is not int or row["case"] < 0
                   or not _is_number(row["median_us"]) or row["median_us"] <= 0
                   for row in rows)
            or len({row["case"] for row in rows}) != len(rows)
            or not _is_number(baseline["control_median_us"])
            or baseline["control_median_us"] <= 0):
        raise AuditError("baseline timing values are invalid")
    baseline_median = math.exp(sum(math.log(row["median_us"]) for row in rows) / len(rows))
    if (not _is_number(receipt["baseline_median_us"])
            or not math.isclose(receipt["baseline_median_us"], baseline_median,
                                rel_tol=1e-12)):
        raise AuditError("baseline aggregate timing is invalid")
    calibration = receipt["calibration"]
    if (not isinstance(calibration, dict)
            or set(calibration) != {"before", "after", "local_reference_median_us",
                                    "baseline_reference_median_us", "normalization_factor"}):
        raise AuditError("calibration evidence is invalid")
    medians = []
    for phase in ("before", "after"):
        evidence = calibration[phase]
        samples = evidence.get("samples_us") if isinstance(evidence, dict) else None
        median = evidence.get("median_us") if isinstance(evidence, dict) else None
        if (not isinstance(evidence, dict)
                or set(evidence) != {"samples_us", "median_us", "handle"}
                or not isinstance(samples, list) or len(samples) < 3
                or not all(_is_number(value) and value > 0 for value in samples)
                or not _is_number(median) or median <= 0
                or not math.isclose(median, statistics.median(samples), rel_tol=1e-12)
                or not isinstance(evidence["handle"], str) or not evidence["handle"]):
            raise AuditError("calibration control timing is invalid")
        medians.append(median)
    reference = math.sqrt(medians[0] * medians[1])
    factor = baseline["control_median_us"] / reference
    if (not all(_is_number(calibration[field]) for field in (
                "local_reference_median_us", "baseline_reference_median_us",
                "normalization_factor"))
            or not math.isclose(calibration["local_reference_median_us"], reference, rel_tol=1e-12)
            or not math.isclose(calibration["baseline_reference_median_us"],
                                baseline["control_median_us"], rel_tol=1e-12)
            or not math.isclose(calibration["normalization_factor"], factor, rel_tol=1e-12)):
        raise AuditError("calibration normalization inputs are invalid")
    normalized = receipt["normalized_samples_us"]
    expected = [value * factor for value in receipt["samples_us"]]
    if (not isinstance(normalized, list) or len(normalized) != len(expected)
            or any(not _is_number(value) or not math.isclose(value, wanted, rel_tol=1e-12)
                   for value, wanted in zip(normalized, expected))
            or not _is_number(receipt["normalized_median_us"])
            or not _is_number(receipt["speedup_vs_baseline"])
            or not math.isclose(receipt["normalized_median_us"], statistics.median(expected),
                                rel_tol=1e-12)
            or not math.isclose(receipt["speedup_vs_baseline"],
                                baseline_median / statistics.median(expected), rel_tol=1e-12)):
        raise AuditError("normalized timing evidence is invalid")


def _validate_handles(receipt: dict, policy: dict) -> None:
    submitted = policy["submitted_handles"]
    observed = policy["observed_handles"]
    valid_list = lambda values: (
        isinstance(values, list)
        and all(isinstance(value, str) and value for value in values)
        and len(set(values)) == len(values)
    )
    if (not valid_list(submitted) or not valid_list(observed)
            or receipt["handle"] not in submitted or receipt["handle"] not in observed
            or any(handle not in submitted for handle in observed)):
        raise AuditError("controller handle submission/observation proof is invalid")
    history = policy.get("operation_history")
    if history is None:
        return
    if (not isinstance(history, list) or not history
            or type(policy.get("measurement_generation")) is not int
            or policy["measurement_generation"] < 0):
        raise AuditError("controller operation history is invalid")
    expected_submitted = []
    expected_observed = []
    retry_count = 0
    for record in history:
        if (not isinstance(record, dict)
                or set(record) != {"request_sha256", "mode", "status", "terminal",
                                   "handle", "action", "attempt_id"}
                or not _valid_hash(record["request_sha256"])
                or record["mode"] not in {"submit", "retry_submit", "observe"}
                or not isinstance(record["status"], str)
                or type(record["terminal"]) is not bool
                or not isinstance(record["action"], str)
                or (record["attempt_id"] is not None
                    and not isinstance(record["attempt_id"], str))
                or (record["handle"] is not None
                    and not isinstance(record["handle"], str))):
            raise AuditError("controller operation history record is invalid")
        retained = record["handle"]
        if record["mode"] != "submit":
            retry_count += 1
        if (record["mode"] in {"submit", "retry_submit"} and retained
                and retained not in expected_submitted):
            expected_submitted.append(retained)
        if record["terminal"] and retained and retained not in expected_observed:
            expected_observed.append(retained)
    if (submitted != expected_submitted or observed != expected_observed
            or policy.get("infra_retries") != retry_count):
        raise AuditError("controller operation history does not bind handle or retry proof")


def validate_controller_receipt(receipt: object, candidate_hash: str,
                                manifest_hash: str | None = None) -> dict:
    """Validate a compact policy-v1 controller receipt."""
    if not isinstance(receipt, dict):
        raise AuditError("controller receipt must be a JSON object")
    if receipt.get("candidate_sha256") != candidate_hash:
        raise AuditError("controller receipt belongs to a different candidate")
    if manifest_hash is not None and receipt.get("manifest_sha256") != manifest_hash:
        raise AuditError("controller receipt belongs to a different manifest")
    if not isinstance(receipt.get("handle"), str) or not receipt["handle"]:
        raise AuditError("controller receipt lacks a durable handle")
    if receipt.get("status") not in {"ok", "candidate_error", "measurement_pending"}:
        raise AuditError("controller receipt has an invalid terminal status")
    policy = receipt.get("policy")
    if not isinstance(policy, dict) or policy.get("schema") != CONTROLLER_POLICY_SCHEMA:
        raise AuditError("controller receipt lacks policy proof")
    required = {
        "selected_device", "admission_controls", "submission_candidate_sha256",
        "submitted_handles", "observed_handles", "infra_retries", "retry_budget",
        "quarantined_devices", "quarantine_controls", "confirmation_count",
        "post_control", "variability_threshold",
    }
    if not required.issubset(policy):
        raise AuditError("controller policy proof is incomplete")
    if policy["submission_candidate_sha256"] != candidate_hash:
        raise AuditError("infrastructure retry changed the candidate")

    selected = policy["selected_device"]
    if not isinstance(selected, str) or not selected or receipt.get("device") != selected:
        raise AuditError("controller selected device does not match the result device")
    admitted = policy["admission_controls"]
    if not isinstance(admitted, list) or not any(
        isinstance(item, dict)
        and item.get("device") == selected
        and item.get("status") == "pass"
        and item.get("healthy") is True
        and item.get("idle") is True
        and item.get("warmed") is True
        for item in admitted
    ):
        raise AuditError("controller did not admit the selected device with a known-good control")

    _validate_handles(receipt, policy)
    retries, budget = policy["infra_retries"], policy["retry_budget"]
    if (type(retries) is not int or retries < 0 or type(budget) is not int
            or budget < 0 or retries > budget
            or (policy.get("operation_history") is None
                and retries != len(policy["submitted_handles"]) - 1)):
        raise AuditError("controller retry count is invalid")

    quarantined, controls = policy["quarantined_devices"], policy["quarantine_controls"]
    valid_quarantine = (
        isinstance(quarantined, list)
        and all(isinstance(device, str) and device for device in quarantined)
    )
    if (
        not valid_quarantine
        or len(set(quarantined)) != len(quarantined)
        or selected in quarantined
        or not isinstance(controls, dict)
        or any(controls.get(device) != ["fail", "fail"] for device in quarantined)
    ):
        raise AuditError("device quarantine lacks two known-good control failures")

    count = policy["confirmation_count"]
    if type(count) is not int or count not in (0, 1):
        raise AuditError("controller used more than one timing confirmation")
    if policy["post_control"] not in {"stable", "drift", "not_run"}:
        raise AuditError("controller post-control result is invalid")
    if policy["post_control"] == "drift" and receipt.get("status") != "measurement_pending":
        raise AuditError("post-run control drift must be measurement_pending")
    if not _is_number(policy["variability_threshold"]) or policy["variability_threshold"] < 0:
        raise AuditError("controller variability threshold is invalid")

    samples, confirmation = receipt.get("samples_us"), policy.get("confirmation")
    status, post_control = receipt["status"], policy["post_control"]
    timing_fields_present = (
        samples is not None
        or "median_us" in receipt
        or "sample_count" in policy
        or "variability_ratio" in policy
    )
    if status == "candidate_error":
        if timing_fields_present or ({"baseline_median_us", "baseline", "calibration",
                                      "normalized_samples_us", "normalized_median_us",
                                      "speedup_vs_baseline"} & receipt.keys()):
            raise AuditError("candidate_error must not contain performance timing")
        if post_control != "not_run":
            raise AuditError("candidate_error requires post_control=not_run")
    else:
        if samples is None:
            raise AuditError(f"{status} requires fixed-sample timing")
        if status == "ok" and post_control != "stable":
            raise AuditError("status=ok requires a stable post-control")
        if post_control == "not_run":
            raise AuditError(f"{status} cannot use post_control=not_run")
    if samples is None:
        if count or confirmation is not None:
            raise AuditError("timing confirmation exists without primary samples")
        return receipt
    variability = _timing_summary(
        samples, policy.get("sample_count"), receipt.get("median_us"),
        policy["variability_threshold"], policy.get("variability_ratio"), "controller",
    )
    noisy = variability > policy["variability_threshold"]
    accepted = policy.get("accepted_timing")
    primary = policy.get("primary")
    if accepted is not None:
        if accepted not in {"primary", "confirmation"} or not isinstance(primary, dict):
            raise AuditError("accepted timing selection is invalid")
        proofs = {"primary": primary, "confirmation": confirmation}
        for name, proof in proofs.items():
            if proof is None:
                continue
            if (not isinstance(proof, dict)
                    or proof.get("candidate_sha256") != candidate_hash
                    or not isinstance(proof.get("kernel_name"), str)
                    or not proof["kernel_name"]
                    or not isinstance(proof.get("handle"), str) or not proof["handle"]
                    or not isinstance(proof.get("case_results"), list)
                    or not isinstance(proof.get("compact_artifacts"), list)):
                raise AuditError(f"{name} timing identity is invalid")
            _timing_summary(
                proof.get("samples_us"), proof.get("sample_count"), proof.get("median_us"),
                policy["variability_threshold"], proof.get("variability_ratio"), name,
            )
        selected = proofs.get(accepted)
        if (selected is None or receipt.get("handle") != selected["handle"]
                or receipt.get("kernel_name") != selected["kernel_name"]
                or receipt.get("samples_us") != selected["samples_us"]
                or receipt.get("median_us") != selected["median_us"]
                or receipt.get("case_results") != selected["case_results"]
                or receipt.get("compact_artifacts") != selected["compact_artifacts"]):
            raise AuditError("published timing does not match the accepted capture")
        if accepted == "confirmation":
            if (count != 1 or confirmation is None
                    or primary["kernel_name"] != confirmation["kernel_name"]
                    or [row.get("case") for row in primary["case_results"]]
                    != [row.get("case") for row in confirmation["case_results"]]
                    or primary["variability_ratio"] <= policy["variability_threshold"]):
                raise AuditError("confirmation timing identity or trigger is invalid")
        elif count or confirmation is not None:
            raise AuditError("primary timing cannot carry a confirmation")
        if noisy and receipt.get("status") != "measurement_pending":
            raise AuditError("noisy accepted timing must be measurement_pending")
    else:
        if not noisy and (count or confirmation is not None):
            raise AuditError("stable timing must not consume a confirmation")
        if noisy and receipt.get("status") != "measurement_pending" and (
            count != 1 or not isinstance(confirmation, dict)
        ):
            raise AuditError("noisy timing requires one confirmation or measurement_pending")
    if count and accepted is None:
        if (
            not isinstance(confirmation, dict)
            or confirmation.get("candidate_sha256") != candidate_hash
        ):
            raise AuditError("timing confirmation belongs to a different candidate")
        confirmed = _timing_summary(
            confirmation.get("samples_us"), confirmation.get("sample_count"),
            confirmation.get("median_us"), policy["variability_threshold"],
            confirmation.get("variability_ratio"), "confirmation",
        )
        if (
            confirmed > policy["variability_threshold"]
            and receipt.get("status") != "measurement_pending"
        ):
            raise AuditError("noisy timing confirmation must be measurement_pending")
    _validate_performance_evidence(receipt)
    return receipt


def extract_thread_id(output: str, expected: str | None) -> str:
    """Extract and bind a persistent thread ID from JSONL events."""
    found = []
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
            found.append(event["thread_id"])
    if len(set(found)) > 1:
        raise AuditError("structured event stream contains conflicting thread ids")
    thread = found[-1] if found else expected
    if not thread:
        raise AuditError("Codex did not report a persistent thread id")
    if expected and thread != expected:
        raise AuditError(f"persistent session changed from {expected} to {thread}")
    return thread


def _redact_value(match: re.Match) -> str:
    value = match.group("value")
    quote = value[0] if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'" else ""
    return f"{match.group('prefix')}{quote}<redacted>{quote}"


def credential_values(text: str) -> tuple[str, ...]:
    """Return credential values explicitly bound to recognized keys/options."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    matches = sorted(
        (*_SECRET_OPTION.finditer(text), *_SECRET_ASSIGNMENT.finditer(text)),
        key=lambda match: match.start("value"),
    )
    found = []
    for match in matches:
        value = match.group("value")
        if len(value) > 1 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value and value not in found:
            found.append(value)
    return tuple(found)


def _replace_exact_value(text: str, value: str) -> str:
    key_character = lambda character: character.isalnum() or character in "_-"
    left = r"(?<![a-zA-Z0-9_-])" if key_character(value[0]) else ""
    right = r"(?![a-zA-Z0-9_-])" if key_character(value[-1]) else ""
    return re.sub(f"{left}{re.escape(value)}{right}", "<redacted>", text)


def redact_text(text: str, extra_secrets: Iterable[str] = ()) -> str:
    """Redact credential values without replacing equal substrings elsewhere."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if isinstance(extra_secrets, (str, bytes)):
        raise TypeError("extra_secrets must be an iterable of strings")
    for pattern in (_SECRET_OPTION, _SECRET_ASSIGNMENT):
        text = pattern.sub(_redact_value, text)
    secrets = tuple(extra_secrets)
    if any(not isinstance(secret, str) for secret in secrets):
        raise TypeError("extra_secrets must contain only strings")
    for secret in sorted({secret for secret in secrets if secret}, key=len, reverse=True):
        text = _replace_exact_value(text, secret)
    return text


def _redact(command: str, output: str) -> tuple[str, str]:
    return redact_text(command), redact_text(output, credential_values(command))


def sanitize_argv(arguments: Sequence[str]) -> list[str]:
    """Return controller argv safe for durable reproducibility metadata."""
    sanitized = list(arguments)
    redact_next = False
    for index, argument in enumerate(sanitized):
        if redact_next:
            sanitized[index] = "<redacted>"
            redact_next = False
            continue
        option, separator, _ = argument.partition("=")
        name = option.lstrip("-").replace("_", "-").lower()
        is_option = option.startswith("-")
        is_assignment = bool(separator) and not is_option
        if _SECRET_KEY_PATTERN.fullmatch(name) and (is_option or is_assignment):
            if separator:
                sanitized[index] = f"{option}=<redacted>"
            else:
                redact_next = True
    return sanitized


@dataclass(frozen=True)
class ParsedEvents:
    thread_id: str
    report: dict
    commands: tuple[dict, ...]


def _event_parts(output: str) -> tuple[tuple[dict, ...], object]:
    commands: list[dict] = []
    final: object = None
    for line in output.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item") if event.get("type") == "item.completed" else None
        if not isinstance(item, dict):
            continue
        if item.get("type") == "command_execution":
            command = item.get("command")
            exit_code = item.get("exit_code")
            raw_output = item.get("aggregated_output")
            if (
                not isinstance(command, str)
                or not command.strip()
                or type(exit_code) is not int
                or not 0 <= exit_code <= 255
                or not isinstance(raw_output, str)
            ):
                raise AuditError("structured command event is invalid")
            safe_command, safe_output = _redact(command, raw_output)
            commands.append({
                "command": safe_command,
                "exit_code": exit_code,
                "output_sha256": sha256_bytes(raw_output.encode()),
                "output_excerpt": safe_output[:MAX_EXCERPT],
                "output_truncated": len(safe_output) > MAX_EXCERPT,
            })
        elif item.get("type") == "agent_message":
            try:
                final = json.loads(str(item.get("text", "")))
            except json.JSONDecodeError:
                final = None
    return tuple(commands), final


def parse_agent_events(output: str, expected_thread: str | None, *,
                       require_commands: bool = True) -> ParsedEvents:
    thread = extract_thread_id(output, expected_thread)
    commands, final = _event_parts(output)
    if final is None:
        raise AuditError("structured event stream contains no valid final report")
    if require_commands and not commands:
        raise AuditError("structured event stream contains no command evidence")
    return ParsedEvents(thread, validate_report(final), commands)


def parse_command_events(output: str, expected_thread: str | None) -> tuple[str, tuple[dict, ...]]:
    """Extract preparation commands without requiring a premature report."""
    thread = extract_thread_id(output, expected_thread)
    commands, _ = _event_parts(output)
    if not commands:
        raise AuditError("structured preparation contains no command evidence")
    return thread, commands


# Compatibility aliases for the integrated prototype while stacked PRs migrate.
_sha = sha256_bytes
_json_sha = sha256_json
_validate_sources = validate_sources
_sanitized_argv = sanitize_argv
