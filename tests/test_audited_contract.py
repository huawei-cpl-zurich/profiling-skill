from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import audited_contract as contract  # noqa: E402


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def report() -> dict:
    return {
        "hypothesis": "tiling should reduce launch overhead",
        "expected_result": "the fixed-sample median decreases",
        "change": "changed the tile width",
        "evidence": "the retained controller receipt and local tests passed",
        "observed_result": "median decreased",
        "decision": "retain",
        "postmortem": "the result matched the hypothesis",
        "next_experiment": "change one other parameter",
        "candidate_sha256": digest("candidate"),
        "manifest_sha256": digest("manifest"),
        "controller_handle": "job:1",
        "controller_receipt_sha256": digest("receipt"),
        "sources": [],
        "no_sources_reason": "The self-contained task needed no external source.",
    }


def receipt(*, status: str = "ok") -> dict:
    candidate = digest("candidate")
    manifest = digest("manifest")
    samples = [10.0, 10.1, 10.2]
    return {
        "status": status,
        "handle": "job:1",
        "candidate_sha256": candidate,
        "manifest_sha256": manifest,
        "device": "selected-device",
        "samples_us": samples,
        "median_us": 10.1,
        "policy": {
            "schema": contract.CONTROLLER_POLICY_SCHEMA,
            "selected_device": "selected-device",
            "admission_controls": [{
                "device": "selected-device", "status": "pass", "healthy": True,
                "idle": True, "warmed": True,
            }],
            "submission_candidate_sha256": candidate,
            "submitted_handles": ["job:1"],
            "observed_handles": ["job:1"],
            "infra_retries": 0,
            "retry_budget": 1,
            "quarantined_devices": [],
            "quarantine_controls": {},
            "sample_count": 3,
            "variability_threshold": 0.05,
            "variability_ratio": (10.2 - 10.0) / 10.1,
            "confirmation_count": 0,
            "post_control": "stable",
        },
    }


def event_stream(document: dict, *, thread: str = "thread-1") -> str:
    events = [
        {"type": "thread.started", "thread_id": thread},
        {"type": "item.completed", "item": {
            "type": "command_execution",
            "command": "python check.py --api-key=TOPSECRET",
            "exit_code": 0,
            "aggregated_output": "TOKEN=TOPSECRET\n" + "x" * 600,
        }},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(document),
        }},
    ]
    return "\n".join(json.dumps(event) for event in events)


def test_report_accepts_sources_and_requires_complete_compact_fields():
    document = report()
    document["sources"] = [{
        "query": "Triton tiling",
        "locator": "ref://triton/tiling",
        "content_sha256": digest("source"),
        "summary": "Larger contiguous tiles reduce launch overhead.",
        "influence": "Motivated the tile-width experiment.",
    }]
    document["no_sources_reason"] = ""
    assert contract.validate_report(document) is document

    for mutation, match in (
        (lambda value: value.pop("postmortem"), "postmortem"),
        (lambda value: value.update(decision="maybe"), "decision"),
        (lambda value: value.update(sources=[], no_sources_reason=""), "source"),
        (lambda value: value.update(hypothesis="x" * 8193), "compact"),
    ):
        invalid = copy.deepcopy(document)
        mutation(invalid)
        with pytest.raises(contract.AuditError, match=match):
            contract.validate_report(invalid)


def test_source_rejects_extra_fields_blank_values_and_non_sha_hashes():
    for source in (
        {"query": "q"},
        {"query": "q", "locator": "ref://x", "content_sha256": "bad",
         "summary": "s", "influence": "i"},
        {"query": "q", "locator": "ref://x", "content_sha256": digest("x"),
         "summary": "s", "influence": "i", "raw_document": "leak"},
    ):
        document = report()
        document["sources"] = [source]
        with pytest.raises(contract.AuditError, match="source 0"):
            contract.validate_report(document)


def test_structured_events_bind_session_parse_final_report_and_redact_secrets():
    parsed = contract.parse_agent_events("[]\n" + event_stream(report()), None)
    assert parsed.thread_id == "thread-1"
    assert parsed.report == report()
    assert len(parsed.commands) == 1
    command = parsed.commands[0]
    assert "TOPSECRET" not in json.dumps(command)
    assert "<redacted>" in command["command"]
    assert command["output_sha256"] == digest("TOKEN=TOPSECRET\n" + "x" * 600)
    assert len(command["output_excerpt"]) == contract.MAX_EXCERPT
    assert command["output_truncated"] is True


def test_command_evidence_redacts_environment_and_json_credentials():
    secrets = (
        "openai-secret", "github-secret", "hf-secret", "flag-secret",
        "json-secret", "json-hf-secret", "spaced secret",
    )
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {
            "type": "command_execution",
            "command": (
                "export SERVICE_SECRET='spaced secret'; "
                "OPENAI_API_KEY=openai-secret GITHUB_TOKEN=github-secret "
                "python check.py --hf-token hf-secret --token flag-secret "
                "--label public-secret"
            ),
            "exit_code": 0,
            "aggregated_output": (
                '{"database_password": "json-secret", '
                '"hf_token": "json-hf-secret", "token": ""}\n'
            ),
        }},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(report()),
        }},
    ]
    parsed = contract.parse_agent_events(
        "\n".join(json.dumps(event) for event in events), None
    )
    retained = json.dumps(parsed.commands)
    assert "<redacted>" in retained
    assert all(secret not in retained for secret in secrets)
    assert "public-secret" in retained
    assert parsed.commands[0]["output_truncated"] is False


def test_redaction_changes_only_credential_value_spans():
    events = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": {
            "type": "command_execution",
            "command": (
                "HF_TOKEN=tok tool --access-token ace --label tok "
                "--description value-secret"
            ),
            "exit_code": 0,
            "aggregated_output": (
                '{"hf_token":"tok","access_token":"ace",'
                '"tokenizer":"tok","note":"value-secret"}'
            ),
        }},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(report()),
        }},
    ]
    command = contract.parse_agent_events(
        "\n".join(json.dumps(event) for event in events), None
    ).commands[0]
    assert command["command"] == (
        "HF_TOKEN=<redacted> tool --access-token <redacted> --label tok "
        "--description value-secret"
    )
    assert command["output_excerpt"] == (
        '{"hf_token":"<redacted>","access_token":"<redacted>",'
        '"tokenizer":"<redacted>","note":"value-secret"}'
    )
    assert contract.redact_text("OPENAI_API_KEY=tok status=tok") == (
        "OPENAI_API_KEY=<redacted> status=tok"
    )


def test_command_credentials_are_redacted_when_echoed_bare_in_output():
    command = "OPENAI_API_KEY=openai-secret tool --access-token tok"
    output = "login failed for openai-secret; bearer tok rejected; hf_token intact"

    assert contract.credential_values(command) == ("openai-secret", "tok")
    assert contract.redact_text(output, contract.credential_values(command)) == (
        "login failed for <redacted>; bearer <redacted> rejected; hf_token intact"
    )
    assert contract._redact(command, output) == (
        "OPENAI_API_KEY=<redacted> tool --access-token <redacted>",
        "login failed for <redacted>; bearer <redacted> rejected; hf_token intact",
    )


def test_structured_events_reject_session_loss_change_and_incomplete_logs():
    with pytest.raises(contract.AuditError, match="thread id"):
        contract.parse_agent_events(json.dumps({"type": "turn.completed"}), None)
    with pytest.raises(contract.AuditError, match="session changed"):
        contract.parse_agent_events(event_stream(report(), thread="wrong"), "expected")
    only_report = "\n".join((
        json.dumps({"type": "thread.started", "thread_id": "thread-1"}),
        json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": json.dumps(report()),
        }}),
    ))
    with pytest.raises(contract.AuditError, match="command evidence"):
        contract.parse_agent_events(only_report, None)
    thread, commands = contract.parse_command_events(event_stream({"prepared": True}), None)
    assert thread == "thread-1" and len(commands) == 1


@pytest.mark.parametrize("item", [
    {"type": "command_execution", "exit_code": 0, "aggregated_output": "ok"},
    {"type": "command_execution", "command": "", "exit_code": 0,
     "aggregated_output": "ok"},
    {"type": "command_execution", "command": 12, "exit_code": 0,
     "aggregated_output": "ok"},
    {"type": "command_execution", "command": "check", "aggregated_output": "ok"},
    {"type": "command_execution", "command": "check", "exit_code": True,
     "aggregated_output": "ok"},
    {"type": "command_execution", "command": "check", "exit_code": -1,
     "aggregated_output": "ok"},
    {"type": "command_execution", "command": "check", "exit_code": 256,
     "aggregated_output": "ok"},
    {"type": "command_execution", "command": "check", "exit_code": 0,
     "aggregated_output": None},
])
def test_command_events_reject_malformed_records(item):
    output = "\n".join(json.dumps(event) for event in (
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "item.completed", "item": item},
    ))
    with pytest.raises(contract.AuditError, match="command event"):
        contract.parse_command_events(output, None)


def test_sanitize_argv_redacts_separate_and_equals_secret_options():
    argv = [
        "OPENAI_API_KEY=zero", "controller", "--github-token", "one",
        "--hf_token=two", "--label", "public-secret", "--mode", "profile",
    ]
    assert contract.sanitize_argv(argv) == [
        "OPENAI_API_KEY=<redacted>", "controller", "--github-token", "<redacted>",
        "--hf_token=<redacted>", "--label", "public-secret", "--mode", "profile",
    ]


def test_policy_v1_accepts_bound_finite_fixed_sample_evidence():
    document = receipt()
    assert contract.validate_controller_receipt(
        document, digest("candidate"), digest("manifest")
    ) is document


def test_receipt_status_binds_performance_timing_and_post_control():
    pending = receipt(status="measurement_pending")
    pending["policy"]["post_control"] = "drift"
    assert contract.validate_controller_receipt(pending, digest("candidate")) is pending

    candidate_error = receipt(status="candidate_error")
    candidate_error.pop("samples_us")
    candidate_error.pop("median_us")
    candidate_error["policy"].pop("sample_count")
    candidate_error["policy"].pop("variability_ratio")
    candidate_error["policy"]["post_control"] = "not_run"
    assert contract.validate_controller_receipt(
        candidate_error, digest("candidate")
    ) is candidate_error


@pytest.mark.parametrize("status,remove_timing,post_control,match", [
    ("ok", True, "stable", "requires fixed-sample timing"),
    ("ok", False, "not_run", "requires a stable post-control"),
    ("measurement_pending", True, "stable", "requires fixed-sample timing"),
    ("measurement_pending", False, "not_run", "cannot use post_control=not_run"),
    ("candidate_error", False, "not_run", "must not contain performance timing"),
    ("candidate_error", True, "stable", "requires post_control=not_run"),
])
def test_receipt_status_rejects_incoherent_timing(
    status, remove_timing, post_control, match
):
    document = receipt(status=status)
    document["policy"]["post_control"] = post_control
    if remove_timing:
        document.pop("samples_us")
        document.pop("median_us")
        document["policy"].pop("sample_count")
        document["policy"].pop("variability_ratio")
    with pytest.raises(contract.AuditError, match=match):
        contract.validate_controller_receipt(document, digest("candidate"))


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d.update(status="infrastructure_error"), "terminal status"),
    (lambda d: d.update(candidate_sha256="other"), "different candidate"),
    (lambda d: d["policy"].update(submission_candidate_sha256="other"), "retry changed"),
    (lambda d: d.update(device="other"), "selected device"),
    (lambda d: d["policy"].update(observed_handles=[]), "observation"),
    (lambda d: d["policy"].update(submitted_handles=["job:1", "job:1"]), "observation"),
    (lambda d: d["policy"].update(infra_retries=2), "retry count"),
    (lambda d: d["policy"].update(retry_budget=-1), "retry count"),
    (lambda d: d["policy"]["admission_controls"][0].update(idle=False), "known-good"),
])
def test_policy_v1_rejects_hash_device_handle_and_retry_mismatches(mutation, match):
    document = receipt()
    mutation(document)
    with pytest.raises(contract.AuditError, match=match):
        contract.validate_controller_receipt(document, digest("candidate"), digest("manifest"))


@pytest.mark.parametrize("samples", [
    [1.0, 0.0, 1.0], [1.0, "1", 1.0], [1.0, math.nan, 1.0],
    [1.0, math.inf, 1.0],
])
def test_policy_v1_rejects_nonpositive_or_nonfinite_samples(samples):
    document = receipt()
    document["samples_us"] = samples
    with pytest.raises(contract.AuditError, match="timing proof"):
        contract.validate_controller_receipt(document, digest("candidate"))


def test_policy_v1_requires_two_control_failures_to_quarantine_device():
    document = receipt()
    document["policy"].update(
        quarantined_devices=["suspect"], quarantine_controls={"suspect": ["fail"]}
    )
    with pytest.raises(contract.AuditError, match="two known-good"):
        contract.validate_controller_receipt(document, digest("candidate"))
    document["policy"]["quarantined_devices"] = [{}]
    with pytest.raises(contract.AuditError, match="two known-good"):
        contract.validate_controller_receipt(document, digest("candidate"))


def test_noisy_timing_gets_at_most_one_bound_confirmation():
    document = receipt()
    document["samples_us"] = [1.0, 2.0, 3.0]
    document["median_us"] = 2.0
    document["policy"].update(variability_ratio=1.0)
    with pytest.raises(contract.AuditError, match="requires one confirmation"):
        contract.validate_controller_receipt(document, digest("candidate"))

    document["policy"].update(confirmation_count=1, confirmation={
        "candidate_sha256": digest("candidate"),
        "samples_us": [2.0, 2.02, 2.04], "sample_count": 3,
        "median_us": 2.02, "variability_ratio": (2.04 - 2.0) / 2.02,
    })
    assert contract.validate_controller_receipt(document, digest("candidate")) is document
    document["policy"]["confirmation_count"] = 2
    with pytest.raises(contract.AuditError, match="more than one"):
        contract.validate_controller_receipt(document, digest("candidate"))


def test_noisy_confirmation_and_post_control_drift_become_measurement_pending():
    noisy = receipt()
    noisy["samples_us"] = [1.0, 2.0, 3.0]
    noisy["median_us"] = 2.0
    noisy["policy"].update(
        variability_ratio=1.0,
        confirmation_count=1,
        confirmation={
            "candidate_sha256": digest("candidate"),
            "samples_us": [1.0, 2.0, 3.0], "sample_count": 3,
            "median_us": 2.0, "variability_ratio": 1.0,
        },
    )
    with pytest.raises(contract.AuditError, match="measurement_pending"):
        contract.validate_controller_receipt(noisy, digest("candidate"))
    noisy["status"] = "measurement_pending"
    assert contract.validate_controller_receipt(noisy, digest("candidate")) is noisy

    drift = receipt()
    drift["policy"]["post_control"] = "drift"
    with pytest.raises(contract.AuditError, match="measurement_pending"):
        contract.validate_controller_receipt(drift, digest("candidate"))
