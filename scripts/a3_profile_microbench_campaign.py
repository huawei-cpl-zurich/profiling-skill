#!/usr/bin/env python3
"""Resumable, shardable A3 microbenchmark profile campaign coordinator."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "a3-profile-hypotheses.json"
ACQUIRE = ROOT / "scripts" / "acquire_profile.py"
INFRA_FAILURES = {"transport_error", "host_environment", "device_unavailable"}
RETRYABLE_ACQUISITION_FAILURES = INFRA_FAILURES | {"evidence_failure", "profiler_failure"}
TARGET_RUNTIMES = {"bz-a3-1": "py311-torch", "bz-a3-2": "py311-torch", "gz-a3": None}
QUALIFICATION_PROBES = (
    "loop-bound/short-local/before",
    "head-rank/heads-16/after",
)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _acquire_module():
    module_spec = importlib.util.spec_from_file_location("campaign_acquire_profile", ACQUIRE)
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec.loader
    module_spec.loader.exec_module(module)
    return module


def enumerate_cells(manifest: dict) -> list[dict]:
    cells = []
    for kind, key in (("hypothesis", "hypotheses"), ("timeline", "timeline_probes")):
        for experiment in manifest[key]:
            for sweep in experiment["sweeps"]:
                for variant in ("before", "after"):
                    cells.append({
                        "id": f"{experiment['id']}/{sweep['id']}/{variant}",
                        "experiment": experiment["id"], "sweep": sweep["id"],
                        "sweep_parameters": sweep,
                        "variant": variant, "kind": kind,
                        "kernel_names": experiment["kernel_names"][variant],
                        "expected_evidence": experiment["expected_evidence"],
                        "falsifier": experiment["falsifier"],
                    })
    return cells


def select_shard(cells: list[dict], index: int, count: int) -> list[dict]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("shard index must be in [0, shard count)")
    # Before/after cells are adjacent by construction. Keep each pair on one
    # shard so attribution stays within one campaign supervisor/server.
    return [cell for position, cell in enumerate(cells) if (position // 2) % count == index]


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    temporary.replace(path)


def _receipt_handle(path: Path) -> str | None:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError):
        return None
    handle = value.get("handle")
    return handle if isinstance(handle, str) else None


def _retire(paths: list[Path]) -> list[str]:
    retired = []
    for path in paths:
        if not path.exists():
            continue
        destination = path.with_name(path.name + ".retired")
        counter = 1
        while destination.exists():
            destination = path.with_name(path.name + f".retired-{counter}")
            counter += 1
        path.replace(destination)
        retired.append(str(destination))
    return retired


def _fresh_artifact_path(preferred: Path) -> Path:
    if not list(preferred.parent.glob(preferred.name + ".retired*")):
        return preferred
    counter = 1
    candidate = preferred.with_name(f"{preferred.stem}-retry-{counter}{preferred.suffix}")
    while candidate.exists() or list(candidate.parent.glob(candidate.name + ".retired*")):
        counter += 1
        candidate = preferred.with_name(f"{preferred.stem}-retry-{counter}{preferred.suffix}")
    return candidate


def invoke_acquisition(
    *, phase: str, target: str, dispatch_key: str, bundle: Path, entrypoint: str,
    workload_args: list[str], evidence_path: Path, selector: str | list[str] | None,
    basic_evidence: Path | None, cpl_remote: Path | None, launch_count: int = 1,
) -> dict:
    """Run or resume one acquire_profile pass and return compact evidence."""
    receipt = Path(str(evidence_path) + ".dispatch.json")
    if evidence_path.exists():
        module = _acquire_module()
        identity = module.build_bundle(
            bundle=bundle, entrypoint=entrypoint, arguments=workload_args
        ).identity
        selectors = ([selector] if isinstance(selector, str) else list(selector or []))
        expected = {
            "schema_version": 1, "mode": phase, "product": "a3", "target": target,
            "runtime": module.TARGETS["a3"][target], "dispatch_key": dispatch_key,
            "workload": identity, "kernel_names": selectors,
            "launch_count": launch_count,
        }
        try:
            receipt_value = json.loads(receipt.read_bytes())
            evidence = json.loads(evidence_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            retired = _retire([evidence_path, receipt])
            return {"ok": False, "classification": "evidence_failure", "phase": phase,
                    "message": f"cached acquisition metadata is unreadable: {exc}",
                    "retired_artifacts": retired}
        if (receipt_value.get("request") != expected or
                receipt_value.get("request_sha256") != hashlib.sha256(canonical(expected)).hexdigest()):
            retired = _retire([evidence_path, receipt])
            return {"ok": False, "classification": "evidence_failure", "phase": phase,
                    "message": "cached acquisition does not match current dispatch identity",
                    "retired_artifacts": retired}
        try:
            module.validate_evidence(evidence, phase, "a3", target, dispatch_key, identity)
        except module.AcquisitionError as exc:
            retired = _retire([evidence_path, receipt])
            return {"ok": False, "classification": "evidence_failure", "phase": phase,
                    "message": str(exc), "retired_artifacts": retired}
        return {"ok": True, "handle": _receipt_handle(receipt),
                "evidence": evidence}
    command = [
        sys.executable, str(ACQUIRE), phase, "--product", "a3", "--target", target,
        "--dispatch-key", dispatch_key, "--bundle", str(bundle),
        "--entrypoint", entrypoint, "--launch-count", str(launch_count),
        "--evidence", str(evidence_path), "--dispatch-receipt", str(receipt),
    ]
    for argument in workload_args:
        command.append("--workload-arg=" + argument)
    if cpl_remote:
        command.extend(("--cpl-remote", str(cpl_remote)))
    selectors = [selector] if isinstance(selector, str) else list(selector or [])
    for name in selectors:
        command.extend(("--kernel-name", name))
    if basic_evidence:
        command.extend(("--basic-evidence", str(basic_evidence)))
    handle = _receipt_handle(receipt)
    if receipt.exists() and handle is None:
        retired = _retire([receipt])
        return {"ok": False, "classification": "evidence_failure", "phase": phase,
                "message": "cached dispatch receipt is unreadable",
                "retired_artifacts": retired}
    if handle:
        command.extend(("--resume-handle", handle))
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode:
        match = re.search(
            r"classification=([^ ]+) phase=([^ ]+)(?: handle=([^: ]+:[^ ]+))?",
            result.stderr,
        )
        failure = {
            "ok": False,
            "classification": match.group(1) if match else "controller_failure",
            "phase": match.group(2) if match else phase,
            "handle": _receipt_handle(receipt) or (match.group(3) if match else None) or handle,
            "message": result.stderr[-2000:],
        }
        # Transport loss may leave a live job and must retain its receipt for
        # observation. Terminal host/device/evidence failures are retired so a
        # numbered campaign retry receives a fresh dispatch identity.
        if failure["classification"] in {
            "host_environment", "device_unavailable", "evidence_failure", "profiler_failure"
        } and receipt.exists():
            retired = receipt.with_name(receipt.name + ".retired")
            counter = 1
            while retired.exists():
                retired = receipt.with_name(receipt.name + f".retired-{counter}")
                counter += 1
            receipt.replace(retired)
            failure["retired_receipt"] = str(retired)
        return failure
    return {"ok": True, "handle": _receipt_handle(receipt),
            "evidence": json.loads(evidence_path.read_bytes())}


def _numeric_rows(pipe_results: list[dict]) -> tuple[float, dict[str, list[float]]]:
    selector_times = []
    signals: dict[str, list[float]] = {}
    for result in pipe_results:
        rows = result["evidence"]["capture"]["rows"]
        times = []
        for row in rows:
            for key, raw in row.items():
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                if key in {"aic_time(us)", "aiv_time(us)"}:
                    times.append(value)
                if key.endswith("_ratio"):
                    signals.setdefault(key, []).append(value)
        if not times:
            raise ValueError("PipeUtilization evidence lacks aic/aiv reported time")
        selector_times.append(max(times))
    return sum(selector_times), signals


def _summary(samples: list[float], required: int, threshold: float) -> dict:
    selected = None
    relative_range = None
    if len(samples) >= required:
        for start in range(len(samples) - required + 1):
            window = samples[start:start + required]
            center = statistics.median(window)
            spread = ((max(window) - min(window)) / center) if center else float("inf")
            if required == 1 or spread <= threshold:
                selected = window
                relative_range = spread
                break
    median = statistics.median(selected) if selected else (
        statistics.median(samples) if samples else None
    )
    stable = selected is not None
    return {
        "metric": "PipeUtilization reported core-time proxy",
        "unit": "us", "samples": samples, "median": median,
        "relative_range": relative_range, "stable": stable,
        "selected_window": selected,
        "required_samples": required, "maximum_relative_range": threshold,
    }


def run_cell(
    cell: dict, cell_dir: Path, acquire, *, samples: int, max_attempts: int,
    max_relative_range: float, campaign_identity_sha256: str | None = None,
    attempt_limit: int | None = None,
) -> dict:
    run_contract = {
        "schema_version": 1,
        "campaign_identity_sha256": campaign_identity_sha256,
        "cell": cell,
        "samples": samples,
        "max_attempts": max_attempts,
        "max_relative_range": max_relative_range,
    }
    cell_identity_sha256 = hashlib.sha256(canonical(run_contract)).hexdigest()
    state_path = cell_dir / "result.json"
    prior = {}
    if state_path.exists():
        try:
            prior = json.loads(state_path.read_bytes())
        except (OSError, json.JSONDecodeError):
            _retire([state_path])
            prior = {}
        if prior and (
            prior.get("campaign_identity_sha256") != cell_identity_sha256
            or prior.get("run_contract") != run_contract
        ):
            _retire([state_path])
            prior = {}
        elif prior.get("status") in {
            "success", "candidate_failure", "profiling_failure", "inconclusive",
            "infrastructure_exhausted",
        }:
            return prior
    attempts = list(prior.get("attempts", []))
    values = [float(item["duration_proxy_us"]) for item in attempts
              if item.get("classification") == "valid" and item.get("duration_proxy_us")]
    signal_values: dict[str, list[float]] = {}
    for item in attempts:
        if item.get("classification") != "valid":
            continue
        for name, observations in item.get("signals", {}).items():
            signal_values.setdefault(name, []).extend(observations)
    if not signal_values:
        signal_values = {name: list(item.get("values", []))
                         for name, item in prior.get("signals", {}).items()}
    infra = int(prior.get("discarded_infrastructure_attempts", 0))
    provenance = prior.get("provenance", {})
    handles = list(provenance.get("handles", []))
    sources = list(provenance.get("sources", []))
    devices = list(provenance.get("devices", []))
    placements = list(provenance.get("placements", []))
    timeline_evidence = prior.get("timeline")

    def persist(result):
        result["campaign_identity_sha256"] = cell_identity_sha256
        result["run_contract"] = run_contract
        _atomic_json(state_path, result)
        return result

    def finish_timeline(attempt, measurements, selected_signals):
        nonlocal infra, timeline_evidence
        timeline = acquire("timeline", cell, attempt, cell["kernel_names"])
        handles.extend(x for x in [timeline.get("handle")] if x)
        if timeline["ok"]:
            capture = timeline["evidence"]["capture"]
            timeline_evidence = {
                "status": capture.get("availability", "unavailable"),
                "reason": capture.get(
                    "reason", "attempted acquisition exported no TimelineDetail availability"
                ),
                "sources": capture.get("sources", []),
                "semantics": capture.get("timeline_semantics"),
                "handle": timeline.get("handle"),
            }
            sources.extend(capture.get("sources", []))
            return persist(_result(
                cell, "success", measurements, selected_signals, attempts, handles,
                sources, devices, placements, infra, timeline_evidence,
            ))

        classification = timeline.get("classification")
        record = {k: timeline.get(k) for k in
                  ("classification", "phase", "message", "handle")}
        record["identity_consumed"] = classification != "transport_error"
        attempts.append(record)
        infra += classification in INFRA_FAILURES
        timeline_evidence = {
            "status": "pending",
            "reason": timeline.get("message", "TimelineDetail acquisition interrupted"),
            "handle": timeline.get("handle"),
        }
        if classification in RETRYABLE_ACQUISITION_FAILURES:
            consumed = sum(item.get("identity_consumed", True) for item in attempts)
            if classification == "transport_error" or consumed < max_attempts:
                return persist(_result(
                    cell, "running", measurements, selected_signals, attempts, handles,
                    sources, devices, placements, infra, timeline_evidence,
                ))
            status = ("infrastructure_exhausted" if classification in INFRA_FAILURES
                      else "profiling_failure")
            timeline_evidence["status"] = (
                "infrastructure_failure" if status == "infrastructure_exhausted"
                else "profiling_failure"
            )
            return persist(_result(
                cell, status, measurements, selected_signals, attempts, handles,
                sources, devices, placements, infra, timeline_evidence,
            ))
        status = "candidate_failure" if classification == "workload_failure" \
            else "profiling_failure"
        timeline_evidence["status"] = status
        return persist(_result(
            cell, status, measurements, selected_signals, attempts, handles,
            sources, devices, placements, infra, timeline_evidence,
        ))

    consumed_attempts = sum(item.get("identity_consumed", True) for item in attempts)
    stop = min(max_attempts, attempt_limit if attempt_limit is not None else max_attempts)
    basic = None
    while basic is None and consumed_attempts < stop:
        basic_attempt = consumed_attempts
        candidate = acquire("basic", cell, basic_attempt, None)
        handles.extend(x for x in [candidate.get("handle")] if x)
        if candidate["ok"]:
            basic = candidate
            break
        record = {k: candidate.get(k) for k in ("classification", "phase", "message", "handle")}
        record["identity_consumed"] = candidate.get("classification") != "transport_error"
        attempts.append(record)
        if candidate.get("classification") not in RETRYABLE_ACQUISITION_FAILURES:
            return persist(_failure_result(cell, attempts, handles, infra, candidate))
        infra += candidate.get("classification") in INFRA_FAILURES
        partial = persist(_result(
            cell, "running", _summary(values, samples, max_relative_range),
            signal_values, attempts, handles, sources, devices, placements, infra,
            timeline_evidence,
        ))
        if candidate.get("classification") == "transport_error":
            return partial
        consumed_attempts += 1
    if basic is None:
        acquisition_failed = any(item.get("classification") in {
            "evidence_failure", "profiler_failure"
        } for item in attempts)
        status = ("running" if stop < max_attempts else
                  ("profiling_failure" if acquisition_failed else "infrastructure_exhausted"))
        return persist(_result(
            cell, status, _summary(values, samples, max_relative_range), signal_values,
            attempts, handles, sources, devices, placements, infra, timeline_evidence,
        ))
    observed = basic["evidence"]["capture"]["observed_kernel_names"]
    if not all(selector in observed for selector in cell["kernel_names"]):
        failure = {"classification": "evidence_failure", "phase": "basic",
                   "message": f"expected {cell['kernel_names']!r}, observed {observed!r}"}
        return persist(_failure_result(cell, attempts, handles, infra, failure))

    existing_measurements = _summary(values, samples, max_relative_range)
    if (cell["kind"] == "timeline" and existing_measurements["stable"]
            and (timeline_evidence or {}).get("status") not in {"available", "unavailable"}):
        return finish_timeline(
            consumed_attempts, existing_measurements,
            _selected_signal_values(attempts, existing_measurements, signal_values),
        )

    for attempt in range(consumed_attempts, stop):
        pipes = []
        for selector in cell["kernel_names"]:
            pipe = acquire("pipe", cell, attempt, selector)
            handles.extend(x for x in [pipe.get("handle")] if x)
            if not pipe["ok"]:
                record = {k: pipe.get(k) for k in
                          ("classification", "phase", "message", "handle")}
                record["identity_consumed"] = pipe.get("classification") != "transport_error"
                attempts.append(record)
                if pipe.get("classification") in RETRYABLE_ACQUISITION_FAILURES:
                    infra += pipe.get("classification") in INFRA_FAILURES
                    pipes = []
                    partial = persist(_result(
                        cell, "running", _summary(values, samples, max_relative_range),
                        signal_values, attempts, handles, sources, devices, placements, infra,
                        timeline_evidence,
                    ))
                    if pipe.get("classification") == "transport_error":
                        return partial
                    break
                result = _failure_result(cell, attempts, handles, infra, pipe)
                return persist(result)
            pipes.append(pipe)
        if not pipes:
            continue
        try:
            duration, current_signals = _numeric_rows(pipes)
        except ValueError as exc:
            failure = {"classification": "evidence_failure", "phase": "pipe",
                       "message": str(exc)}
            result = _failure_result(cell, attempts, handles, infra, failure)
            return persist(result)
        values.append(duration)
        for name, observations in current_signals.items():
            signal_values.setdefault(name, []).extend(observations)
        for item in [basic, *pipes]:
            evidence = item["evidence"]
            devices.append(evidence.get("device"))
            if evidence.get("device"):
                placements.append({"target": evidence.get("target"),
                                   "physical_id": evidence["device"].get("physical_id")})
            sources.extend(evidence["capture"].get("sources", []))
        attempts.append({"classification": "valid", "attempt": attempt,
                         "selectors": cell["kernel_names"], "duration_proxy_us": duration,
                         "signals": current_signals})
        measurements = _summary(values, samples, max_relative_range)
        selected_signals = _selected_signal_values(attempts, measurements, signal_values)
        partial = _result(cell, "success" if measurements["stable"] else "running",
                          measurements, selected_signals, attempts, handles, sources,
                          devices, placements, infra, timeline_evidence)
        persist(partial)
        if measurements["stable"]:
            if cell["kind"] != "timeline":
                return partial
            return finish_timeline(
                sum(item.get("identity_consumed", True) for item in attempts),
                measurements, selected_signals,
            )
    measurements = _summary(values, samples, max_relative_range)
    if stop < max_attempts:
        return persist(_result(
            cell, "running", measurements, signal_values, attempts, handles,
            sources, devices, placements, infra, timeline_evidence,
        ))
    acquisition_failed = any(item.get("classification") in {
        "evidence_failure", "profiler_failure"
    } for item in attempts)
    status = ("inconclusive" if values else
              ("profiling_failure" if acquisition_failed else "infrastructure_exhausted"))
    result = _result(cell, status, measurements, signal_values, attempts, handles,
                     sources, devices, placements, infra, timeline_evidence)
    return persist(result)


def _selected_signal_values(
    attempts: list[dict], measurements: dict, fallback: dict[str, list[float]],
) -> dict[str, list[float]]:
    window = measurements.get("selected_window")
    if not window:
        return fallback
    valid = [item for item in attempts if item.get("classification") == "valid"]
    durations = [float(item["duration_proxy_us"]) for item in valid]
    width = len(window)
    for start in range(len(durations) - width + 1):
        if durations[start:start + width] != window:
            continue
        selected: dict[str, list[float]] = {}
        for item in valid[start:start + width]:
            for name, observations in item.get("signals", {}).items():
                selected.setdefault(name, []).extend(observations)
        return selected or fallback
    return fallback


def run_pair_interleaved(
    cells: list[dict], root: Path, acquire_for, *, samples: int, max_attempts: int,
    max_relative_range: float, identities: dict[str, str],
) -> list[dict]:
    """Advance paired cells one capture at a time in before/after order."""
    if len(cells) != 2 or {cell["variant"] for cell in cells} != {"before", "after"}:
        raise ValueError("interleaved execution requires one before/after pair")
    cells = sorted(cells, key=lambda item: item["variant"] != "before")
    acquisitions = {}
    for cell in cells:
        raw = acquire_for(cell, root / cell["id"].replace("/", "__"))
        cached_basic = {}
        def cache(phase, current, attempt, selector, *, _raw=raw, _cache=cached_basic):
            if phase == "basic" and "value" in _cache:
                return _cache["value"]
            value = _raw(phase, current, attempt, selector)
            if phase == "basic" and value.get("ok"):
                _cache["value"] = value
            return value
        acquisitions[cell["id"]] = cache
    results = {}
    for limit in range(1, max_attempts + 1):
        for cell in cells:
            if results.get(cell["id"], {}).get("status") in {
                "success", "candidate_failure", "profiling_failure", "inconclusive",
                "infrastructure_exhausted",
            }:
                continue
            cell_dir = root / cell["id"].replace("/", "__")
            results[cell["id"]] = run_cell(
                cell, cell_dir, acquisitions[cell["id"]], samples=samples,
                max_attempts=max_attempts, max_relative_range=max_relative_range,
                campaign_identity_sha256=identities[cell["id"]], attempt_limit=limit,
            )
    return [results[cell["id"]] for cell in cells]


def build_qualification_receipt(identity: dict, probes: list[dict]) -> dict:
    reasons = []
    normalized = []
    probe_ids = [probe.get("id") for probe in probes]
    exact_probes = (
        len(probe_ids) == len(QUALIFICATION_PROBES)
        and len(set(probe_ids)) == len(probe_ids)
        and set(probe_ids) == set(QUALIFICATION_PROBES)
    )
    if not exact_probes:
        reasons.append(
            "qualification requires each designated probe exactly once: "
            + ", ".join(QUALIFICATION_PROBES)
        )
    for probe in probes:
        reasons.extend(f"{probe.get('id')}: {reason}"
                       for reason in probe.get("acquisition_reasons", []))
        samples = [float(value) for value in probe.get("samples", [])]
        stable = _summary(samples, 3, 0.05)
        center = statistics.median(samples) if samples else None
        all_range = ((max(samples) - min(samples)) / center
                     if len(samples) == 5 and center else None)
        handles = probe.get("handles", [])
        devices = probe.get("devices", [])
        sources = probe.get("sources", [])
        provenance_complete = (
            len(handles) >= 5 and len(devices) == 5 and len(sources) >= 5
            and all(isinstance(item, dict) and item.get("sha256") for item in sources)
        )
        passed = (len(samples) == 5 and all_range is not None and all_range <= 0.10
                  and stable["stable"] and provenance_complete)
        if not passed:
            reasons.append(
                f"{probe.get('id')}: all-five range must be <=10%, include a stable-3 "
                "window, and retain handle/device/source provenance"
            )
        normalized.append({**probe, "samples": samples, "all_five_relative_range": all_range,
                           "stable_window": stable["selected_window"], "passed": passed})
    return {"schema_version": 1, "kind": "a3-profile-qualification",
            "identity": identity, "passed": exact_probes and not reasons,
            "reasons": reasons, "probes": normalized}


def validate_qualification(path: Path, identity: dict) -> dict:
    try:
        receipt = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"qualification receipt is unreadable: {exc}") from exc
    if receipt.get("identity") != identity:
        raise ValueError("qualification identity does not match campaign identity")
    rebuilt = build_qualification_receipt(identity, receipt.get("probes", []))
    if (receipt.get("schema_version") != 1
            or receipt.get("kind") != "a3-profile-qualification"
            or not receipt.get("passed") or not rebuilt["passed"]):
        raise ValueError("qualification did not pass")
    return receipt


def campaign_identity(target: str, manifest: dict) -> dict:
    return {
        "target": target,
        "runtime": TARGET_RUNTIMES[target],
        "manifest_sha256": hashlib.sha256(canonical(manifest)).hexdigest(),
        "benchmark_sha256": hashlib.sha256(
            (ROOT / "benchmarks/a3_profile_microbench.py").read_bytes()
        ).hexdigest(),
        "coordinator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "acquisition_sha256": hashlib.sha256(ACQUIRE.read_bytes()).hexdigest(),
    }


def run_qualification_probe(cell: dict, acquire) -> dict:
    """Collect exactly five timing captures after one successful discovery."""
    handles, devices, sources, samples, reasons = [], [], [], [], []
    basic = acquire("basic", cell, 0, None)
    handles.extend(x for x in [basic.get("handle")] if x)
    if not basic.get("ok"):
        reasons.append(f"BasicInfo {basic.get('classification', 'failure')}: {basic.get('message', '')}")
    elif not all(name in basic["evidence"]["capture"]["observed_kernel_names"]
                 for name in cell["kernel_names"]):
        reasons.append("BasicInfo did not export every expected exact selector")
    else:
        for attempt in range(5):
            pipes = [acquire("pipe", cell, attempt, selector)
                     for selector in cell["kernel_names"]]
            handles.extend(result.get("handle") for result in pipes if result.get("handle"))
            failed = next((result for result in pipes if not result.get("ok")), None)
            if failed:
                reasons.append(
                    f"capture {attempt}: {failed.get('classification', 'failure')}: "
                    f"{failed.get('message', '')}"
                )
                break
            try:
                duration, _ = _numeric_rows(pipes)
            except ValueError as exc:
                reasons.append(f"capture {attempt}: {exc}")
                break
            samples.append(duration)
            for result in pipes:
                value = result["evidence"]
                if value.get("device"):
                    devices.append(value["device"])
                sources.extend(value["capture"].get("sources", []))
    return {"id": cell["id"], "samples": samples, "handles": handles,
            "devices": devices, "sources": sources, "acquisition_reasons": reasons}


def _remote_qualification_gate(cpl_remote: Path, target: str) -> tuple[list[dict], list[str]]:
    checks, reasons = [], []
    for action in ("capabilities", "preflight"):
        result = subprocess.run(
            [str(cpl_remote), action, target], text=True, capture_output=True, check=False
        )
        checks.append({
            "command": ["cpl-remote", action, target],
            "exit_code": result.returncode,
            "stdout_sha256": hashlib.sha256(result.stdout.encode()).hexdigest(),
            "stderr_sha256": hashlib.sha256(result.stderr.encode()).hexdigest(),
        })
        if result.returncode:
            reasons.append(f"cpl-remote {action} failed with exit {result.returncode}")
    return checks, reasons


def _signals(values: dict[str, list[float]]) -> dict:
    return {name: {"mean": statistics.fmean(items), "samples": len(items),
                   "values": items} for name, items in sorted(values.items())}


def _result(cell, status, measurements, signals, attempts, handles, sources,
            devices, placements, infra, timeline_evidence=None):
    value = {
        **cell, "status": status, "measurements": measurements,
        "signals": _signals(signals), "attempts": attempts,
        "discarded_infrastructure_attempts": infra,
        "provenance": {"handles": handles, "devices": devices,
                       "placements": placements, "sources": sources},
    }
    if cell["kind"] == "timeline":
        value["timeline"] = timeline_evidence or {
            "status": "not_attempted",
            "reason": "TimelineDetail acquisition did not reach an export attempt",
        }
    return value


def _failure_result(cell, attempts, handles, infra, failure):
    status = ("candidate_failure" if failure.get("classification") == "workload_failure"
              else "profiling_failure")
    value = _result(cell, status, _summary([], 1, 0.05), {},
                    attempts, handles, [], [], [], infra)
    value["failure"] = {k: failure.get(k) for k in
                        ("classification", "phase", "message") if failure.get(k)}
    return value


def _direction(before: float, after: float, expected: dict, *, absolute: bool) -> str:
    direction = expected["direction"]
    threshold = expected.get(
        "minimum_absolute_change" if absolute else "minimum_relative_change", 0.0
    )
    change = before - after if absolute else (before - after) / before
    if direction == "decrease":
        return "supports" if change >= threshold else (
            "refutes" if change <= -threshold else "inconclusive"
        )
    if direction == "increase":
        return "supports" if change <= -threshold else (
            "refutes" if change >= threshold else "inconclusive"
        )
    return "supports" if abs(change) <= threshold else "refutes"


def _measurement_status(before: dict | None, after: dict | None) -> str:
    statuses = {item.get("status") for item in (before, after) if item}
    if "candidate_failure" in statuses:
        return "candidate_failure"
    if "infrastructure_exhausted" in statuses:
        return "infrastructure_failure"
    if "profiling_failure" in statuses:
        return "profiling_failure"
    return "unavailable"


def _timeline_signal(spec: dict, name: str, before: dict | None, after: dict | None):
    timelines = [(item or {}).get("timeline", {}) for item in (before, after)]
    if any(item.get("status") != "available" for item in timelines):
        return "not_exported", "TimelineDetail semantic evidence was unavailable"
    semantics = [item.get("semantics") for item in timelines]
    if any(not isinstance(item, dict) or not item.get("selectors_complete") for item in semantics):
        return "not_exported", "TimelineDetail did not cover every declared selector"
    left, right = semantics
    supported = None
    detail = ""
    if name == "timeline_phase_order":
        left_order, right_order = left.get("phase_order", []), right.get("phase_order", [])
        if not left_order or not right_order:
            return "not_exported", "TimelineDetail exported no attributable phase order"
        if spec["id"] == "ordered-phases":
            supported = left_order == right_order and all(
                phase in left_order for phase in ("load", "compute", "store")
            )
        elif spec["id"] == "reversed-phases":
            supported = left_order != right_order
        else:
            supported = len(right_order) > len(left_order)
        detail = f"before={left_order!r}; after={right_order!r}"
    elif name == "timeline_phase_duration":
        before_duration = left.get("phase_duration_us", {}).get("compute")
        after_duration = right.get("phase_duration_us", {}).get("compute")
        if not isinstance(before_duration, (int, float)) or not isinstance(
            after_duration, (int, float)
        ):
            return "not_exported", "TimelineDetail exported no compute-phase duration"
        threshold = spec["expected"].get("minimum_relative_change", 0.0)
        supported = before_duration > 0 and after_duration >= before_duration * (1 + threshold)
        detail = f"compute before={before_duration:g}us; after={after_duration:g}us"
    elif name == "timeline_pipe_overlap":
        before_overlap = left.get("load_compute_overlap_us")
        after_overlap = right.get("load_compute_overlap_us")
        if not isinstance(before_overlap, (int, float)) or not isinstance(
            after_overlap, (int, float)
        ):
            return "not_exported", "TimelineDetail exported no load/compute overlap"
        supported = after_overlap > before_overlap
        detail = f"overlap before={before_overlap:g}us; after={after_overlap:g}us"
    elif name == "timeline_kernel_boundaries":
        before_boundaries = left.get("kernel_boundaries", [])
        after_boundaries = right.get("kernel_boundaries", [])
        supported = len(after_boundaries) > len(before_boundaries)
        detail = f"boundaries before={len(before_boundaries)}; after={len(after_boundaries)}"
    else:
        return "context_only", f"unrecognized timeline signal {name!r} retained for review"
    relationship = "tracks_outcome" if supported else "contradicts_outcome"
    return relationship, detail


def summarize_hypotheses(raw_cells: list[dict], manifest: dict) -> list[dict]:
    """Pair variant cells into build_profile_reverse_report's input contract."""
    by_key = {
        (cell.get("experiment"), cell.get("sweep"), cell.get("variant")): cell
        for cell in raw_cells
    }
    results = []
    for key in ("hypotheses", "timeline_probes"):
        for spec in manifest[key]:
            measurements, assessments = [], {item["name"]: [] for item in spec["signal_predictions"]}
            for sweep in spec["sweeps"]:
                sweep_id = sweep["id"]
                before = by_key.get((spec["id"], sweep_id, "before"))
                after = by_key.get((spec["id"], sweep_id, "after"))
                valid = (
                    before and after and before.get("status") == after.get("status") == "success"
                    and before.get("measurements", {}).get("median")
                    and after.get("measurements", {}).get("median")
                )
                placement_rows = []
                for cell in (before, after):
                    placement_rows.extend(
                        (cell or {}).get("provenance", {}).get("placements", [])
                    )
                placements = {
                    (item.get("target"), item.get("physical_id"))
                    for item in placement_rows if item.get("physical_id") is not None
                }
                # Devices within one server target are equivalent for this
                # experiment. Preserve their IDs as provenance, but do not
                # reject a pair merely because retries used another device on
                # the same server.
                placement_targets = {
                    item.get("target") for item in placement_rows if item.get("target")
                }
                placement_valid = not placement_rows or len(placement_targets) == 1
                sanity = spec.get("sanity", {})
                sanity_changes = []
                for name in sanity.get("signals", []):
                    left = (before or {}).get("signals", {}).get(name, {}).get("mean")
                    right = (after or {}).get("signals", {}).get(name, {}).get("mean")
                    if left is not None and right is not None:
                        sanity_changes.append(abs(float(left) - float(right)))
                minimum_sanity = float(sanity.get(
                    "minimum_absolute_change", sanity.get("minimum", 0.0)
                ))
                sanity_valid = (not sanity or bool(sanity_changes) and
                                max(sanity_changes) >= minimum_sanity)
                if valid and placement_valid and sanity_valid:
                    before_value = float(before["measurements"]["median"])
                    after_value = float(after["measurements"]["median"])
                    measurement = {
                        "status": "valid", "sweep": sweep_id,
                        "before": before_value, "after": after_value,
                        "metric": before["measurements"]["metric"],
                        "unit": before["measurements"]["unit"],
                        "before_cell_id": before["id"], "after_cell_id": after["id"],
                    }
                    measurements.append(measurement)
                    outcome = _direction(before_value, after_value, spec["expected"], absolute=False)
                elif valid and not placement_valid:
                    measurements.append({
                        "status": "infrastructure_failure", "sweep": sweep_id,
                        "reason": "before/after were not measured on one server target",
                        "placements": [list(item) for item in sorted(placements)],
                    })
                    outcome = "unavailable"
                elif valid and not sanity_valid:
                    measurements.append({
                        "status": "profiling_failure", "sweep": sweep_id,
                        "reason": "declared pipe sanity signal was absent or below its minimum",
                    })
                    outcome = "unavailable"
                else:
                    measurements.append({
                        "status": _measurement_status(before, after), "sweep": sweep_id,
                        "before_status": before.get("status") if before else "missing",
                        "after_status": after.get("status") if after else "missing",
                    })
                    outcome = "unavailable"
                for prediction in spec["signal_predictions"]:
                    name = prediction["name"]
                    if prediction.get("availability") == "required":
                        relationship, detail = _timeline_signal(spec, name, before, after)
                        assessments[name].append((sweep_id, relationship, detail, None))
                        continue
                    left = (before or {}).get("signals", {}).get(name, {}).get("mean")
                    right = (after or {}).get("signals", {}).get(name, {}).get("mean")
                    if left is None or right is None:
                        assessments[name].append((sweep_id, "not_exported", left, right))
                        continue
                    signal = _direction(float(left), float(right), prediction, absolute=True)
                    if outcome == "unavailable" or "inconclusive" in {outcome, signal}:
                        relationship = "context_only"
                    elif outcome == signal:
                        relationship = "tracks_outcome"
                    else:
                        relationship = "contradicts_outcome"
                    assessments[name].append((sweep_id, relationship, float(left), float(right)))
            signals = []
            for prediction in spec["signal_predictions"]:
                name, rows = prediction["name"], assessments[prediction["name"]]
                relationships = {row[1] for row in rows}
                relationship = next(iter(relationships)) if len(relationships) == 1 else "context_only"
                if relationship == "not_exported" and key == "timeline_probes":
                    summary = "TimelineDetail was unavailable for every controlled capture"
                else:
                    details = []
                    for sweep_id, relation, left, right in rows:
                        if right is None:
                            details.append(f"{sweep_id}: {relation}; {left or ''}".rstrip("; "))
                        else:
                            details.append(
                                f"{sweep_id}: before={left:g}, after={right:g}; "
                                + ({
                                    "tracks_outcome": "signal and latency track the declared outcome",
                                    "contradicts_outcome": "signal contradicts the observed latency outcome",
                                    "context_only": "signal is contextual or below its declared threshold",
                                    "not_exported": "signal was not exported",
                                }[relation])
                            )
                    summary = "; ".join(details)
                signals.append({"name": name, "relationship": relationship, "summary": summary})
            results.append({
                "hypothesis_id": spec["id"], "title": spec.get("question", spec["id"]),
                "expected": spec["expected"], "falsification": spec["falsifier"],
                "measurements": measurements, "signals": signals,
            })
    return results


def validate_report_hypotheses(document: dict) -> bool:
    """Validate the subset consumed by build_profile_reverse_report.py."""
    if document.get("schema_version") != 1 or not isinstance(document.get("results"), list):
        raise ValueError("report hypotheses require schema_version 1 and results")
    relationships = {"tracks_outcome", "context_only", "contradicts_outcome", "not_exported"}
    for item in document["results"]:
        if not isinstance(item.get("hypothesis_id"), str) or not item["hypothesis_id"]:
            raise ValueError("hypothesis_id is required")
        expected = item.get("expected", {})
        if expected.get("direction") not in {"decrease", "increase", "no_change"}:
            raise ValueError("invalid expected direction")
        threshold = expected.get("minimum_relative_change")
        if not isinstance(threshold, (int, float)) or not 0 <= threshold < 1:
            raise ValueError("invalid expected threshold")
        if not isinstance(item.get("measurements"), list) or not isinstance(item.get("signals"), list):
            raise ValueError("measurements and signals must be arrays")
        for row in item["measurements"]:
            if row.get("status") == "valid" and not (
                float(row["before"]) > 0 and float(row["after"]) > 0
            ):
                raise ValueError("valid paired measurements must be positive")
        for signal in item["signals"]:
            if not signal.get("name") or signal.get("relationship") not in relationships:
                raise ValueError("invalid signal relationship")
    return True


def merge_shards(shards: list[dict], manifest: dict) -> dict:
    if not shards:
        raise ValueError("at least one shard is required")
    digests = {item["manifest_sha256"] for item in shards}
    expected_digest = hashlib.sha256(canonical(manifest)).hexdigest()
    if digests != {expected_digest}:
        raise ValueError("shard manifest identity does not match the merge manifest")
    identities = {canonical(item.get("campaign_identity")) for item in shards}
    if len(identities) != 1:
        raise ValueError("shard campaign identity mismatch; cross-target merge is forbidden")
    merged_identity = shards[0].get("campaign_identity")
    if not isinstance(merged_identity, dict) or merged_identity.get("manifest_sha256") != expected_digest:
        raise ValueError("shard campaign identity does not match the merge manifest")
    target = merged_identity.get("target")
    if target not in TARGET_RUNTIMES or merged_identity != campaign_identity(target, manifest):
        raise ValueError("shard campaign identity does not match current frozen sources")
    shard_descriptors = [item.get("shard") for item in shards]
    if any(not isinstance(item, dict) for item in shard_descriptors):
        raise ValueError("every shard requires count/index identity")
    counts = {item.get("count") for item in shard_descriptors}
    if len(counts) != 1:
        raise ValueError("shard count mismatch")
    count = next(iter(counts))
    indexes = [item.get("index") for item in shard_descriptors]
    if type(count) is not int or count < 1 or any(
        type(index) is not int or not 0 <= index < count for index in indexes
    ):
        raise ValueError("invalid shard count/index identity")
    if len(indexes) != len(set(indexes)):
        raise ValueError("duplicate shard index")
    missing_shards = sorted(set(range(count)) - set(indexes))
    if missing_shards:
        raise ValueError(f"missing shard indexes: {missing_shards}")
    results, ids = [], set()
    for shard in shards:
        for result in shard["raw_cells"]:
            if result["id"] in ids:
                raise ValueError(f"duplicate cell {result['id']}")
            ids.add(result["id"])
            results.append(result)
    expected_ids = {item["id"] for item in enumerate_cells(manifest)}
    if ids != expected_ids:
        raise ValueError(
            "merged cell coverage mismatch: "
            f"missing={sorted(expected_ids - ids)!r} unexpected={sorted(ids - expected_ids)!r}"
        )
    raw_cells = sorted(results, key=lambda item: item["id"])
    value = {"schema_version": 1, "manifest_sha256": expected_digest,
             "campaign_identity": merged_identity,
             "raw_cells": raw_cells,
             "results": summarize_hypotheses(raw_cells, manifest)}
    validate_report_hypotheses(value)
    return value


def _bundle(output: Path, manifest: Path) -> Path:
    bundle = output / "bundle"
    (bundle / "benchmarks").mkdir(parents=True, exist_ok=True)
    (bundle / "experiments").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "benchmarks/a3_profile_microbench.py",
                 bundle / "benchmarks/a3_profile_microbench.py")
    shutil.copy2(manifest, bundle / "experiments/a3-profile-hypotheses.json")
    return bundle


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    run = sub.add_parser("run")
    qualify = sub.add_parser("qualify")
    merge = sub.add_parser("merge")
    for command in (plan, run):
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--shard-count", type=int, default=1)
    plan.add_argument("--output", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    run.add_argument("--target", choices=tuple(TARGET_RUNTIMES), required=True)
    run.add_argument("--samples", type=int, default=3)
    run.add_argument("--max-attempts", type=int, default=7)
    run.add_argument("--max-relative-range", type=float, default=0.05)
    run.add_argument("--cell-limit", type=int)
    run.add_argument("--cpl-remote", type=Path)
    run.add_argument("--qualification", type=Path, required=True)
    qualify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    qualify.add_argument("--output-dir", type=Path, required=True)
    qualify.add_argument("--target", choices=tuple(TARGET_RUNTIMES), required=True)
    qualify.add_argument("--cpl-remote", type=Path)
    merge.add_argument("--input", type=Path, nargs="+", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.command == "merge":
        manifest = json.loads(args.manifest.read_bytes())
        value = merge_shards([json.loads(path.read_bytes()) for path in args.input], manifest)
        _atomic_json(args.output, value)
        return 0
    manifest = json.loads(args.manifest.read_bytes())
    identity_base = campaign_identity(getattr(args, "target", "bz-a3-1"), manifest) \
        if args.command in {"run", "qualify"} else None
    if args.command == "qualify":
        args.output_dir.mkdir(parents=True, exist_ok=True)
        bundle = _bundle(args.output_dir, args.manifest)
        client = _acquire_module().resolve_cpl_remote(args.cpl_remote)
        checks, gate_reasons = _remote_qualification_gate(client, args.target)
        cells_by_id = {cell["id"]: cell for cell in enumerate_cells(manifest)}
        probes = []
        if not gate_reasons:
            for cell_id in QUALIFICATION_PROBES:
                cell = cells_by_id[cell_id]
                cell_dir = args.output_dir / "probes" / cell_id.replace("/", "__")
                successful_basic: dict[str, Path] = {}

                def acquire(phase, current, attempt, selector, *, _cell_dir=cell_dir):
                    if phase == "basic":
                        basic_path = _fresh_artifact_path(
                            _cell_dir / "discovery" / f"basic-{attempt}.json"
                        )
                        evidence_path = basic_path
                        dispatch_attempt = evidence_path.name
                    else:
                        basic_path = successful_basic.get("path")
                        if basic_path is None:
                            return {"ok": False, "classification": "evidence_failure",
                                    "phase": phase,
                                    "message": "successful BasicInfo evidence is unavailable"}
                        evidence_path = _fresh_artifact_path(
                            _cell_dir / f"capture-{attempt}" / f"pipe-{selector}.json"
                        )
                        dispatch_attempt = evidence_path.name
                    result = invoke_acquisition(
                        phase=phase, target=args.target,
                        dispatch_key=hashlib.sha256(
                            f"a3-profile-qualification/v1/{current['id']}/{dispatch_attempt}/{phase}/{selector}".encode()
                        ).hexdigest(), bundle=bundle,
                        entrypoint="benchmarks/a3_profile_microbench.py",
                        workload_args=["--experiment", current["experiment"], "--sweep",
                                       current["sweep"], "--variant", current["variant"],
                                       "--launches", "1"], evidence_path=evidence_path,
                        selector=selector,
                        basic_evidence=basic_path if phase == "pipe" else None,
                        cpl_remote=client,
                        launch_count=(len(current["kernel_names"]) if phase == "basic" else 1),
                    )
                    if phase == "basic" and result.get("ok"):
                        successful_basic["path"] = basic_path
                    return result

                probes.append(run_qualification_probe(cell, acquire))
        receipt = build_qualification_receipt(identity_base, probes)
        receipt["remote_checks"] = checks
        receipt["reasons"] = gate_reasons + receipt["reasons"]
        receipt["passed"] = receipt["passed"] and not gate_reasons
        output = args.output_dir / "qualification.json"
        _atomic_json(output, receipt)
        print(json.dumps({"qualification": str(output), "passed": receipt["passed"]},
                         sort_keys=True))
        return 0 if receipt["passed"] else 2
    cells = select_shard(enumerate_cells(manifest), args.shard_index, args.shard_count)
    if args.command == "plan":
        value = {"schema_version": 1, "shard": {"index": args.shard_index,
                 "count": args.shard_count}, "cells": cells}
        if args.output:
            _atomic_json(args.output, value)
        else:
            print(json.dumps(value, sort_keys=True))
        return 0
    if args.samples < 1 or args.max_attempts < args.samples:
        raise SystemExit("samples must be positive and no greater than max-attempts")
    validate_qualification(args.qualification, identity_base)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundle = _bundle(args.output_dir, args.manifest)
    manifest_sha256 = hashlib.sha256(canonical(manifest)).hexdigest()
    cells = cells[:args.cell_limit] if args.cell_limit else cells
    cells_root = args.output_dir / "cells"
    identities = {cell["id"]: hashlib.sha256(canonical({
            "schema_version": 1, "manifest_sha256": manifest_sha256,
            "cell": cell, "target": args.target, "samples": args.samples,
            "max_attempts": args.max_attempts,
            "max_relative_range": args.max_relative_range,
            "runtime": TARGET_RUNTIMES[args.target],
            "benchmark_sha256": identity_base["benchmark_sha256"],
            "coordinator_sha256": identity_base["coordinator_sha256"],
            "acquisition_sha256": identity_base["acquisition_sha256"],
        })).hexdigest() for cell in cells}

    def acquire_for(cell, cell_dir):
        successful_basic: dict[str, Path] = {}
        def acquire(phase, current, attempt, selector):
            if phase == "basic":
                basic_path = _fresh_artifact_path(
                    cell_dir / "discovery" / f"basic-{attempt}.json"
                )
                evidence_path = basic_path
                dispatch_attempt = evidence_path.name
            else:
                basic_path = successful_basic.get("path")
                if basic_path is None:
                    return {"ok": False, "classification": "evidence_failure",
                            "phase": phase,
                            "message": "successful BasicInfo evidence is unavailable"}
                suffix = ("-all" if isinstance(selector, list) else
                          ("" if selector is None else "-" + selector))
                evidence_path = _fresh_artifact_path(
                    Path(str(cell_dir / f"attempt-{attempt}" / phase) + suffix + ".json")
                )
                dispatch_attempt = evidence_path.name
            result = invoke_acquisition(
                phase=phase, target=args.target,
                dispatch_key=hashlib.sha256(
                    f"a3-profile-microbench/v2/{current['id']}/{dispatch_attempt}/{phase}/{selector}".encode()
                ).hexdigest(), bundle=bundle,
                entrypoint="benchmarks/a3_profile_microbench.py",
                workload_args=["--experiment", current["experiment"], "--sweep",
                               current["sweep"], "--variant", current["variant"],
                               "--launches", "1"], evidence_path=evidence_path,
                selector=selector,
                basic_evidence=basic_path if phase in {"pipe", "timeline"} else None,
                cpl_remote=args.cpl_remote,
                launch_count=(len(current["kernel_names"]) if phase == "basic" else 1),
            )
            if phase == "basic" and result.get("ok"):
                successful_basic["path"] = basic_path
            return result
        return acquire

    results = []
    for position in range(0, len(cells), 2):
        pair = cells[position:position + 2]
        if len(pair) == 2:
            results.extend(run_pair_interleaved(
                pair, cells_root, acquire_for, samples=args.samples,
                max_attempts=args.max_attempts, max_relative_range=args.max_relative_range,
                identities=identities,
            ))
        else:
            cell = pair[0]
            cell_dir = cells_root / cell["id"].replace("/", "__")
            results.append(run_cell(
                cell, cell_dir, acquire_for(cell, cell_dir), samples=args.samples,
                max_attempts=args.max_attempts, max_relative_range=args.max_relative_range,
                campaign_identity_sha256=identities[cell["id"]],
            ))
    value = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "campaign_identity": identity_base,
        "target": args.target,
        "shard": {"index": args.shard_index, "count": args.shard_count},
        "raw_cells": results,
        "results": summarize_hypotheses(results, manifest),
    }
    validate_report_hypotheses(value)
    _atomic_json(args.output_dir / "shard-result.json", value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
