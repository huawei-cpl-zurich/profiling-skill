#!/usr/bin/env python3
"""Durable audited-controller adapter for native BZ-A3 benchmark jobs.

The adapter is deliberately transport neutral.  It sends JSON requests to a
configured benchmark backend, which is normally ``benchmark_backend.py`` wired
to ``bz_a3_job_client.py``.  The backend owns remote staging and in-place
``msprof op`` analysis; this layer owns the audited experiment policy.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Callable


CONFIG_SCHEMA = "profiling-skill/audited-bz-controller-config/v1"
POLICY_SCHEMA = "profiling-skill/controller-policy/v1"
REMEASURE_TRANSITION_SCHEMA = "profiling-skill/controller-remeasure-transition/v1"
CANDIDATE_FAILURES = {
    "candidate_error", "submission_error", "compile_error", "compilation_error",
    "runtime_error", "correctness_error",
}
MAX_DIAGNOSTICS = 4096


class ControllerError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(document: object) -> str:
    return hashlib.sha256(json.dumps(
        document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _bounded(value: object) -> str:
    text = str(value or "").strip()
    return text if len(text) <= MAX_DIAGNOSTICS else text[:MAX_DIAGNOSTICS] + "...[truncated]"


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w") as stream:
        json.dump(document, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _positive(value: object) -> bool:
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value > 0)


def _valid_selector_mapping(declared: object, resolved: object) -> bool:
    return (isinstance(declared, str) and bool(declared)
            and isinstance(resolved, str) and bool(resolved)
            and (resolved == declared
                 or any(declared == resolved + suffix
                        for suffix in ("_mix_aic", "_mix_aiv"))))


def _selector_identity(value: dict) -> tuple[str, str]:
    kernel = value.get("kernel_name")
    explicit = ("declared_kernel_name" in value or "resolved_kernel_name" in value)
    declared = value.get("declared_kernel_name") if explicit else kernel
    resolved = value.get("resolved_kernel_name") if explicit else kernel
    if kernel != declared or not _valid_selector_mapping(declared, resolved):
        raise ControllerError("profile response selector identity is invalid")
    return declared, resolved


class SubprocessBackend:
    """Invoke the pinned JSON benchmark backend without interpreting remotes."""

    def __init__(self, command: list[str], cwd: Path, timeout: int):
        self.command, self.cwd, self.timeout = command, cwd, timeout

    def __call__(self, request: dict) -> dict:
        return self._invoke(request, observe=False)

    def observe(self, request: dict) -> dict:
        return self._invoke(request, observe=True)

    def _invoke(self, request: dict, *, observe: bool) -> dict:
        environment = dict(os.environ)
        environment.pop("PROFILING_SKILL_CONTROLLER_MODE", None)
        environment["PROFILING_SKILL_CONTROLLER_REQUEST_SHA256"] = _json_sha(request)
        if observe:
            environment["PROFILING_SKILL_CONTROLLER_MODE"] = "observe"
        try:
            run = subprocess.run(
                self.command, cwd=self.cwd, input=json.dumps(request), text=True,
                capture_output=True, timeout=self.timeout, check=False, env=environment,
            )
        except subprocess.TimeoutExpired as failure:
            output = (failure.stdout or "") + (failure.stderr or "")
            return {"status": "infrastructure_error", "failure_type": "transport_timeout",
                    "diagnostics": _bounded(output), "handle": None}
        except OSError as failure:
            return {"status": "infrastructure_error", "failure_type": "transport_error",
                    "diagnostics": _bounded(failure), "handle": None}
        try:
            result = json.loads(run.stdout)
        except json.JSONDecodeError as failure:
            return {"status": "infrastructure_error", "failure_type": "protocol_error",
                    "diagnostics": _bounded(f"{failure}; stderr={run.stderr}"),
                    "handle": None}
        if not isinstance(result, dict):
            return {"status": "infrastructure_error", "failure_type": "protocol_error",
                    "diagnostics": "backend response is not an object", "handle": None}
        if run.stderr:
            result["diagnostics"] = _bounded(
                f"{result.get('diagnostics', '')}\n{run.stderr}".strip())
        if run.returncode and result.get("status") == "ok":
            return {"status": "infrastructure_error", "failure_type": "protocol_error",
                    "diagnostics": f"backend exited {run.returncode} after success",
                    "handle": result.get("handle")}
        return result


class AuditedBzController:
    """Run and durably checkpoint one audited correctness/profile transaction."""

    def __init__(
        self, *, repo: Path, state_dir: Path, benchmark: str,
        backend: Callable[[dict], dict], devices: list[dict],
        development_cases: list[int], all_cases: list[int], round_count: int,
        baseline: dict,
        request_budget: int = 24, profile_repeats: int = 3,
        variability_threshold: float = 0.25, control_drift_threshold: float = 0.2,
        infrastructure_retry_budget: int = 3,
    ):
        self.repo, self.state_dir, self.benchmark = repo.resolve(), state_dir.resolve(), benchmark
        self.backend, self.devices = backend, devices
        self.development_cases, self.all_cases = development_cases, all_cases
        self.baseline = baseline
        self.round_count, self.request_budget = round_count, request_budget
        self.profile_repeats = profile_repeats
        self.variability_threshold = variability_threshold
        self.control_drift_threshold = control_drift_threshold
        self.infrastructure_retry_budget = infrastructure_retry_budget
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        if not isinstance(self.benchmark, str) or not self.benchmark:
            raise ControllerError("benchmark must be non-empty")
        if (type(self.round_count) is not int or self.round_count < 1
                or type(self.request_budget) is not int or self.request_budget < 1
                or type(self.profile_repeats) is not int or self.profile_repeats != 3
                or type(self.infrastructure_retry_budget) is not int
                or self.infrastructure_retry_budget < 0):
            raise ControllerError("round and budget values are invalid")
        if (not _positive(self.variability_threshold)
                or not _positive(self.control_drift_threshold)):
            raise ControllerError("timing thresholds must be positive")
        if not self.devices:
            raise ControllerError("at least one admitted device candidate is required")
        seen = set()
        for item in self.devices:
            if (not isinstance(item, dict) or set(item) != {"id", "device"}
                    or not isinstance(item["id"], str) or not item["id"]
                    or item["id"] in seen or type(item["device"]) is not int
                    or item["device"] < 0):
                raise ControllerError("device entries require unique id and non-negative device")
            seen.add(item["id"])
        for name, cases in (("development_cases", self.development_cases),
                            ("all_cases", self.all_cases)):
            if (not isinstance(cases, list) or not cases
                    or len(set(cases)) != len(cases)
                    or any(type(case) is not int or case < 0 for case in cases)):
                raise ControllerError(f"{name} must contain unique case indices")
        self._validate_baseline()

    def _validate_baseline(self) -> None:
        value = self.baseline
        required = {"schema", "benchmark", "case_medians_us",
                    "control_median_us", "sha256"}
        if (not isinstance(value, dict) or set(value) != required
                or value.get("schema") != "profiling-skill/baseline-timing/v1"
                or value.get("benchmark") != self.benchmark):
            raise ControllerError("baseline timing document is invalid")
        expected_hash = _json_sha({key: item for key, item in value.items()
                                   if key != "sha256"})
        if value.get("sha256") != expected_hash:
            raise ControllerError("baseline sha256 does not bind its timing values")
        rows = value.get("case_medians_us")
        if (not isinstance(rows, list)
                or [row.get("case") if isinstance(row, dict) else None for row in rows]
                != self.development_cases
                or any(set(row) != {"case", "median_us"}
                       or not _positive(row["median_us"]) for row in rows)):
            raise ControllerError("baseline cases must match development cases exactly")
        if not _positive(value.get("control_median_us")):
            raise ControllerError("baseline control median must be positive")

    def run(self, experiment: int, candidate_hash: str, manifest_hash: str,
            *, observe_handle: str | None = None,
            remeasure_handle: str | None = None) -> dict:
        if observe_handle is not None and remeasure_handle is not None:
            return self._infrastructure("observe and remeasure handles are mutually exclusive")
        if type(experiment) is not int or not 1 <= experiment <= self.round_count:
            return self._infrastructure("experiment is outside the configured round range")
        candidate, manifest = self.repo / "candidate.py", self.repo / "candidate.manifest.json"
        if (not candidate.is_file() or not manifest.is_file()
                or _sha(candidate) != candidate_hash or _sha(manifest) != manifest_hash):
            return self._infrastructure("frozen candidate or manifest hash does not match")
        key = hashlib.sha256(
            f"{experiment}:{candidate_hash}:{manifest_hash}".encode()).hexdigest()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        root = self.state_dir / key
        root.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / "controller.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            budget_path = self.state_dir / "budget.json"
            if budget_path.is_file():
                try:
                    budget = json.loads(budget_path.read_text())
                except (OSError, json.JSONDecodeError):
                    return self._infrastructure("controller budget ledger is corrupt")
            else:
                budget = {"schema": "profiling-skill/audited-bz-budget/v1",
                          "operations_consumed": 0, "request_budget": self.request_budget}
            if (budget.get("schema") != "profiling-skill/audited-bz-budget/v1"
                    or budget.get("request_budget") != self.request_budget
                    or type(budget.get("operations_consumed")) is not int
                    or budget["operations_consumed"] < 0):
                return self._infrastructure("controller budget ledger does not match configuration")
            state_path = root / "state.json"
            if state_path.is_file():
                try:
                    state = json.loads(state_path.read_text())
                except (OSError, json.JSONDecodeError):
                    return self._infrastructure("controller state is corrupt")
            else:
                state = self._new_state(experiment, candidate_hash, manifest_hash)
            state.setdefault("infrastructure_retries", 0)
            state.setdefault("measurement_generation", 0)
            if state.get("baseline_sha256") != self.baseline["sha256"]:
                return self._infrastructure("controller baseline changed during the experiment")
            state["operations_consumed"] = budget["operations_consumed"]
            terminal = state.get("terminal")
            remeasure_source = None
            if terminal:
                if remeasure_handle is None:
                    return terminal
                if (terminal.get("status") != "measurement_pending"
                        or terminal.get("handle") != remeasure_handle):
                    return self._infrastructure(
                        "remeasure handle does not match a pending measurement",
                        handle=remeasure_handle,
                    )
                remeasure_source = {
                    "receipt_sha256": _json_sha(terminal),
                    "handle": terminal["handle"],
                    "measurement_generation": state["measurement_generation"],
                }
                state.pop("terminal")
                state.pop("profile", None)
                state.pop("confirmation", None)
                state.pop("post_control", None)
                state.pop("post_control_drift_ratio", None)
                state["measurement_generation"] += 1
                state["confirmation_count"] = 0
                state["stage"] = "profile"
                _atomic_json(state_path, state)
            elif remeasure_handle is not None:
                return self._infrastructure(
                    "remeasure requires a completed measurement_pending receipt",
                    handle=remeasure_handle,
                )
            pending = state.get("pending")
            if observe_handle is not None and (
                    not isinstance(pending, dict)
                    or pending.get("handle") != observe_handle):
                return self._infrastructure(
                    "observe handle does not match the checkpointed operation",
                    handle=observe_handle,
                )
            if observe_handle is None and isinstance(pending, dict) and pending.get("handle"):
                return self._infrastructure(
                    "checkpoint has a durable handle; reobserve that handle",
                    handle=pending["handle"],
                )
            receipt = self._advance(state, state_path, budget, budget_path)
            if (remeasure_source is not None
                    and receipt.get("status") in {
                        "ok", "candidate_error", "measurement_pending",
                    }):
                history = receipt.get("policy", {}).get("operation_history")
                post_control = receipt.get("calibration", {}).get("after", {}).get("handle")
                receipt["remeasure_transition"] = {
                    "schema": REMEASURE_TRANSITION_SCHEMA,
                    "pending_receipt_sha256": remeasure_source["receipt_sha256"],
                    "pending_handle": remeasure_source["handle"],
                    "candidate_sha256": candidate_hash,
                    "manifest_sha256": manifest_hash,
                    "experiment": experiment,
                    "from_measurement_generation": remeasure_source[
                        "measurement_generation"
                    ],
                    "to_measurement_generation": state["measurement_generation"],
                    "profile_handle": receipt.get("handle"),
                    "post_control_handle": post_control,
                    "operation_history_sha256": _json_sha(history),
                }
            if (observe_handle is not None
                    and receipt.get("status") == "infrastructure_error"
                    and isinstance(receipt.get("handle"), str)
                    and receipt["handle"] != observe_handle):
                receipt.update(
                    experiment=experiment,
                    candidate_sha256=candidate_hash,
                    manifest_sha256=manifest_hash,
                    observe_transition={
                        "schema": "profiling-skill/controller-observe-transition/v1",
                        "observed_handle": observe_handle,
                        "pending_handle": receipt["handle"],
                        "operation_history": self._operation_history(state),
                    },
                )
            if receipt.get("status") in {"ok", "candidate_error", "measurement_pending"}:
                state["terminal"] = receipt
                state["pending"] = None
                _atomic_json(state_path, state)
            return receipt

    def _new_state(self, experiment: int, candidate_hash: str, manifest_hash: str) -> dict:
        selected = self.devices[(experiment - 1) % len(self.devices)]
        return {
            "schema": "profiling-skill/audited-bz-controller-state/v1",
            "experiment": experiment, "candidate_sha256": candidate_hash,
            "manifest_sha256": manifest_hash, "selected": selected,
            "baseline_sha256": self.baseline["sha256"],
            "stage": "admission", "operations_consumed": 0,
            "infrastructure_attempts": 0, "operations": [], "pending": None,
            "infrastructure_retries": 0, "measurement_generation": 0,
            "admission_controls": [], "quarantined_devices": [],
            "quarantine_controls": {}, "confirmation_count": 0,
        }

    def _request(self, state: dict, state_path: Path, request: dict,
                 budget: dict, budget_path: Path) -> dict | None:
        pending = state.get("pending")
        if isinstance(pending, dict):
            if state["infrastructure_retries"] >= self.infrastructure_retry_budget:
                return self._infrastructure(
                    "infrastructure retry budget exhausted", handle=pending.get("handle"))
            state["infrastructure_retries"] += 1
            mode = "observe" if pending.get("handle") else "retry_submit"
            request = pending["request"]
        elif budget["operations_consumed"] >= self.request_budget:
            return self._infrastructure(
                f"{self.request_budget}-operation experiment budget exhausted"
            )
        else:
            mode = "submit"
        observe = getattr(self.backend, "observe", None)
        result = observe(request) if mode == "observe" and callable(observe) \
            else self.backend(request)
        if not isinstance(result, dict):
            result = {"status": "infrastructure_error", "failure_type": "protocol_error",
                      "diagnostics": "backend response is not an object", "handle": None}
        status = result.get("status")
        terminal = status in ({"ok"} | CANDIDATE_FAILURES)
        record = {"request": request, "request_sha256": _json_sha(request),
                  "mode": mode, "status": status, "terminal": terminal,
                  "handle": result.get("handle"),
                  "fusion_gate": result.get("fusion_gate"),
                  "fusion_handle": result.get("fusion_handle"),
                  "failure_type": result.get("failure_type"),
                  "diagnostics": _bounded(result.get("diagnostics"))}
        if status == "infrastructure_error" or status not in ({"ok"} | CANDIDATE_FAILURES):
            state["infrastructure_attempts"] += 1
            state["operations"].append(record)
            state["pending"] = {"request": request, "handle": result.get("handle")}
            _atomic_json(state_path, state)
            reason = record["diagnostics"] or record["failure_type"] or "backend infrastructure error"
            return self._infrastructure(reason, handle=result.get("handle"))
        state["pending"] = None
        budget["operations_consumed"] += 1
        state["operations_consumed"] = budget["operations_consumed"]
        state["operations"].append(record)
        _atomic_json(budget_path, budget)
        _atomic_json(state_path, state)
        result["diagnostics"] = _bounded(result.get("diagnostics"))
        return result

    def _advance(self, state: dict, state_path: Path,
                 budget: dict, budget_path: Path) -> dict:
        while True:
            selected = state["selected"]
            base = {"protocol_version": 1, "benchmark": self.benchmark,
                    "device": selected["device"]}
            stage = state["stage"]
            if stage == "admission":
                request = {**base, "action": "calibrate", "phase": "before",
                           "wave": state["experiment"],
                           "attempt_id": f"experiment-{state['experiment']}-before"}
                result = self._request(state, state_path, request, budget, budget_path)
                if result.get("status") == "infrastructure_error":
                    return result
                if result.get("status") != "ok":
                    return self._infrastructure("known-good admission control did not pass",
                                                handle=result.get("handle"))
                try:
                    state["calibration_before"] = self._calibration(result)
                except ControllerError as failure:
                    return self._infrastructure(str(failure), handle=result.get("handle"))
                state["admission_controls"].append({
                    "device": selected["id"], "status": "pass", "healthy": True,
                    "idle": True, "warmed": True, "handle": result.get("handle"),
                })
                state["stage"] = "check"
                _atomic_json(state_path, state)
            elif stage == "check":
                full = state["experiment"] == self.round_count
                request = {**base, "action": "check", "scope": "full" if full else "development",
                           "round": state["experiment"],
                           "cases": self.all_cases if full else self.development_cases}
                result = self._request(state, state_path, request, budget, budget_path)
                if result.get("status") == "infrastructure_error":
                    return result
                if result.get("status") in CANDIDATE_FAILURES or result.get("passed") is False:
                    return self._candidate_receipt(state, result)
                if result.get("status") != "ok" or result.get("passed") is not True:
                    return self._infrastructure("correctness backend returned no pass decision",
                                                handle=result.get("handle"))
                state["stage"] = "profile"
                _atomic_json(state_path, state)
            elif stage in {"profile", "confirmation"}:
                generation = state["measurement_generation"]
                request = {**base, "action": "profile", "cases": self.development_cases,
                           "repeats": self.profile_repeats, "round": state["experiment"],
                           "attempt_id": (f"experiment-{state['experiment']}-measurement-"
                                          f"{generation}-primary"
                                          if stage == "profile" else
                                          f"experiment-{state['experiment']}-measurement-"
                                          f"{generation}-confirmation")}
                result = self._request(state, state_path, request, budget, budget_path)
                if result.get("status") == "infrastructure_error":
                    return result
                if result.get("status") in CANDIDATE_FAILURES:
                    return self._candidate_receipt(state, result)
                try:
                    timing = self._timing(result)
                except ControllerError as failure:
                    return self._infrastructure(str(failure), handle=result.get("handle"))
                timing["handle"] = result.get("handle")
                timing["compact_artifacts"] = self._compact_artifacts(result)
                if not isinstance(timing["handle"], str) or not timing["handle"]:
                    return self._infrastructure("profile timing lacks a durable handle")
                if stage == "profile":
                    state["profile"] = timing
                    if timing["variability_ratio"] > self.variability_threshold:
                        state["confirmation_count"] = 1
                        state["stage"] = "confirmation"
                    else:
                        state["stage"] = "post_control"
                else:
                    if _selector_identity(timing) != _selector_identity(state["profile"]):
                        return self._infrastructure(
                            "confirmation identity does not match the primary kernel",
                            handle=timing["handle"],
                        )
                    state["confirmation"] = timing
                    state["stage"] = "post_control"
                _atomic_json(state_path, state)
            elif stage == "post_control":
                generation = state["measurement_generation"]
                request = {**base, "action": "calibrate", "phase": "after",
                           "wave": state["experiment"],
                           "attempt_id": (f"experiment-{state['experiment']}-measurement-"
                                          f"{generation}-after")}
                result = self._request(state, state_path, request, budget, budget_path)
                if result.get("status") == "infrastructure_error":
                    return result
                if result.get("status") != "ok":
                    return self._infrastructure("known-good post control did not pass",
                                                handle=result.get("handle"))
                try:
                    state["calibration_after"] = self._calibration(result)
                except ControllerError as failure:
                    return self._infrastructure(str(failure), handle=result.get("handle"))
                before = state["calibration_before"]["median_us"]
                after = state["calibration_after"]["median_us"]
                drift = abs(after - before) / before
                state["post_control"] = "drift" if drift > self.control_drift_threshold else "stable"
                state["post_control_drift_ratio"] = drift
                return self._success_receipt(state)
            else:
                return self._infrastructure(f"unknown controller stage {stage!r}")

    def _timing(self, result: dict) -> dict:
        rows = result.get("cases")
        declared, resolved = _selector_identity(result)
        if (not isinstance(rows, list) or len(rows) != len(self.development_cases)
                or [row.get("case") if isinstance(row, dict) else None for row in rows]
                != self.development_cases):
            raise ControllerError("profile response has invalid case rows")
        samples_by_case = []
        normalized = []
        for row in rows:
            samples = row.get("samples_us")
            if (not isinstance(samples, list) or len(samples) != self.profile_repeats
                    or not all(_positive(value) for value in samples)
                    or not _positive(row.get("median_us"))
                    or not math.isclose(statistics.median(samples), row["median_us"], rel_tol=1e-12)):
                raise ControllerError("profile response has invalid fixed samples")
            values = [float(value) for value in samples]
            samples_by_case.append(values)
            normalized.append({"case": row["case"], "samples_us": values,
                               "median_us": float(row["median_us"])})
        aggregate = [math.exp(sum(math.log(value) for value in repetition) / len(repetition))
                     for repetition in zip(*samples_by_case)]
        median = statistics.median(aggregate)
        timing = {"kernel_name": declared, "declared_kernel_name": declared,
                  "resolved_kernel_name": resolved, "case_results": normalized,
                  "samples_us": aggregate, "median_us": median,
                  "variability_ratio": (max(aggregate) - min(aggregate)) / median}
        fusion_gate, fusion_handle = result.get("fusion_gate"), result.get("fusion_handle")
        if fusion_gate is not None or fusion_handle is not None:
            if (not isinstance(fusion_handle, str) or not fusion_handle
                    or not isinstance(fusion_gate, dict)
                    or set(fusion_gate) != {
                        "entrypoint", "kernel_name", "cases", "logical_launches_per_case"
                    }
                    or not isinstance(fusion_gate["entrypoint"], str)
                    or not fusion_gate["entrypoint"]
                    or fusion_gate["kernel_name"] != declared
                    or fusion_gate["cases"] != self.all_cases
                    or fusion_gate["logical_launches_per_case"] != 1):
                raise ControllerError("profile response has invalid fusion authorization")
            timing.update(fusion_gate=fusion_gate, fusion_handle=fusion_handle)
        return timing

    def _calibration(self, result: dict) -> dict:
        samples = result.get("samples_us")
        median = result.get("median_us")
        if (not isinstance(samples, list) or len(samples) != self.profile_repeats
                or not all(_positive(value) for value in samples)
                or not _positive(median)
                or not math.isclose(statistics.median(samples), median, rel_tol=1e-12)):
            raise ControllerError("known-good control lacks fixed-sample timing")
        return {"samples_us": [float(value) for value in samples],
                "median_us": float(median), "handle": result.get("handle")}

    @staticmethod
    def _compact_artifacts(result: dict) -> list[str]:
        artifacts = result.get("artifacts")
        if not isinstance(artifacts, dict):
            return []
        value = artifacts.get("remote_profile_evidence")
        return [value] if isinstance(value, str) and value else []

    @staticmethod
    def _operation_history(state: dict) -> list[dict]:
        history = []
        for record in state["operations"]:
            retained = {key: record.get(key) for key in (
                "request_sha256", "mode", "status", "terminal", "handle"
            )} | {
                "action": record["request"].get("action"),
                "attempt_id": record["request"].get("attempt_id"),
            }
            if record.get("fusion_gate") is not None or record.get("fusion_handle") is not None:
                retained.update(
                    fusion_gate=record.get("fusion_gate"),
                    fusion_handle=record.get("fusion_handle"),
                )
            history.append(retained)
        return history

    def _policy(self, state: dict, handle: str, post_control: str) -> dict:
        history = self._operation_history(state)
        submitted = []
        observed = []
        for record in history:
            retained = record["handle"]
            if (record["mode"] in {"submit", "retry_submit"}
                    and isinstance(retained, str) and retained not in submitted):
                submitted.append(retained)
            if (record["terminal"] and isinstance(retained, str)
                    and retained not in observed):
                observed.append(retained)
        return {
            "schema": POLICY_SCHEMA, "selected_device": state["selected"]["id"],
            "admission_controls": state["admission_controls"],
            "submission_candidate_sha256": state["candidate_sha256"],
            "submitted_handles": submitted, "observed_handles": observed,
            "infra_retries": state["infrastructure_retries"],
            "retry_budget": self.infrastructure_retry_budget,
            "operation_history": history,
            "quarantined_devices": state["quarantined_devices"],
            "quarantine_controls": state["quarantine_controls"],
            "confirmation_count": state["confirmation_count"],
            "post_control": post_control,
            "variability_threshold": self.variability_threshold,
            "operations_consumed": state["operations_consumed"],
            "request_budget": self.request_budget,
            "infrastructure_attempts": state["infrastructure_attempts"],
            "measurement_generation": state["measurement_generation"],
            "post_control_drift_ratio": state.get("post_control_drift_ratio"),
        }

    def _candidate_receipt(self, state: dict, result: dict) -> dict:
        handle = result.get("handle")
        if not isinstance(handle, str) or not handle:
            return self._infrastructure("candidate failure lacks durable handle")
        failure = result.get("failure_type") or result.get("status")
        if failure == "candidate_error":
            failure = "correctness_error"
        return {
            "status": "candidate_error", "failure_type": failure,
            "reason": _bounded(result.get("diagnostics")) or str(failure),
            "experiment": state["experiment"],
            "candidate_sha256": state["candidate_sha256"],
            "manifest_sha256": state["manifest_sha256"], "handle": handle,
            "device": state["selected"]["id"],
            "policy": self._policy(state, handle, "not_run"),
        }

    def _success_receipt(self, state: dict) -> dict:
        primary = state["profile"]
        confirmation = state.get("confirmation")
        profile = confirmation or primary
        declared, resolved = _selector_identity(profile)
        accepted_timing = "confirmation" if confirmation else "primary"
        noisy = profile["variability_ratio"] > self.variability_threshold
        status = "measurement_pending" if (
            state["post_control"] == "drift" or noisy
        ) else "ok"
        policy = self._policy(state, profile["handle"], state["post_control"])
        policy["sample_count"] = self.profile_repeats
        policy["variability_ratio"] = profile["variability_ratio"]
        policy["accepted_timing"] = accepted_timing
        policy["primary"] = self._timing_proof(state, primary)
        if confirmation:
            policy["confirmation"] = self._timing_proof(state, confirmation)
        baseline_median = math.exp(sum(
            math.log(row["median_us"]) for row in self.baseline["case_medians_us"]
        ) / len(self.baseline["case_medians_us"]))
        before, after = state["calibration_before"], state["calibration_after"]
        local_reference = math.sqrt(before["median_us"] * after["median_us"])
        factor = self.baseline["control_median_us"] / local_reference
        normalized_samples = [value * factor for value in profile["samples_us"]]
        normalized_median = statistics.median(normalized_samples)
        receipt = {
            "status": status, "experiment": state["experiment"],
            "candidate_sha256": state["candidate_sha256"],
            "manifest_sha256": state["manifest_sha256"],
            "handle": profile["handle"], "device": state["selected"]["id"],
            "kernel_name": declared,
            "declared_kernel_name": declared,
            "resolved_kernel_name": resolved,
            "samples_us": profile["samples_us"], "median_us": profile["median_us"],
            "baseline_median_us": baseline_median, "baseline": self.baseline,
            "calibration": {
                "before": before, "after": after,
                "local_reference_median_us": local_reference,
                "baseline_reference_median_us": self.baseline["control_median_us"],
                "normalization_factor": factor,
            },
            "normalized_samples_us": normalized_samples,
            "normalized_median_us": normalized_median,
            "speedup_vs_baseline": baseline_median / normalized_median,
            "case_results": profile["case_results"],
            "compact_artifacts": profile["compact_artifacts"], "policy": policy,
        }
        if "fusion_gate" in profile:
            receipt.update(
                fusion_gate=profile["fusion_gate"],
                fusion_handle=profile["fusion_handle"],
            )
        return receipt

    def _timing_proof(self, state: dict, timing: dict) -> dict:
        proof = {
            "candidate_sha256": state["candidate_sha256"],
            "kernel_name": timing["kernel_name"], "handle": timing["handle"],
            "declared_kernel_name": timing.get(
                "declared_kernel_name", timing["kernel_name"]),
            "resolved_kernel_name": timing.get(
                "resolved_kernel_name", timing["kernel_name"]),
            "samples_us": timing["samples_us"], "sample_count": self.profile_repeats,
            "median_us": timing["median_us"],
            "variability_ratio": timing["variability_ratio"],
            "case_results": timing["case_results"],
            "compact_artifacts": timing["compact_artifacts"],
        }
        if "fusion_gate" in timing:
            proof.update(
                fusion_gate=timing["fusion_gate"],
                fusion_handle=timing["fusion_handle"],
            )
        return proof

    @staticmethod
    def _infrastructure(reason: str, handle: str | None = None) -> dict:
        receipt = {"status": "infrastructure_error", "terminal": False,
                   "reason": _bounded(reason)}
        if handle:
            receipt["handle"] = handle
        return receipt


def _load_config(path: Path) -> dict:
    try:
        config = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as failure:
        raise ControllerError(f"cannot load controller config: {failure}") from failure
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ControllerError(f"controller config requires schema {CONFIG_SCHEMA}")
    command = config.get("backend_command")
    if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command):
        raise ControllerError("backend_command must be a non-empty string array")
    return config


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--experiment", type=int, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--observe-handle")
    parser.add_argument("--remeasure-handle")
    args = parser.parse_args()
    try:
        config = _load_config(args.config)
        repo = Path.cwd()
        backend = SubprocessBackend(
            config["backend_command"], repo, int(config.get("timeout_seconds", 3600)))
        adapter = AuditedBzController(
            repo=repo, state_dir=args.state_dir, benchmark=config["benchmark"],
            backend=backend, devices=config["devices"],
            development_cases=config["development_cases"], all_cases=config["all_cases"],
            baseline=config["baseline"],
            round_count=config["round_count"], request_budget=config.get("request_budget", 24),
            profile_repeats=config.get("profile_repeats", 3),
            variability_threshold=config.get("variability_threshold", 0.25),
            control_drift_threshold=config.get("control_drift_threshold", 0.2),
            infrastructure_retry_budget=config.get("infrastructure_retry_budget", 3),
        )
        receipt = adapter.run(
            args.experiment, args.candidate_sha256, args.manifest_sha256,
            observe_handle=args.observe_handle,
            remeasure_handle=args.remeasure_handle,
        )
    except (ControllerError, KeyError, TypeError, ValueError) as failure:
        receipt = AuditedBzController._infrastructure(str(failure))
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["status"] in {"ok", "candidate_error", "measurement_pending"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
