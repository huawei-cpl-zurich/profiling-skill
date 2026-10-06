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


class SubprocessBackend:
    """Invoke the pinned JSON benchmark backend without interpreting remotes."""

    def __init__(self, command: list[str], cwd: Path, timeout: int):
        self.command, self.cwd, self.timeout = command, cwd, timeout

    def __call__(self, request: dict) -> dict:
        try:
            run = subprocess.run(
                self.command, cwd=self.cwd, input=json.dumps(request), text=True,
                capture_output=True, timeout=self.timeout, check=False,
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
            if state.get("baseline_sha256") != self.baseline["sha256"]:
                return self._infrastructure("controller baseline changed during the experiment")
            state["operations_consumed"] = budget["operations_consumed"]
            terminal = state.get("terminal")
            if terminal:
                if remeasure_handle is None:
                    return terminal
                if (terminal.get("status") != "measurement_pending"
                        or terminal.get("handle") != remeasure_handle):
                    return self._infrastructure(
                        "remeasure handle does not match a pending measurement",
                        handle=remeasure_handle,
                    )
                state.pop("terminal")
                state.pop("profile", None)
                state.pop("confirmation", None)
                state.pop("post_control", None)
                state.pop("post_control_drift_ratio", None)
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
            "admission_controls": [], "quarantined_devices": [],
            "quarantine_controls": {}, "confirmation_count": 0,
        }

    def _request(self, state: dict, state_path: Path, request: dict,
                 budget: dict, budget_path: Path) -> dict | None:
        pending = state.get("pending")
        if isinstance(pending, dict):
            request = pending["request"]
        elif budget["operations_consumed"] >= self.request_budget:
            return self._infrastructure("24-operation experiment budget exhausted")
        result = self.backend(request)
        if not isinstance(result, dict):
            result = {"status": "infrastructure_error", "failure_type": "protocol_error",
                      "diagnostics": "backend response is not an object", "handle": None}
        status = result.get("status")
        record = {"request": request, "status": status,
                  "handle": result.get("handle"),
                  "failure_type": result.get("failure_type"),
                  "diagnostics": _bounded(result.get("diagnostics"))}
        if status == "infrastructure_error" or status not in ({"ok"} | CANDIDATE_FAILURES):
            state["infrastructure_attempts"] += 1
            state["operations"].append(record)
            state["pending"] = {"request": request, "handle": result.get("handle")}
            _atomic_json(state_path, state)
            reason = record["diagnostics"] or record["failure_type"] or "backend infrastructure error"
            if state["infrastructure_attempts"] > self.infrastructure_retry_budget:
                reason = f"infrastructure retry budget exhausted: {reason}"
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
                request = {**base, "action": "profile", "cases": self.development_cases,
                           "repeats": self.profile_repeats, "round": state["experiment"]}
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
                if stage == "profile":
                    state["profile"] = timing
                    if timing["variability_ratio"] > self.variability_threshold:
                        state["confirmation_count"] = 1
                        state["stage"] = "confirmation"
                    else:
                        state["stage"] = "post_control"
                else:
                    state["confirmation"] = timing
                    state["stage"] = "post_control"
                _atomic_json(state_path, state)
            elif stage == "post_control":
                request = {**base, "action": "calibrate", "phase": "after",
                           "wave": state["experiment"],
                           "attempt_id": f"experiment-{state['experiment']}-after"}
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
        return {"case_results": normalized, "samples_us": aggregate, "median_us": median,
                "variability_ratio": (max(aggregate) - min(aggregate)) / median}

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

    def _policy(self, state: dict, handle: str, post_control: str) -> dict:
        return {
            "schema": POLICY_SCHEMA, "selected_device": state["selected"]["id"],
            "admission_controls": state["admission_controls"],
            "submission_candidate_sha256": state["candidate_sha256"],
            "submitted_handles": [handle], "observed_handles": [handle],
            "infra_retries": 0, "retry_budget": self.infrastructure_retry_budget,
            "quarantined_devices": state["quarantined_devices"],
            "quarantine_controls": state["quarantine_controls"],
            "confirmation_count": state["confirmation_count"],
            "post_control": post_control,
            "variability_threshold": self.variability_threshold,
            "operations_consumed": state["operations_consumed"],
            "request_budget": self.request_budget,
            "infrastructure_attempts": state["infrastructure_attempts"],
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
        profile = state["profile"]
        confirmation = state.get("confirmation")
        noisy = profile["variability_ratio"] > self.variability_threshold
        confirmation_noisy = bool(
            confirmation and confirmation["variability_ratio"] > self.variability_threshold)
        status = "measurement_pending" if (
            state["post_control"] == "drift" or (noisy and confirmation_noisy)
        ) else "ok"
        policy = self._policy(state, profile["handle"], state["post_control"])
        policy["sample_count"] = self.profile_repeats
        policy["variability_ratio"] = profile["variability_ratio"]
        if confirmation:
            policy["confirmation"] = {
                "candidate_sha256": state["candidate_sha256"],
                "samples_us": confirmation["samples_us"],
                "sample_count": self.profile_repeats,
                "median_us": confirmation["median_us"],
                "variability_ratio": confirmation["variability_ratio"],
            }
        baseline_median = math.exp(sum(
            math.log(row["median_us"]) for row in self.baseline["case_medians_us"]
        ) / len(self.baseline["case_medians_us"]))
        before, after = state["calibration_before"], state["calibration_after"]
        local_reference = math.sqrt(before["median_us"] * after["median_us"])
        factor = self.baseline["control_median_us"] / local_reference
        normalized_samples = [value * factor for value in profile["samples_us"]]
        normalized_median = statistics.median(normalized_samples)
        return {
            "status": status, "experiment": state["experiment"],
            "candidate_sha256": state["candidate_sha256"],
            "manifest_sha256": state["manifest_sha256"],
            "handle": profile["handle"], "device": state["selected"]["id"],
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
