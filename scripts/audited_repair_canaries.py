#!/usr/bin/env python3
"""Run the three repair-rollout canaries without dispatching a campaign."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Callable

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
try:
    import audited_campaign_production as production
    import audited_runtime
    from audited_campaign import InfrastructureFailure
    from audited_resource_admission import (
        AdmissionError, CplRemoteResourcePool, file_sha256,
    )
finally:
    sys.path.pop(0)


RESULTS_SCHEMA = "profiling-skill/audited-repair-canary-results/v1"
RESUME_SCHEMA = "profiling-skill/audited-repair-canary-resume/v1"
INJECTION_SCHEMA = "profiling-skill/audited-repair-canary-injection/v1"
OBSERVER_SCHEMA = "profiling-skill/canary-observer-interrupt/v1"


class CanaryError(RuntimeError):
    pass


class CanaryInterrupted(CanaryError):
    """Test adapter equivalent of the deliberate observer interruption."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


def _valid_observer_marker(marker: object) -> bool:
    expected = {
        "schema", "job_request_sha256", "controller_request_sha256", "handle",
        "target", "device", "job_identity", "candidate_sha256", "manifest_sha256",
    }
    return (isinstance(marker, dict) and set(marker) == expected
            and marker.get("schema") == OBSERVER_SCHEMA
            and all(isinstance(marker.get(key), str)
                    and len(marker[key]) == 64
                    and all(character in "0123456789abcdef" for character in marker[key])
                    for key in ("job_request_sha256", "controller_request_sha256",
                                "candidate_sha256", "manifest_sha256"))
            and isinstance(marker.get("handle"), str) and marker["handle"]
            and isinstance(marker.get("target"), str) and marker["target"]
            and isinstance(marker.get("device"), int)
            and isinstance(marker.get("job_identity"), dict))


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _retain(path: Path, value: dict) -> dict:
    if path.is_file():
        try:
            retained = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise CanaryError(f"retained artifact is invalid JSON: {path}") from error
        if retained != value:
            raise CanaryError(f"retained artifact changed across resume: {path}")
        return retained
    _atomic_json(path, value)
    return value


def _retain_bytes(path: Path, value: bytes) -> None:
    if path.is_file():
        if path.read_bytes() != value:
            raise CanaryError(f"retained artifact changed across resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(value)
    temporary.replace(path)


class InjectedController:
    """Persist one replayable repair receipt around the real BZ controller."""

    durable_attempt_checkpoints = True

    def __init__(self, controller, mode: str, marker: Path):
        if mode not in {"repair", "passthrough"}:
            raise CanaryError(f"unsupported canary injection mode: {mode}")
        self.controller, self.mode, self.marker = controller, mode, Path(marker)

    def reproducibility_metadata(self) -> dict:
        metadata = getattr(self.controller, "reproducibility_metadata", lambda: {})()
        identity = {"adapter": "InjectedController", "mode": self.mode,
                    "controller": metadata}
        identity["identity_sha256"] = _json_sha(identity)
        return identity

    def __call__(self, number: int, candidate: str, manifest: str) -> dict:
        receipt = self.controller(number, candidate, manifest)
        if self.mode == "passthrough" or receipt.get("status") != "ok":
            return receipt
        retained = None
        if self.marker.is_file():
            try:
                retained = json.loads(self.marker.read_text())
            except json.JSONDecodeError as error:
                raise CanaryError("canary injection marker is invalid") from error
            if (not isinstance(retained, dict)
                    or retained.get("schema") != INJECTION_SCHEMA
                    or retained.get("mode") != self.mode):
                raise CanaryError("canary injection marker does not match this cell")
        matching = (isinstance(retained, dict)
                    and retained.get("experiment") == number
                    and retained.get("real_handle") == receipt.get("handle")
                    and retained.get("candidate_sha256") == candidate
                    and retained.get("manifest_sha256") == manifest
                    and retained.get("real_receipt_sha256") == _json_sha(receipt))
        if retained is not None and not matching:
            return receipt
        if retained is None:
            failed = copy.deepcopy(receipt)
            failed.update(status="candidate_error", failure_type="compile_error",
                          reason="deterministic canary compile diagnostic")
            for key in (
                "samples_us", "median_us", "baseline_median_us", "baseline",
                "calibration", "normalized_samples_us", "normalized_median_us",
                "speedup_vs_baseline", "case_results", "compact_artifacts",
                "kernel_name", "declared_kernel_name", "resolved_kernel_name",
            ):
                failed.pop(key, None)
            policy = failed["policy"]
            for key in ("sample_count", "variability_ratio", "accepted_timing",
                        "primary", "confirmation"):
                policy.pop(key, None)
            policy.update(confirmation_count=0, post_control="not_run")
            retained = {
                "schema": INJECTION_SCHEMA, "mode": self.mode,
                "experiment": number, "candidate_sha256": candidate,
                "manifest_sha256": manifest, "real_handle": receipt.get("handle"),
                "real_receipt_sha256": _json_sha(receipt),
                "injected_receipt": failed,
            }
            _retain(self.marker, retained)
        return copy.deepcopy(retained["injected_receipt"])

    def observe(self, number: int, candidate: str, manifest: str, handle: str) -> dict:
        return self.controller.observe(number, candidate, manifest, handle)

    def remeasure(self, number: int, candidate: str, manifest: str, handle: str) -> dict:
        return self.controller.remeasure(number, candidate, manifest, handle)


class LiveLauncher:
    """Narrow canary adapter over the production cell launcher."""

    def __init__(self, config: dict, mode: str):
        def controller_factory(command, repo, *, timeout):
            base = audited_runtime.CommandController(command, repo, timeout=timeout)
            injection = "repair" if mode == "repair" else "passthrough"
            return InjectedController(
                base, injection, repo.parent / "state" / "repair-injection.json",
            )

        self.launcher = production.ProductionCellLauncher(
            config, infrastructure_failure_type=InfrastructureFailure,
            controller_factory=controller_factory,
        )

    def launch(self, cell: dict, slot: dict) -> dict:
        return self.launcher.launch(cell, slot)

    def observe(self, cell: dict, slot: dict, handle: str) -> dict:
        return self.launcher.observe(cell, slot, handle)

    def verify(self, cell: dict) -> dict:
        return self.launcher.verify(cell)


class CanaryRunner:
    def __init__(self, config: dict, definition: Path, definition_sha256: str,
                 run_root: Path, admission_pool,
                 *, launcher_factory: Callable[[dict, str], object] = LiveLauncher):
        self.definition_path = Path(definition).resolve()
        self.definition_sha256 = definition_sha256
        self.run_root = Path(run_root).resolve()
        self.pool, self.launcher_factory = admission_pool, launcher_factory
        if _sha(self.definition_path) != definition_sha256:
            raise CanaryError("canary definition hash does not match pinned input")
        try:
            self.definition = json.loads(self.definition_path.read_text())
        except json.JSONDecodeError as error:
            raise CanaryError("canary definition is invalid JSON") from error
        if not isinstance(self.definition, dict):
            raise CanaryError("canary definition must be a JSON object")
        self._validate_definition()
        self.runtime_config_sha256 = production.document_sha256(config)
        self.config = copy.deepcopy(config)
        self.config["run_id"] = f"repair-canaries-{definition_sha256[:12]}"
        self.config["run_root"] = str(self.run_root / "cells")
        self.config["canary_definition"] = {
            "path": str(self.definition_path), "sha256": definition_sha256,
        }

    def _validate_definition(self) -> None:
        rows = self.definition.get("canaries")
        schema = self.definition.get("schema")
        if schema == "profiling-skill/audited-repair-canaries/v1":
            treatments = tuple(production.TREATMENT_SKILLS)
        elif schema == "profiling-skill/audited-repair-canaries/v2":
            raw_treatments = self.definition.get("treatments")
            treatments = tuple(raw_treatments) if isinstance(raw_treatments, list) else ()
        else:
            treatments = ()
        expected_rows = [
            (treatment, production.CANARY_EVIDENCE[treatment])
            for treatment in treatments if treatment in production.CANARY_EVIDENCE
        ]
        actual_rows = ([(item.get("treatment"), item.get("required_evidence"))
                        for item in rows
                        if set(item) == {"id", "treatment", "required_evidence"}
                        and isinstance(item.get("id"), str) and item["id"]]
                       if isinstance(rows, list)
                       and all(isinstance(item, dict) for item in rows) else [])
        identifiers = ([item.get("id") for item in rows]
                       if isinstance(rows, list)
                       and all(isinstance(item, dict) for item in rows) else [])
        rows_match = (actual_rows == expected_rows if schema ==
                      "profiling-skill/audited-repair-canaries/v2"
                      else {name: evidence for name, evidence in actual_rows}
                      == {name: evidence for name, evidence in expected_rows})
        gate = {"all_canaries_terminal_ok": True,
                "all_branches_offline_valid": True,
                "all_final_timings_positive": True,
                "minimum_repaired_canaries": sum(
                    "candidate-repair" in evidence for _, evidence in expected_rows),
                "resume_canary_required": any(
                    "checkpoint-resume" in evidence for _, evidence in expected_rows)}
        if (not treatments or len(treatments) != len(set(treatments))
                or len(expected_rows) != len(treatments)
                or len(identifiers) != len(set(identifiers))
                or self.definition.get("benchmark") != "matmul"
                or self.definition.get("request_budget") != 48
                or self.definition.get("max_candidate_repairs_per_round") != 2
                or self.definition.get("placement") != "dynamic-bz-a3-admission"
                or self.definition.get("gate") != gate
                or not rows_match):
            raise CanaryError("canary definition is not the declared treatment contract")

    def _cell(self, declaration: dict) -> dict:
        task = self.config.get("tasks", {}).get("matmul", {})
        prompt = self.config.get("prompt", {})
        return {
            "cell_id": declaration["id"], "task": "matmul",
            "treatment": declaration["treatment"], "round_count": 4,
            "request_budget": 48,
            "skills": list(production.TREATMENT_SKILLS[declaration["treatment"]]),
            "task_sha256": task.get("sha256"),
            "prompt_contract": {"task_sha256": task.get("sha256"),
                                "invariant_sha256": prompt.get("sha256")},
            **({"canary_fault": "interrupt-after-dispatch-once"}
               if "checkpoint-resume" in declaration["required_evidence"] else {}),
        }

    def _slot(self, cell: dict) -> dict:
        retained = (Path(self.config["run_root"]) / cell["cell_id"]
                    / "state" / "placement.json")
        if retained.is_file():
            try:
                return json.loads(retained.read_text())
            except json.JSONDecodeError as error:
                raise CanaryError("retained canary placement is invalid") from error
        slots = [slot for slot in self.pool.admit()
                 if isinstance(slot, dict)
                 and slot.get("healthy") is True and slot.get("idle") is True]
        if not slots:
            raise CanaryError("no healthy idle BZ-A3 device was admitted")
        return sorted(slots, key=lambda item: (item["target"], item["device"]))[0]

    def _checkpoint(self, cell: dict) -> tuple[dict, bytes]:
        path = (Path(self.config["run_root"]) / cell["cell_id"] / "repo"
                / ".experiment" / "blocked.json")
        try:
            raw, value = path.read_bytes(), json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise CanaryError("resume canary did not retain a readable checkpoint") from error
        receipt = value.get("receipt") if isinstance(value, dict) else None
        if (not isinstance(value, dict)
                or value.get("schema") != "profiling-skill/audited-blocked/v2"
                or value.get("stage") != "controller"
                or not isinstance(value.get("session_id"), str)
                or not isinstance(receipt, dict)
                or not isinstance(receipt.get("handle"), str)):
            raise CanaryError("resume canary checkpoint is not exact controller state")
        return value, raw

    @staticmethod
    def _bound_artifact(binding: object, label: str) -> tuple[dict, bytes]:
        if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
                or not isinstance(binding.get("path"), str)
                or not Path(binding["path"]).is_absolute()
                or not isinstance(binding.get("sha256"), str)):
            raise CanaryError(f"retained {label} binding is invalid")
        path = Path(binding["path"])
        try:
            raw = path.read_bytes()
            value = json.loads(raw)
        except (OSError, json.JSONDecodeError) as error:
            raise CanaryError(f"retained {label} is unavailable") from error
        if hashlib.sha256(raw).hexdigest() != binding["sha256"] or not isinstance(value, dict):
            raise CanaryError(f"retained {label} hash or shape is invalid")
        return value, raw

    def _load_resume_intent(self, path: Path, cell: dict) -> dict | None:
        if not path.is_file():
            return None
        try:
            intent = json.loads(path.read_text())
        except json.JSONDecodeError as error:
            raise CanaryError("retained resume intent is invalid JSON") from error
        if (not isinstance(intent, dict)
                or set(intent) != {"schema", "canary_id", "checkpoint", "marker"}
                or intent.get("schema") !=
                "profiling-skill/audited-repair-canary-resume-intent/v1"
                or intent.get("canary_id") != cell["cell_id"]):
            raise CanaryError("retained resume intent is invalid")
        checkpoint, raw = self._bound_artifact(intent["checkpoint"], "original checkpoint")
        marker, _ = self._bound_artifact(intent["marker"], "observer marker")
        handle = (checkpoint.get("receipt") or {}).get("handle")
        if (checkpoint.get("schema") != "profiling-skill/audited-blocked/v2"
                or checkpoint.get("stage") != "controller"
                or not isinstance(checkpoint.get("session_id"), str)
                or not _valid_observer_marker(marker)
                or marker.get("handle") != handle
                or marker.get("candidate_sha256") != checkpoint.get("candidate_sha256")
                or marker.get("manifest_sha256") != checkpoint.get("manifest_sha256")
                or intent["checkpoint"]["sha256"] != hashlib.sha256(raw).hexdigest()):
            raise CanaryError("resume intent does not bind the original checkpoint")
        intent["checkpoint_document"] = checkpoint
        intent["marker_document"] = marker
        return intent

    def _capture_resume_intent(self, cell: dict, artifact_root: Path,
                               marker_path: Path, intent_path: Path) -> dict:
        checkpoint, checkpoint_raw = self._checkpoint(cell)
        try:
            marker_raw = marker_path.read_bytes()
            marker = json.loads(marker_raw)
        except (OSError, json.JSONDecodeError) as error:
            raise CanaryError("resume canary observer marker is unavailable") from error
        handle = (checkpoint.get("receipt") or {}).get("handle")
        if (not _valid_observer_marker(marker)
                or marker.get("handle") != handle
                or marker.get("candidate_sha256") != checkpoint.get("candidate_sha256")
                or marker.get("manifest_sha256") != checkpoint.get("manifest_sha256")):
            raise CanaryError("observer marker does not bind the controller checkpoint")
        checkpoint_copy = artifact_root / "original-checkpoint.json"
        marker_copy = artifact_root / "original-observer-marker.json"
        _retain_bytes(checkpoint_copy, checkpoint_raw)
        _retain_bytes(marker_copy, marker_raw)
        intent = {
            "schema": "profiling-skill/audited-repair-canary-resume-intent/v1",
            "canary_id": cell["cell_id"],
            "checkpoint": {"path": str(checkpoint_copy.resolve()),
                           "sha256": _sha(checkpoint_copy)},
            "marker": {"path": str(marker_copy.resolve()),
                       "sha256": _sha(marker_copy)},
        }
        _retain(intent_path, intent)
        return self._load_resume_intent(intent_path, cell)

    def _run_one(self, declaration: dict) -> dict:
        cell = self._cell(declaration)
        mode = ("repair" if "candidate-repair" in declaration["required_evidence"]
                else "resume" if "checkpoint-resume" in declaration["required_evidence"]
                else "none")
        launcher = self.launcher_factory(self.config, mode)
        slot = self._slot(cell)
        artifact_root = self.run_root / "artifacts" / cell["cell_id"]
        intent_path = artifact_root / "resume-intent.json"
        resume_path = artifact_root / "resume.json"
        intent = self._load_resume_intent(intent_path, cell) if mode == "resume" else None
        marker = (Path(self.config["run_root"]) / cell["cell_id"] / "state"
                  / "canary-observer-interrupt.json")
        blocked = (Path(self.config["run_root"]) / cell["cell_id"] / "repo"
                   / ".experiment" / "blocked.json")
        if mode == "resume" and intent is None and marker.is_file() and blocked.is_file():
            intent = self._capture_resume_intent(cell, artifact_root, marker, intent_path)

        def original_checkpoint_is_active() -> bool:
            return (intent is not None and blocked.is_file()
                    and _sha(blocked) == intent["checkpoint"]["sha256"])

        if original_checkpoint_is_active():
            try:
                handle = intent["checkpoint_document"]["receipt"]["handle"]
                receipt = launcher.observe(cell, slot, handle)
            except Exception as error:
                raise CanaryError(
                    f"canary {cell['cell_id']} exact observation did not complete: {error}"
                ) from error
        else:
            try:
                receipt = launcher.launch(cell, slot)
            except Exception as error:
                if mode != "resume" or intent is not None:
                    raise CanaryError(
                        f"canary {cell['cell_id']} did not complete: {error}"
                    ) from error
                intent = self._capture_resume_intent(
                    cell, artifact_root, marker, intent_path,
                )
                handle = intent["checkpoint_document"]["receipt"]["handle"]
                try:
                    receipt = launcher.observe(cell, slot, handle)
                except Exception as observe_error:
                    raise CanaryError(
                        f"canary {cell['cell_id']} exact observation did not complete: "
                        f"{observe_error}"
                    ) from observe_error
        if receipt.get("status") != "complete":
            raise CanaryError(f"canary {cell['cell_id']} is not terminal complete")
        verified = launcher.verify(cell)
        resume = None
        if intent is not None:
            checkpoint = intent["checkpoint_document"]
            marker_document = intent["marker_document"]
            if verified.get("session_id") != checkpoint["session_id"]:
                raise CanaryError("resume canary changed the checkpointed agent session")
            matches = []
            for round_receipt in receipt.get("rounds", []):
                history = round_receipt.get("policy", {}).get("operation_history", [])
                for operation in history:
                    if (isinstance(operation, dict)
                            and operation.get("mode") == "observe"
                            and operation.get("terminal") is True
                            and operation.get("status") == "ok"
                            and operation.get("handle") == marker_document["handle"]
                            and operation.get("request_sha256") ==
                            marker_document["controller_request_sha256"]):
                        matches.append((round_receipt.get("round"), operation))
            if len(matches) != 1:
                raise CanaryError("resume canary lacks one exact terminal observe operation")
            resume = {
                "schema": RESUME_SCHEMA, "canary_id": cell["cell_id"],
                "checkpoint": intent["checkpoint"], "marker": intent["marker"],
                "checkpoint_sha256": intent["checkpoint"]["sha256"],
                "experiment": checkpoint["experiment"],
                "candidate_sha256": checkpoint["candidate_sha256"],
                "manifest_sha256": checkpoint["manifest_sha256"],
                "session_id_before": checkpoint["session_id"],
                "session_id_after": verified["session_id"],
                "durable_handle_before": marker_document["handle"],
                "durable_handle_after": marker_document["handle"],
                "observe_round": matches[0][0], "observe_operation": matches[0][1],
            }
        receipt_path, verifier_path = artifact_root / "cell-receipt.json", artifact_root / "verifier.json"
        _retain(receipt_path, receipt)
        _retain(verifier_path, verified)
        resume_binding = None
        if resume is not None:
            _retain(resume_path, resume)
            resume_binding = {"path": str(resume_path.resolve()), "sha256": _sha(resume_path)}
        return {
            "id": cell["cell_id"], "treatment": cell["treatment"],
            "experiment_commit": receipt["commits"][-1],
            "verifier_report": {"path": str(verifier_path.resolve()),
                                "sha256": _sha(verifier_path)},
            "cell_receipt": {"path": str(receipt_path.resolve()),
                             "sha256": _sha(receipt_path)},
            "resume_receipt": resume_binding,
        }

    def run(self, output: Path) -> dict:
        output = Path(output).resolve()
        if output != self.run_root / "canary-results.json":
            raise CanaryError("canary results must be retained under the canary run root")
        if output.is_file():
            production.validate_canary_gate(
                self.config, output, _sha(output), self.runtime_config_sha256,
                tuple(item["treatment"] for item in self.definition["canaries"]),
            )
            result = json.loads(output.read_text())
            state_path = self.run_root / "state" / "run.json"
            try:
                state = json.loads(state_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise CanaryError("retained canary run state is unavailable") from error
            expected_state = {
                "schema", "status", "definition_sha256", "source_revision",
                "runtime_closure_sha256", "runtime_config_sha256",
                "production_campaign_launched",
            }
            if (not isinstance(state, dict) or set(state) != expected_state
                    or state.get("schema") !=
                    "profiling-skill/audited-repair-canary-run/v1"
                    or state.get("status") not in {"running", "complete"}
                    or state.get("definition_sha256") != self.definition_sha256
                    or state.get("runtime_config_sha256") !=
                    self.runtime_config_sha256
                    or state.get("production_campaign_launched") is not False):
                raise CanaryError("retained canary run state is invalid")
            if state["status"] == "running":
                state["status"] = "complete"
                _atomic_json(state_path, state)
            return result
        state = {
            "schema": "profiling-skill/audited-repair-canary-run/v1",
            "status": "running", "definition_sha256": self.definition_sha256,
            "source_revision": self.config["provenance"]["source_revision"],
            "runtime_closure_sha256": self.config["runtime_scripts"]["sha256"],
            "runtime_config_sha256": self.runtime_config_sha256,
            "production_campaign_launched": False,
        }
        _retain(self.run_root / "state" / "run.json", state)
        records = [self._run_one(item) for item in self.definition["canaries"]]
        results = {
            "schema": RESULTS_SCHEMA, "definition_sha256": self.definition_sha256,
            "source_revision": state["source_revision"],
            "runtime_closure_sha256": state["runtime_closure_sha256"],
            "runtime_config_sha256": state["runtime_config_sha256"],
            "results": records,
        }
        pending = output.with_name(f".{output.name}.pending")
        _atomic_json(pending, results)
        production.validate_canary_gate(
            self.config, pending, _sha(pending), self.runtime_config_sha256,
            tuple(item["treatment"] for item in self.definition["canaries"]),
        )
        pending.replace(output)
        state["status"] = "complete"
        running = self.run_root / "state" / "run.json"
        _atomic_json(running, state)
        return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--runtime-config-sha256", required=True)
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--definition-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if file_sha256(args.runtime_config) != args.runtime_config_sha256:
            raise CanaryError("runtime config hash does not match pinned input")
        config = json.loads(args.runtime_config.read_text())
        if not isinstance(config, dict):
            raise CanaryError("runtime config must be a JSON object")
        pool = CplRemoteResourcePool(
            args.admission, args.admission_sha256,
            provider_id=config["admission_provider_id"],
            allowlist_sha256=config["admission_allowlist_sha256"],
            cpl_remote_closure_sha256=config["cpl_remote_closure_sha256"],
        )
        durable = production.DurableAdmissionPool(
            pool, args.run_root.resolve() / "state" / "admission-evidence.json",
            f"repair-canaries-{args.definition_sha256[:12]}",
        )
        runner = CanaryRunner(
            config, args.definition, args.definition_sha256, args.run_root, durable,
        )
        output = args.run_root.resolve() / "canary-results.json"
        result = runner.run(output)
    except (AdmissionError, CanaryError, production.ProductionError,
            InfrastructureFailure,
            OSError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps({"results": str(output), "sha256": _sha(output),
                      "canaries": len(result["results"]),
                      "production_campaign_launched": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
