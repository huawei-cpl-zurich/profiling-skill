#!/usr/bin/env python3
"""Connect the one-shot campaign to host-owned BZ-A3 correctness checks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import diagnostic_campaign
from bz_a3_diagnostic_client import RemoteTransport, BzA3DiagnosticClient
from diagnostic_campaign import (CommandLauncher, CommandTerminalHook,
                                 DiagnosticCampaign, DiagnosticError, TREATMENTS)
from two_shot_smoke_campaign import (TwoShotSmokeCampaign, manifest_identity,
                                     DEFAULT_SUCCESS_POLICY, validate_matmul_gate)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise DiagnosticError(f"{path} must contain a JSON object")
    return value


def validate_placements(value: dict) -> dict:
    if set(value) != set(TREATMENTS):
        raise DiagnosticError("placements must name exactly the three treatments")
    primary = set()
    for treatment in TREATMENTS:
        choices = value[treatment]
        if not isinstance(choices, list) or len(choices) != 2:
            raise DiagnosticError(f"{treatment} requires primary and fallback placements")
        for choice in choices:
            if (not isinstance(choice, dict) or choice.get("profile") not in {"bz-a3-1", "bz-a3-2"}
                    or isinstance(choice.get("device"), bool)
                    or not isinstance(choice.get("device"), int) or choice["device"] < 0):
                raise DiagnosticError(f"invalid BZ placement for {treatment}")
        if all(choices[0][key] == choices[1][key] for key in ("profile", "device")):
            raise DiagnosticError(f"fallback placement must differ for {treatment}")
        primary.add((choices[0]["profile"], choices[0]["device"]))
    if len(primary) != 3:
        raise DiagnosticError("primary cells require three distinct physical devices")
    return value


def freeze_assets(assets: dict, root: Path, campaign_id: str) -> tuple[dict, dict]:
    core_names = {"baseline", "case_spec", "runner"}
    if not isinstance(assets, dict) or set(assets) - core_names != ({"supplementary"}
            if "supplementary" in assets else set()) or not core_names.issubset(assets):
        raise DiagnosticError("three core assets and optional supplementary assets are required")
    supplementary = assets.get("supplementary", {})
    reserved = {"baseline.py", "cases.jsonl", "runner.py", "candidate.py",
                "candidate.manifest.json", "AGENTS.md"}
    if (not isinstance(supplementary, dict)
            or any(not isinstance(name, str) or not name or name in {".", ".."}
                   or name in reserved or Path(name).name != name
                   or not isinstance(source, str)
                   for name, source in supplementary.items())):
        raise DiagnosticError("supplementary assets require safe filename-to-path entries")
    snapshot_root = root.parent / f".{root.name}-inputs-{campaign_id}"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    frozen, hashes = {}, {}
    try:
        for name in ("baseline", "case_spec", "runner"):
            source_name = assets[name]
            source = Path(source_name)
            if not source.is_file():
                raise DiagnosticError(f"benchmark asset is missing: {source}")
            destination = snapshot_root / name
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o444)
            frozen[name] = str(destination)
            hashes[name] = diagnostic_campaign.sha256_file(destination)
        frozen_supplementary, supplementary_hashes = {}, {}
        for name, source_name in sorted(supplementary.items()):
            source = Path(source_name)
            if not source.is_file():
                raise DiagnosticError(f"benchmark asset is missing: {source}")
            destination = snapshot_root / "supplementary" / name
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o444)
            frozen_supplementary[name] = str(destination)
            supplementary_hashes[name] = diagnostic_campaign.sha256_file(destination)
        if frozen_supplementary:
            frozen["supplementary"] = frozen_supplementary
    except BaseException:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise
    evidence = {"root": str(snapshot_root), "sha256": hashes}
    if supplementary_hashes:
        evidence["supplementary_sha256"] = supplementary_hashes
    return frozen, evidence


class FrozenAgentLauncher:
    """Invoke each logical cell once, then replay its immutable submission on retry."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.frozen: dict[str, tuple[dict, dict[str, bytes]]] = {}

    @staticmethod
    def _restore(workspace: Path, files: dict[str, bytes]) -> None:
        for name, content in files.items():
            with tempfile.NamedTemporaryFile("wb", dir=workspace, delete=False) as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
                temporary = Path(stream.name)
            os.chmod(temporary, 0o444)
            os.replace(temporary, workspace / name)

    @classmethod
    def _freeze_submission(cls, workspace: Path, files: dict[str, bytes]) -> None:
        if not files:
            return
        snapshot = workspace.parent / "frozen-submission"
        if snapshot.exists():
            if (not snapshot.is_dir()
                    or {path.name for path in snapshot.iterdir()} != set(files)
                    or any((snapshot / name).read_bytes() != content
                           for name, content in files.items())):
                raise DiagnosticError("existing frozen submission does not match receipt")
            cls._restore(workspace, files)
            return
        snapshot.mkdir(exist_ok=False)
        cls._restore(snapshot, files)
        os.chmod(snapshot, 0o555)
        cls._restore(workspace, files)

    def launch(self, request: dict, timeout_seconds: int) -> dict:
        cell = request["cell_id"]
        workspace = Path(request["workspace"])
        cell_root = (workspace.parent.parent
                     if workspace.parent.name.startswith("attempt-") else workspace)
        receipt_path = cell_root / "agent-launch.json"
        if cell in self.frozen:
            result, files = self.frozen[cell]
            self._freeze_submission(workspace, files)
            return {**result, "submission_replayed": True}
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("state") != "completed":
                return {"status": "infrastructure_error",
                        "failure_type": "uncertain_agent_launch",
                        "diagnostics": "prior billed launch has no durable terminal receipt"}
            snapshot = Path(receipt["snapshot"])
            expected = receipt.get("candidate_sha256")
            allowed = {"candidate.py", "candidate.manifest.json"}
            if not isinstance(expected, dict) or not set(expected).issubset(allowed):
                raise DiagnosticError("durable launch receipt has invalid candidate hashes")
            files = {name: (snapshot / name).read_bytes() for name in expected}
            if any(hashlib.sha256(files[name]).hexdigest() != digest
                   for name, digest in expected.items()):
                raise DiagnosticError("durable launch snapshot digest mismatch")
            self.frozen[cell] = (receipt["result"], files)
            self._freeze_submission(workspace, files)
            return {**receipt["result"], "submission_replayed": True}
        diagnostic_campaign._atomic_json(receipt_path, {
            "protocol_version": 1, "cell_id": cell, "state": "started",
        })
        try:
            result = self.launcher.launch(request, timeout_seconds)
        except Exception as error:
            result = {"status": "infrastructure_error", "failure_type": "launcher_error",
                      "diagnostics": f"launcher raised {type(error).__name__}: {error}"}
        files = {}
        # Cache the normalized launch outcome before touching agent-owned files.
        # Snapshot failures may make this attempt infrastructure, but must never
        # permit another billed invocation for the same logical cell.
        self.frozen[cell] = (result, files)
        try:
            for name in ("candidate.py", "candidate.manifest.json"):
                path = workspace / name
                if path.is_file():
                    files[name] = path.read_bytes()
            self._freeze_submission(workspace, files)
        except OSError as error:
            files.clear()
            self.frozen[cell] = ({
                "status": "infrastructure_error", "failure_type": "snapshot_error",
                "diagnostics": f"submission snapshot failed: {type(error).__name__}: {error}",
            }, files)
            raise
        snapshot = workspace.parent / "frozen-submission"
        candidate_sha256 = {name: hashlib.sha256(content).hexdigest()
                            for name, content in files.items()}
        result = {**result, "candidate_sha256": candidate_sha256}
        self.frozen[cell] = (result, files)
        diagnostic_campaign._atomic_json(receipt_path, {
            "protocol_version": 1, "cell_id": cell, "state": "completed",
            "result": result, "snapshot": str(snapshot),
            "candidate_sha256": candidate_sha256,
        })
        return result

    def cancel(self) -> None:
        cancel = getattr(self.launcher, "cancel", None)
        if cancel is not None:
            cancel()


class BzTerminalHook:
    """Map campaign terminal requests to deterministic BZ host/device jobs."""

    def __init__(self, client: BzA3DiagnosticClient, placements: dict, assets: dict,
                 campaign_id: str):
        self.client = client
        self.placements = validate_placements(placements)
        self.assets = assets
        self.campaign_id = campaign_id
        self._condition = threading.Condition()
        self._active = 0
        self._cancelled = False
        self._uncertain: dict[str, dict] = {}

    def check(self, request: dict, timeout_seconds: int) -> dict:
        with self._condition:
            if self._cancelled:
                return {"status": "infrastructure_error", "failure_type": "cancelled",
                        "diagnostics": "terminal hook was cancelled"}
            self._active += 1
        try:
            return self._check(request, timeout_seconds)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def resume(self, request: dict, handle: str, timeout_seconds: int) -> dict:
        """Observe an explicitly reconciled handle without dispatching a job."""
        return self.check(
            {**request, "manual_retained_handle": handle}, timeout_seconds)

    def _check(self, request: dict, timeout_seconds: int) -> dict:
        cell = request["cell_id"]
        try:
            _prefix, wave, treatment = cell.split("-", 2)
        except ValueError as exc:
            raise DiagnosticError(f"invalid cell id {cell!r}") from exc
        workspace = Path(request["workspace"])
        submission = workspace.parent / "frozen-submission"
        expected = request.get("candidate_sha256")
        if expected is not None and (
                not isinstance(expected, dict)
                or any(expected.get(name) != diagnostic_campaign.sha256_file(submission / name)
                       for name in ("candidate.py", "candidate.manifest.json"))):
            raise DiagnosticError(f"frozen submission digest mismatch for {cell}")
        attempt = int(workspace.parent.name.removeprefix("attempt-"))
        terminal_attempt = request.get("terminal_attempt", attempt)
        if (isinstance(terminal_attempt, bool) or not isinstance(terminal_attempt, int)
                or terminal_attempt < 1):
            raise DiagnosticError("terminal attempt must be a positive integer")
        placement = self.placements[treatment][min(terminal_attempt - 1, 1)]
        request_timeout = request.get("terminal_request_timeout", timeout_seconds)
        if (isinstance(request_timeout, bool) or not isinstance(request_timeout, int)
                or request_timeout < 1):
            raise DiagnosticError("terminal request timeout must be a positive integer")
        cases = request.get("cases", list(range(7)))
        if (not isinstance(cases, list) or not cases
                or any(isinstance(case, bool) or not isinstance(case, int) or case < 0
                       for case in cases)):
            raise DiagnosticError("terminal cases must be a non-empty list of nonnegative integers")
        benchmark = request.get("benchmark", "matmul")
        if benchmark == "streaming-matmul-add":
            benchmark = "matmul"
        client_request = {
            "campaign": self.campaign_id, "wave": wave,
            # BZ dispatch receipts are keyed by cell. A fallback placement is
            # a new terminal attempt, while repeating this exact request must
            # observe its retained handle instead of redispatching it.
            "cell": f"{treatment}-attempt-{terminal_attempt}",
            "benchmark": benchmark,
            **placement, "timeout": min(request_timeout, 240),
            "candidate": str(submission / "candidate.py"),
            "candidate_manifest": str(submission / "candidate.manifest.json"),
            "baseline": self.assets["baseline"], "case_spec": self.assets["case_spec"],
            "runner": self.assets["runner"], "cases": cases,
        }
        if self.assets.get("supplementary"):
            client_request["supplementary_assets"] = copy.deepcopy(
                self.assets["supplementary"])
        snapshot_hashes = expected if isinstance(expected, dict) else (
            {name: diagnostic_campaign.sha256_file(submission / name)
             for name in ("candidate.py", "candidate.manifest.json")}
            if all((submission / name).is_file()
                   for name in ("candidate.py", "candidate.manifest.json")) else {})
        client_request["candidate_sha256"] = copy.deepcopy(snapshot_hashes)
        manual_handle = request.get("manual_retained_handle")
        if manual_handle is not None:
            if not isinstance(manual_handle, str) or not manual_handle:
                raise DiagnosticError("manual retained handle must be non-empty")
            result = self.client.resume(
                [client_request], manual_handle,
                min(timeout_seconds, client_request["timeout"]))
            if result.get("status") == "infrastructure_error":
                result = {**result, "status": "transport_or_observer_error",
                          "manual_reconciliation_required": True}
            result_cell = str(result.get("cell", ""))
            result_suffix = result_cell.removeprefix(f"{treatment}-attempt-")
            result_attempt = (int(result_suffix) if result_suffix.isdigit()
                              else terminal_attempt)
            return self._map_result(result, result_attempt)
        # A retained observer interruption is not permission to dispatch on a
        # fallback device. Re-enter the exact original request so the durable
        # client observes its receipt. Only terminal/pre-dispatch failures use
        # the attempt-2 placement.
        retained = request.get("retained_terminal_request") or self._uncertain.get(cell)
        if retained is not None:
            immutable = ["campaign", "wave", "benchmark", "baseline", "case_spec",
                         "runner", "cases"]
            if "supplementary_assets" in client_request:
                immutable.append("supplementary_assets")
            retained_candidate = Path(str(retained.get("candidate", "")))
            retained_manifest = Path(str(retained.get("candidate_manifest", "")))
            retained_hashes = retained.get("candidate_sha256")
            prefix = f"{treatment}-attempt-"
            retained_attempt = str(retained.get("cell", "")).removeprefix(prefix)
            retained_placement = (self.placements[treatment][min(int(retained_attempt) - 1, 1)]
                                  if retained_attempt.isdigit() and int(retained_attempt) > 0
                                  else {})
            same_cell = (retained_candidate.name == "candidate.py"
                         and retained_manifest.name == "candidate.manifest.json"
                         and retained_candidate.parent == retained_manifest.parent
                         and retained_candidate.parent.name == "frozen-submission"
                         and retained_candidate.parent.parent.name.startswith("attempt-")
                         and retained_candidate.parent.parent.parent == submission.parent.parent)
            if (not isinstance(retained, dict)
                    or any(retained.get(key) != client_request[key] for key in immutable)
                    or any(retained.get(key) != value
                           for key, value in retained_placement.items())
                    or not retained_placement
                    or not same_cell
                    or not isinstance(retained_hashes, dict)
                    or retained_hashes.get("candidate.py") != diagnostic_campaign.sha256_file(
                        retained_candidate)
                    or retained_hashes.get("candidate.manifest.json") != (
                        diagnostic_campaign.sha256_file(retained_manifest))
                    or isinstance(retained.get("timeout"), bool)
                    or not isinstance(retained.get("timeout"), int)
                    or not 1 <= retained["timeout"] <= 240):
                raise DiagnosticError(f"retained terminal request mismatch for {cell}")
            terminal_attempt = int(retained_attempt)
            client_request = {**retained, "observe_timeout":
                              min(timeout_seconds, retained["timeout"])}
        deadline = time.monotonic() + timeout_seconds
        result = self.client.run(client_request)
        if (result.get("dispatch_uncertain") is True
                and result.get("invocation_timeout") is True
                and not result.get("handle")):
            return {**self._map_result(result, terminal_attempt),
                    "status": "transport_or_observer_error",
                    "manual_reconciliation_required": True}
        uncertain = (result.get("status") == "infrastructure_error"
                     and result.get("failure_type") in {"observer_error", "transport_error"}
                     and bool(result.get("handle")))
        if (retained is not None and result.get("status") == "infrastructure_error"
                and not uncertain):
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                self._uncertain.pop(cell, None)
                return {**result, "terminal_attempt": terminal_attempt}
            terminal_attempt += 1
            placement = self.placements[treatment][min(terminal_attempt - 1, 1)]
            client_request = {
                **client_request, **placement,
                "cell": f"{treatment}-attempt-{terminal_attempt}",
                "timeout": min(remaining, 240),
            }
            client_request.pop("observe_timeout", None)
            result = self.client.run(client_request)
            uncertain = (result.get("status") == "infrastructure_error"
                         and result.get("failure_type") in {"observer_error", "transport_error"}
                         and bool(result.get("handle")))
        if uncertain:
            self._uncertain[cell] = {key: value for key, value in client_request.items()
                                     if key != "observe_timeout"}
            result = {**result, "retained_terminal_request":
                      copy.deepcopy(self._uncertain[cell])}
        else:
            self._uncertain.pop(cell, None)
        return self._map_result(result, terminal_attempt)

    @staticmethod
    def _map_result(result: dict, terminal_attempt: int) -> dict:
        result = {**result, "terminal_attempt": terminal_attempt}
        status = result.get("status")
        if status == "ok":
            return {**result, "passed": True}
        if status == "candidate_timeout":
            return {**result, "status": "timeout"}
        if status in {"compile_error", "runtime_error", "correctness_error"}:
            return result
        return result

    def cancel(self) -> None:
        # BzA3DiagnosticClient owns the durable receipt. Let active observation
        # reach that checkpoint before campaign shutdown abandons daemon cells.
        with self._condition:
            self._cancelled = True
            while self._active:
                self._condition.wait()


def run(config: dict, manifest: dict, placements: dict, root: Path, launcher,
        client: BzA3DiagnosticClient) -> dict:
    required = {"prompt", "prompt_sha256", "assets", "timeouts", "waves"}
    if not required.issubset(config):
        raise DiagnosticError("integration config is incomplete")
    if manifest.get("prompt") != config["prompt"] or manifest.get("prompt_sha256") != config["prompt_sha256"]:
        raise DiagnosticError("manifest does not use the frozen diagnostic prompt")
    timeouts = config["timeouts"]
    if timeouts != {"agent": 360, "cell": 600, "wave": 600}:
        raise DiagnosticError("diagnostic timeouts must remain 360/600/600 seconds")
    if config["waves"] != 4:
        raise DiagnosticError("diagnostic campaign requires exactly four waves")
    identity = _adaptive_inputs(config, manifest, placements)
    existing = root / "ledger.json"
    if existing.is_file():
        ledger = load_json(existing)
        campaign_id = ledger.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise DiagnosticError("fixed ledger has no campaign identity")
        asset_evidence, assets = ledger.get("assets"), _retained_assets(ledger)
    else:
        campaign_id = str(uuid.uuid4())
        assets, asset_evidence = freeze_assets(config["assets"], root, campaign_id)
    hook = BzTerminalHook(client, placements, assets, campaign_id)
    campaign = DiagnosticCampaign(manifest, root, FrozenAgentLauncher(launcher), hook,
                                  waves=4, agent_timeout=360, cell_timeout=600,
                                  wave_timeout=600, ledger_metadata={"assets": asset_evidence},
                                  campaign_id=campaign_id,
                                  campaign_identity={"config_sha256": identity})
    campaign.ledger_metadata["fixed_campaign_identity"] = campaign._adaptive_identity()
    if existing.is_file():
        return campaign.resume_fixed()
    try:
        ledger = campaign.run()
    except BaseException:
        if campaign.ledger_path.is_file():
            ledger = json.loads(campaign.ledger_path.read_text())
            ledger["assets"] = asset_evidence
            diagnostic_campaign._atomic_json(campaign.ledger_path, ledger)
        raise
    ledger["assets"] = asset_evidence
    diagnostic_campaign._atomic_json(campaign.ledger_path, ledger)
    return ledger


def _legacy_supplementary_closure_matches(assets: object, evidence: object) -> bool:
    if not isinstance(assets, dict) or not isinstance(evidence, dict):
        return False
    requested = assets.get("supplementary", {})
    retained = evidence.get("supplementary_sha256", {})
    if not isinstance(requested, dict) or not isinstance(retained, dict):
        return False
    if set(requested) != set(retained):
        return False
    try:
        return all(isinstance(source, str) and Path(source).is_file()
                   and diagnostic_campaign.sha256_file(Path(source)) == retained[name]
                   for name, source in requested.items())
    except OSError:
        return False


def run_smoke(config: dict, manifest: dict, placements: dict, root: Path, launcher,
              client: BzA3DiagnosticClient, matmul_gate: Path | None = None,
              *, resume: bool = False) -> dict:
    """Run the fixed two-wave smoke battery for one configured benchmark."""
    required = {"prompt", "prompt_sha256", "assets", "benchmark", "cases"}
    if not required.issubset(config):
        raise DiagnosticError("smoke integration config is incomplete")
    validate_placements(placements)
    benchmark = config["benchmark"]
    success_policy = config.get("success_policy", DEFAULT_SUCCESS_POLICY)
    normalized_config = {**config, "success_policy": success_policy}
    config_identity = hashlib.sha256(json.dumps(
        normalized_config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if benchmark == "bsa":
        if matmul_gate is None:
            raise DiagnosticError("BSA smoke requires --matmul-gate")
        validate_matmul_gate(
            matmul_gate, prompt_sha256=manifest.get("prompt_sha256"),
            manifest_identity=manifest_identity(manifest),
            success_policy=success_policy)
    elif benchmark not in {"matmul", "gdn"}:
        raise DiagnosticError("smoke benchmark must be matmul, bsa, or gdn")
    if (manifest.get("prompt") != config["prompt"]
            or manifest.get("prompt_sha256") != config["prompt_sha256"]):
        raise DiagnosticError("manifest does not use the frozen smoke prompt")
    if resume:
        ledger = load_json(root / "ledger.json")
        if ledger.get("config_identity") not in {None, config_identity}:
            raise DiagnosticError("smoke resume config changed")
        if (ledger.get("config_identity") is None
                and not _legacy_supplementary_closure_matches(
                    config.get("assets"), ledger.get("assets"))):
            raise DiagnosticError(
                "identity-less smoke resume supplementary asset closure is unproven")
        campaign_id = ledger.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise DiagnosticError("smoke ledger has no campaign identity")
        evidence, assets = ledger.get("assets"), _retained_assets(ledger)
    else:
        campaign_id = str(uuid.uuid4())
        assets, evidence = freeze_assets(config["assets"], root, campaign_id)
    hook = BzTerminalHook(client, placements, assets, campaign_id)
    campaign = TwoShotSmokeCampaign(
        manifest, root, launcher, hook, benchmark=benchmark, cases=config["cases"],
        resume=resume, ledger_metadata={"assets": evidence,
                                        "config_identity": config_identity},
        success_policy=success_policy)
    campaign.campaign_id = campaign_id
    try:
        ledger = campaign.run()
    except BaseException:
        if not (root / "ledger.json").exists():
            shutil.rmtree(evidence["root"], ignore_errors=True)
        raise
    ledger["assets"] = evidence
    ledger["config_identity"] = config_identity
    diagnostic_campaign._atomic_json(root / "ledger.json", ledger)
    return ledger


def _adaptive_config_sha256(config: dict, placements: dict) -> str:
    frozen = {key: value for key, value in config.items()
              if key not in {"prompt", "prompt_sha256"}}
    value = {"config": frozen, "placements": placements}
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _adaptive_inputs(config: dict, manifest: dict, placements: dict) -> str:
    required = {"prompt", "prompt_sha256", "assets", "timeouts", "waves"}
    if not required.issubset(config):
        raise DiagnosticError("integration config is incomplete")
    if (manifest.get("prompt") != config["prompt"]
            or manifest.get("prompt_sha256") != config["prompt_sha256"]):
        raise DiagnosticError("manifest does not use the frozen diagnostic prompt")
    if config["timeouts"] != {"agent": 360, "cell": 600, "wave": 600}:
        raise DiagnosticError("adaptive campaign requires 360/600/600 timeouts")
    if config["waves"] != 4:
        raise DiagnosticError("adaptive campaign requires exactly four waves")
    validate_placements(placements)
    return _adaptive_config_sha256(config, placements)


def _retained_assets(ledger: dict) -> dict:
    evidence = ledger.get("assets")
    if not isinstance(evidence, dict) or set(evidence.get("sha256", {})) != {
            "baseline", "case_spec", "runner"}:
        raise DiagnosticError("adaptive ledger has no frozen asset evidence")
    root = Path(evidence.get("root", ""))
    assets = {name: str(root / name) for name in evidence["sha256"]}
    if any(not Path(path).is_file() for path in assets.values()):
        raise DiagnosticError("frozen adaptive asset is missing")
    if any(diagnostic_campaign.sha256_file(Path(assets[name])) != digest
           for name, digest in evidence["sha256"].items()):
        raise DiagnosticError("frozen adaptive asset changed")
    supplementary_hashes = evidence.get("supplementary_sha256", {})
    if not isinstance(supplementary_hashes, dict):
        raise DiagnosticError("frozen supplementary asset evidence is invalid")
    supplementary = {name: str(root / "supplementary" / name)
                     for name in supplementary_hashes}
    if any(not Path(path).is_file() for path in supplementary.values()):
        raise DiagnosticError("frozen supplementary asset is missing")
    if any(diagnostic_campaign.sha256_file(Path(supplementary[name])) != digest
           for name, digest in supplementary_hashes.items()):
        raise DiagnosticError("frozen supplementary asset changed")
    if supplementary:
        assets["supplementary"] = supplementary
    return assets


def run_wave(config: dict, manifest: dict, placements: dict, root: Path, launcher,
             client: BzA3DiagnosticClient, wave: int) -> dict:
    """Run exactly one adaptive BZ wave, preserving the fixed-run API."""
    identity = _adaptive_inputs(config, manifest, placements)
    created_snapshot = wave == 1 and not (root / "ledger.json").exists()
    if created_snapshot:
        diagnostic_campaign.validate_manifest(manifest)
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise DiagnosticError(f"diagnostic output root is not fresh: {root}")
        campaign_id = str(uuid.uuid4())
        assets, evidence = freeze_assets(config["assets"], root, campaign_id)
    else:
        ledger = load_json(root / "ledger.json")
        campaign_id = ledger.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id:
            raise DiagnosticError("adaptive ledger has no campaign identity")
        evidence, assets = ledger.get("assets"), _retained_assets(ledger)
    hook = BzTerminalHook(client, placements, assets, campaign_id)
    campaign = DiagnosticCampaign(
        manifest, root, FrozenAgentLauncher(launcher), hook, waves=4,
        agent_timeout=360, cell_timeout=600, wave_timeout=600,
        ledger_metadata={"assets": evidence}, campaign_id=campaign_id,
        campaign_identity={"config_sha256": identity},
    )
    try:
        return campaign.run_wave(wave)
    except BaseException:
        if created_snapshot and not campaign.ledger_path.exists():
            shutil.rmtree(evidence["root"], ignore_errors=True)
        raise


def acknowledge_curation(config: dict, manifest: dict, placements: dict,
                          root: Path, receipt: dict) -> dict:
    """Validate and persist the out-of-workspace curation receipt."""
    identity = _adaptive_inputs(config, manifest, placements)
    campaign = DiagnosticCampaign(
        manifest, root, CommandLauncher(["false"]), CommandTerminalHook(["false"]),
        campaign_identity={"config_sha256": identity},
    )
    return campaign.acknowledge_curation(receipt)


def reconcile_terminal(config: dict, manifest: dict, placements: dict,
                       root: Path, cell_id: str, agent_attempt: int,
                       terminal_attempt: int, *, handle: str | None = None,
                       result: dict | None = None) -> dict:
    """Apply an explicit operator reconciliation to one uncertain receipt."""
    identity = _adaptive_inputs(config, manifest, placements)
    ledger = load_json(root / "ledger.json")
    campaign_id = ledger.get("campaign_id")
    if not isinstance(campaign_id, str) or not campaign_id:
        raise DiagnosticError("campaign ledger has no valid campaign id")
    campaign = DiagnosticCampaign(
        manifest, root, CommandLauncher(["false"]), CommandTerminalHook(["false"]),
        campaign_id=campaign_id, campaign_identity={"config_sha256": identity},
    )
    validated = (campaign._load_adaptive_ledger()
                 if ledger.get("adaptive") is True
                 else campaign._load_fixed_ledger())
    if validated.get("campaign_id") != campaign_id:
        raise DiagnosticError("campaign ledger identity mismatch")
    if (validated.get("status") != "reschedule_pending"
            or cell_id not in validated.get("reschedule", [])):
        raise DiagnosticError("cell is not awaiting terminal reconciliation")
    if handle is not None:
        parts = cell_id.split("-", 2)
        if (len(parts) != 3 or parts[2] not in placements
                or terminal_attempt < 1):
            raise DiagnosticError("invalid reconciliation cell or terminal attempt")
        profile = placements[parts[2]][min(terminal_attempt - 1, 1)]["profile"]
        if not handle.startswith("remote:" + profile + ":job:"):
            raise DiagnosticError("reconciliation handle does not match terminal placement")
    receipt = campaign.reconcile_terminal(
        cell_id, agent_attempt, terminal_attempt, handle=handle, result=result)
    return {"status": "reconciled", "receipt": receipt}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("run-all", "run-wave", "run-smoke",
                                              "acknowledge-curation", "reconcile-terminal"),
                        default="run-all")
    parser.add_argument("--wave", type=int)
    parser.add_argument("--curation-receipt", type=Path)
    parser.add_argument("--cell-id")
    parser.add_argument("--agent-attempt", type=int)
    parser.add_argument("--terminal-attempt", type=int)
    parser.add_argument("--matmul-gate", type=Path)
    parser.add_argument("--resume-smoke", action="store_true")
    reconciliation = parser.add_mutually_exclusive_group()
    reconciliation.add_argument("--terminal-handle")
    reconciliation.add_argument("--terminal-result", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--agent-command-json")
    parser.add_argument("--remote-command-json", default='["cpl-remote"]')
    parser.add_argument("--runtime-activate")
    args = parser.parse_args()
    if args.resume_smoke and args.action != "run-smoke":
        parser.error("--resume-smoke requires --action run-smoke")
    config, manifest = load_json(args.config), load_json(args.manifest)
    placements = load_json(args.placements)
    if args.action == "acknowledge-curation":
        if args.curation_receipt is None:
            parser.error("--curation-receipt is required")
        result = acknowledge_curation(config, manifest, placements, args.run_root,
                                      load_json(args.curation_receipt))
    elif args.action == "reconcile-terminal":
        if (not args.cell_id or args.agent_attempt is None
                or args.terminal_attempt is None
                or (args.terminal_handle is None) == (args.terminal_result is None)):
            parser.error("cell, attempts, and exactly one terminal handle/result are required")
        result = reconcile_terminal(
            config, manifest, placements, args.run_root, args.cell_id,
            args.agent_attempt, args.terminal_attempt,
            handle=args.terminal_handle,
            result=(load_json(args.terminal_result) if args.terminal_result else None))
    else:
        if (not args.agent_command_json or args.state_dir is None
                or not args.runtime_activate):
            parser.error("agent, state, and runtime activation arguments are required to run waves")
        commands = [json.loads(value) for value in (args.agent_command_json,
                    args.remote_command_json)]
        if any(not isinstance(value, list) or not value for value in commands):
            parser.error("commands must be non-empty JSON arrays")
        transport = RemoteTransport(commands[1], args.runtime_activate)
        client = BzA3DiagnosticClient(transport, args.state_dir)
        if args.action == "run-wave":
            if args.wave is None:
                parser.error("--wave is required")
            result = run_wave(config, manifest, placements, args.run_root,
                              CommandLauncher(commands[0]), client, args.wave)
        elif args.action == "run-smoke":
            result = run_smoke(config, manifest, placements, args.run_root,
                               CommandLauncher(commands[0]), client, args.matmul_gate,
                               resume=args.resume_smoke)
        else:
            result = run(config, manifest, placements, args.run_root,
                         CommandLauncher(commands[0]), client)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {
        "awaiting_curation", "ready_for_next", "complete", "reconciled",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
