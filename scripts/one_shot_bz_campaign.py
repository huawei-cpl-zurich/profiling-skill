#!/usr/bin/env python3
"""Connect the one-shot campaign to host-owned BZ-A3 correctness checks."""

from __future__ import annotations

import argparse
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
from bz_a3_diagnostic_client import AdapterTransport, BzA3DiagnosticClient
from diagnostic_campaign import (CommandLauncher, CommandTerminalHook,
                                 DiagnosticCampaign, DiagnosticError, TREATMENTS)


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
    if set(assets) != {"baseline", "case_spec", "runner"}:
        raise DiagnosticError("exactly the three frozen benchmark assets are required")
    snapshot_root = root.parent / f".{root.name}-inputs-{campaign_id}"
    snapshot_root.mkdir(parents=True, exist_ok=False)
    frozen, hashes = {}, {}
    try:
        for name, source_name in assets.items():
            source = Path(source_name)
            if not source.is_file():
                raise DiagnosticError(f"benchmark asset is missing: {source}")
            destination = snapshot_root / name
            shutil.copyfile(source, destination)
            os.chmod(destination, 0o444)
            frozen[name] = str(destination)
            hashes[name] = diagnostic_campaign.sha256_file(destination)
    except BaseException:
        shutil.rmtree(snapshot_root, ignore_errors=True)
        raise
    return frozen, {"root": str(snapshot_root), "sha256": hashes}


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
        snapshot.mkdir(exist_ok=False)
        cls._restore(snapshot, files)
        os.chmod(snapshot, 0o555)
        cls._restore(workspace, files)

    def launch(self, request: dict, timeout_seconds: int) -> dict:
        cell = request["cell_id"]
        workspace = Path(request["workspace"])
        if cell in self.frozen:
            result, files = self.frozen[cell]
            self._freeze_submission(workspace, files)
            return {**result, "submission_replayed": True}
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

    def _check(self, request: dict, timeout_seconds: int) -> dict:
        cell = request["cell_id"]
        try:
            _prefix, wave, treatment = cell.split("-", 2)
        except ValueError as exc:
            raise DiagnosticError(f"invalid cell id {cell!r}") from exc
        workspace = Path(request["workspace"])
        submission = workspace.parent / "frozen-submission"
        attempt = int(workspace.parent.name.removeprefix("attempt-"))
        placement = self.placements[treatment][min(attempt - 1, 1)]
        client_request = {
            "campaign": self.campaign_id, "wave": wave,
            # BZ dispatch receipts are keyed by cell. A fallback placement is
            # a new terminal attempt, while repeating this exact request must
            # observe its retained handle instead of redispatching it.
            "cell": f"{treatment}-attempt-{attempt}",
            **placement, "timeout": min(timeout_seconds, 240),
            "candidate": str(submission / "candidate.py"),
            "candidate_manifest": str(submission / "candidate.manifest.json"),
            "baseline": self.assets["baseline"], "case_spec": self.assets["case_spec"],
            "runner": self.assets["runner"], "cases": list(range(7)),
        }
        # A retained observer interruption is not permission to dispatch on a
        # fallback device. Re-enter the exact original request so the durable
        # client observes its receipt. Only terminal/pre-dispatch failures use
        # the attempt-2 placement.
        retained = self._uncertain.get(cell)
        if retained is not None:
            client_request = {**retained, "observe_timeout":
                              min(timeout_seconds, retained["timeout"])}
        deadline = time.monotonic() + timeout_seconds
        result = self.client.run(client_request)
        uncertain = (result.get("status") == "infrastructure_error"
                     and result.get("failure_type") in {"observer_error", "transport_error"}
                     and bool(result.get("handle")))
        if (retained is not None and result.get("status") == "infrastructure_error"
                and not uncertain):
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                return result
            client_request = {
                **client_request, **placement, "cell": f"{treatment}-attempt-{attempt}",
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
        else:
            self._uncertain.pop(cell, None)
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
    campaign_id = str(uuid.uuid4())
    assets, asset_evidence = freeze_assets(config["assets"], root, campaign_id)
    hook = BzTerminalHook(client, placements, assets, campaign_id)
    campaign = DiagnosticCampaign(manifest, root, FrozenAgentLauncher(launcher), hook,
                                  waves=4, agent_timeout=360, cell_timeout=600,
                                  wave_timeout=600, ledger_metadata={"assets": asset_evidence},
                                  campaign_id=campaign_id)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=("run-all", "run-wave", "acknowledge-curation"),
                        default="run-all")
    parser.add_argument("--wave", type=int)
    parser.add_argument("--curation-receipt", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--agent-command-json")
    parser.add_argument("--remote-command-json", default='["cpl-remote"]')
    parser.add_argument("--adapter-command-json")
    args = parser.parse_args()
    config, manifest = load_json(args.config), load_json(args.manifest)
    placements = load_json(args.placements)
    if args.action == "acknowledge-curation":
        if args.curation_receipt is None:
            parser.error("--curation-receipt is required")
        result = acknowledge_curation(config, manifest, placements, args.run_root,
                                      load_json(args.curation_receipt))
    else:
        if not args.agent_command_json or not args.adapter_command_json or args.state_dir is None:
            parser.error("agent, adapter, and state arguments are required to run waves")
        commands = [json.loads(value) for value in (args.agent_command_json,
                    args.remote_command_json, args.adapter_command_json)]
        if any(not isinstance(value, list) or not value for value in commands):
            parser.error("commands must be non-empty JSON arrays")
        transport = AdapterTransport(commands[1], commands[2])
        client = BzA3DiagnosticClient(transport, args.state_dir)
        if args.action == "run-wave":
            if args.wave is None:
                parser.error("--wave is required")
            result = run_wave(config, manifest, placements, args.run_root,
                              CommandLauncher(commands[0]), client, args.wave)
        else:
            result = run(config, manifest, placements, args.run_root,
                         CommandLauncher(commands[0]), client)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"awaiting_curation", "ready_for_next", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
