#!/usr/bin/env python3
"""Connect the one-shot campaign to host-owned BZ-A3 correctness checks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path

import diagnostic_campaign
from bz_a3_diagnostic_client import AdapterTransport, BzA3DiagnosticClient
from diagnostic_campaign import CommandLauncher, DiagnosticCampaign, DiagnosticError, TREATMENTS


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
    for name, source_name in assets.items():
        source = Path(source_name)
        if not source.is_file():
            raise DiagnosticError(f"benchmark asset is missing: {source}")
        destination = snapshot_root / name
        shutil.copyfile(source, destination)
        os.chmod(destination, 0o444)
        frozen[name] = str(destination)
        hashes[name] = diagnostic_campaign.sha256_file(destination)
    return frozen, {"root": str(snapshot_root), "sha256": hashes}


class FrozenAgentLauncher:
    """Invoke each logical cell once, then replay its immutable submission on retry."""

    def __init__(self, launcher):
        self.launcher = launcher
        self.frozen: dict[str, tuple[dict, dict[str, bytes]]] = {}

    def launch(self, request: dict, timeout_seconds: int) -> dict:
        cell = request["cell_id"]
        workspace = Path(request["workspace"])
        if cell in self.frozen:
            result, files = self.frozen[cell]
            for name, content in files.items():
                (workspace / name).write_bytes(content)
            return {**result, "submission_replayed": True}
        try:
            result = self.launcher.launch(request, timeout_seconds)
        except Exception as error:
            result = {"status": "infrastructure_error", "failure_type": "launcher_error",
                      "diagnostics": f"launcher raised {type(error).__name__}: {error}"}
        files = {}
        for name in ("candidate.py", "candidate.manifest.json"):
            path = workspace / name
            if path.is_file():
                files[name] = path.read_bytes()
        self.frozen[cell] = (result, files)
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
        attempt = int(workspace.parent.name.removeprefix("attempt-"))
        placement = self.placements[treatment][min(attempt - 1, 1)]
        client_request = {
            "campaign": self.campaign_id, "wave": wave,
            # BZ dispatch receipts are keyed by cell. A fallback placement is
            # a new terminal attempt, while repeating this exact request must
            # observe its retained handle instead of redispatching it.
            "cell": f"{treatment}-attempt-{attempt}",
            **placement, "timeout": min(timeout_seconds, 240),
            "candidate": str(workspace / "candidate.py"),
            "candidate_manifest": str(workspace / "candidate.manifest.json"),
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
        if (retained is not None and result.get("status") == "infrastructure_error"
                and result.get("failure_type") != "observer_error"):
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                return result
            client_request = {
                **client_request, **placement, "cell": f"{treatment}-attempt-{attempt}",
                "timeout": min(remaining, 240),
            }
            client_request.pop("observe_timeout", None)
            result = self.client.run(client_request)
        if (result.get("status") == "infrastructure_error"
                and result.get("failure_type") == "observer_error"
                and result.get("handle")):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--placements", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--agent-command-json", required=True)
    parser.add_argument("--remote-command-json", default='["cpl-remote"]')
    parser.add_argument("--adapter-command-json", required=True)
    args = parser.parse_args()
    commands = [json.loads(value) for value in (args.agent_command_json,
                args.remote_command_json, args.adapter_command_json)]
    if any(not isinstance(value, list) or not value for value in commands):
        parser.error("commands must be non-empty JSON arrays")
    transport = AdapterTransport(commands[1], commands[2])
    result = run(load_json(args.config), load_json(args.manifest),
                 load_json(args.placements), args.run_root, CommandLauncher(commands[0]),
                 BzA3DiagnosticClient(transport, args.state_dir))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "complete" and not result["reschedule"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
