#!/usr/bin/env python3
"""Connect the one-shot campaign to host-owned BZ-A3 correctness checks."""

from __future__ import annotations

import argparse
import json
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
        primary.add((choices[0]["profile"], choices[0]["device"]))
    if len(primary) != 3:
        raise DiagnosticError("primary cells require three distinct physical devices")
    return value


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
        result = self.launcher.launch(request, timeout_seconds)
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

    def check(self, request: dict, timeout_seconds: int) -> dict:
        cell = request["cell_id"]
        try:
            _prefix, wave, treatment = cell.split("-", 2)
        except ValueError as exc:
            raise DiagnosticError(f"invalid cell id {cell!r}") from exc
        workspace = Path(request["workspace"])
        attempt = int(workspace.parent.name.removeprefix("attempt-"))
        placement = self.placements[treatment][min(attempt - 1, 1)]
        result = self.client.run({
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
        })
        status = result.get("status")
        if status == "ok":
            return {**result, "passed": True}
        if status == "candidate_timeout":
            return {**result, "status": "timeout"}
        if status in {"compile_error", "runtime_error", "correctness_error"}:
            return result
        return result


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
    hook = BzTerminalHook(client, placements, config["assets"], root.name)
    return DiagnosticCampaign(manifest, root, FrozenAgentLauncher(launcher), hook,
                              waves=4, agent_timeout=360, cell_timeout=600,
                              wave_timeout=600).run()


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
