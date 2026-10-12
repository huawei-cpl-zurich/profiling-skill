#!/usr/bin/env python3
"""Translate the simple runner's JSON phases to isolated Codex CLI turns."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


class AdapterError(RuntimeError):
    pass


SCHEMAS = {
    "prepare": {
        "type": "object", "additionalProperties": False,
        "required": ["candidate", "commands", "hypothesis", "retrieved_knowledge"],
        "properties": {
            "candidate": {"type": "string"},
            "commands": {"type": "array", "items": {"type": "string"}},
            "hypothesis": {"type": "string"},
            "retrieved_knowledge": {"type": "array", "items": {"type": "string"}},
        },
    },
    "tune": {
        "type": "object", "additionalProperties": False,
        "required": ["configurations"],
        "properties": {
            "configurations": {
                "type": "array", "maxItems": 128,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["id", "parameters", "hypothesis"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "parameters": {"type": "object"},
                        "hypothesis": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
    },
    "analyze": {
        "type": "object", "additionalProperties": False,
        "required": ["reasoning", "postmortem"],
        "properties": {"reasoning": {"type": "string"}, "postmortem": {"type": "string"}},
    },
}


def private_codex_home() -> Path:
    raw = os.environ.get("CODEX_HOME")
    if not raw:
        raise AdapterError("CODEX_HOME must identify the cell-private Codex home")
    home = Path(raw).resolve()
    auth = home / "auth.json"
    if auth.is_symlink() or not auth.is_file() or stat.S_IMODE(auth.stat().st_mode) & 0o077:
        raise AdapterError("cell-private auth.json is missing or has unsafe permissions")
    if (home / "config.toml").exists():
        raise AdapterError("cell-private Codex home must not contain global configuration")
    return home


def validate_request(value: object) -> dict:
    if not isinstance(value, dict) or value.get("phase") not in SCHEMAS:
        raise AdapterError("request must select prepare, tune, or analyze")
    for field in ("prompt", "repo"):
        if not isinstance(value.get(field), str) or not value[field]:
            raise AdapterError(f"request requires {field}")
    repo = Path(value["repo"]).resolve()
    if not repo.is_dir() or not (repo / ".git").exists():
        raise AdapterError("request repo must be an initialized Git repository")
    if value["phase"] in {"tune", "analyze"} and not isinstance(value.get("session_id"), str):
        raise AdapterError(f"{value['phase']} requires an exact session_id")
    return value


def codex_prompt(request: dict) -> str:
    candidate_contract = ""
    if request.get("candidate_contract") is not None:
        candidate_contract = (
            " The candidate source must define and launch exactly "
            "candidate_contract.entrypoint. Preserve candidate_contract.kernel_name as the "
            "host profiling selector, and implement the complete operator as the declared "
            "fully fused single logical launch. Apply these requirements to initial prepare "
            "and every same-round repair; the exact task-specific names are data in "
            "candidate_contract, not treatment guidance."
        )
    contract = (
        "\n\nFollow the experiment prompt and the JSON phase request below. "
        "Return only the response object required by that phase. Do not access "
        "hardware, remote transports, global skills, or files outside this Git repository."
        + candidate_contract + "\n"
    )
    return request["prompt"] + contract + "REQUEST_JSON\n" + json.dumps(request, sort_keys=True)


def invoke(args: argparse.Namespace, request: dict) -> dict:
    home = private_codex_home()
    with tempfile.TemporaryDirectory(prefix="codex-json-adapter-") as temporary:
        schema = Path(temporary) / "schema.json"
        output = Path(temporary) / "last-message.json"
        schema.write_text(json.dumps(SCHEMAS[request["phase"]], sort_keys=True))
        common = ["--json", "--ignore-user-config", "--ignore-rules",
                  "--model", args.model,
                  "-c", f'model_reasoning_effort="{args.reasoning_effort}"',
                  "--output-schema", str(schema), "-o", str(output)]
        if request["phase"] == "prepare" and not request.get("session_id"):
            command = [args.codex, "exec", *common, "--sandbox", "workspace-write",
                       "-C", str(Path(request["repo"]).resolve()), "-"]
        else:
            session = request.get("session_id")
            if not isinstance(session, str) or not session:
                raise AdapterError("resumed phase requires session_id")
            command = [args.codex, "exec", "resume", *common, session, "-"]
        try:
            result = subprocess.run(command, input=codex_prompt(request), text=True,
                                    capture_output=True, check=False)
        finally:
            generated_config = home / "config.toml"
            if generated_config.is_file() and not generated_config.is_symlink():
                generated_config.unlink()
        if result.returncode:
            raise AdapterError(result.stderr.strip() or "codex exec failed")
        thread = None
        for line in result.stdout.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started" and isinstance(event.get("thread_id"), str):
                thread = event["thread_id"]
        if not output.is_file():
            raise AdapterError("codex did not produce a final response")
        try:
            response = json.loads(output.read_text())
        except json.JSONDecodeError as exc:
            raise AdapterError("codex final response is not JSON") from exc
        if not isinstance(response, dict):
            raise AdapterError("codex final response is not an object")
        required = set(SCHEMAS[request["phase"]]["required"])
        if set(response) != required:
            raise AdapterError("codex final response is incomplete")
        if request["phase"] == "prepare":
            if (not isinstance(response["candidate"], str)
                    or not isinstance(response["commands"], list)
                    or not all(isinstance(item, str) for item in response["commands"])
                    or not isinstance(response["hypothesis"], str)
                    or not isinstance(response["retrieved_knowledge"], list)
                    or not all(isinstance(item, str) for item in response["retrieved_knowledge"])):
                raise AdapterError("codex prepare response has invalid field types")
            if not thread and not request.get("session_id"):
                raise AdapterError("codex did not return a durable thread id")
            response["session_id"] = thread or request["session_id"]
        elif request["phase"] == "tune":
            configurations = response["configurations"]
            if not isinstance(configurations, list) or len(configurations) > 128:
                raise AdapterError("codex tune response has invalid configurations")
            for configuration in configurations:
                if (not isinstance(configuration, dict)
                        or set(configuration) != {"id", "parameters", "hypothesis"}
                        or not isinstance(configuration["id"], str)
                        or not configuration["id"]
                        or not isinstance(configuration["parameters"], dict)
                        or not isinstance(configuration["hypothesis"], str)
                        or not configuration["hypothesis"]):
                    raise AdapterError("codex tune response has invalid configurations")
        elif (not isinstance(response["reasoning"], str)
              or not isinstance(response["postmortem"], str)):
            raise AdapterError("codex analyze response has invalid field types")
        return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", required=True)
    parser.add_argument("--reasoning-effort", default="low")
    args = parser.parse_args()
    try:
        request = validate_request(json.load(sys.stdin))
        print(json.dumps(invoke(args, request), sort_keys=True))
    except (AdapterError, json.JSONDecodeError, OSError) as exc:
        print(f"codex-json-adapter: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
