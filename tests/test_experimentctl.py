from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).parents[1]
CLI = ROOT / "scripts" / "experimentctl.py"


BACKEND = r'''#!/usr/bin/env python3
import json, os, sys
request = json.load(sys.stdin)
with open(os.environ["REQUEST_LOG"], "a") as stream:
    stream.write(json.dumps(request) + "\n")
mode = os.environ.get("BACKEND_MODE", "ok")
if mode == "bad-json":
    print("not json")
    raise SystemExit()
if mode == "compile":
    print(json.dumps({"status":"compile_error", "device":request["device"],
                      "handle":"gz-a3:compile", "diagnostics":"error: invalid operands\nsource.py:17"}))
    raise SystemExit(7)
if mode == "infra":
    print(json.dumps({"status":"infrastructure_error", "device":request["device"],
                      "handle":"gz-a3:infra", "diagnostics":"DCMI unavailable"}))
    raise SystemExit(9)
case = request.get("case", 0)
iteration = request.get("iteration", 0)
latency = (case + 1) * 10 + iteration
print(json.dumps({"status":"ok", "device":request["device"],
                  "handle":f"gz-a3:{request['action']}-{case}-{iteration}",
                  "latency_us":latency, "passed":True}))
'''


def setup(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    backend = tmp_path / "backend.py"
    backend.write_text(BACKEND)
    backend.chmod(0o755)
    config = tmp_path / "campaign.json"
    config.write_text(json.dumps({"cells": {"gdn-skill": {
        "benchmark": "gdn", "device": 1,
        "development_cases": [4, 2], "all_cases": [0, 1, 2],
        "backend": {"command": [str(backend)], "timeout_seconds": 10},
    }}}))
    env = {**os.environ, "REQUEST_LOG": str(tmp_path / "requests.jsonl")}
    return config, env


def run(tmp_path: Path, *arguments: str, mode: str = "ok"):
    config, env = setup(tmp_path)
    env["BACKEND_MODE"] = mode
    process = subprocess.run(
        ["python3", str(CLI), "--config", str(config), "--cell", "gdn-skill", *arguments],
        text=True, capture_output=True, env=env,
    )
    return process, json.loads(process.stdout), tmp_path / "requests.jsonl"


def requests(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_rank_is_warmed_repeated_sorted_and_device_bound(tmp_path: Path):
    process, result, log = run(tmp_path, "rank", "--benchmark", "gdn", "--warmups", "2", "--repeats", "3")
    assert process.returncode == 0
    assert [row["case"] for row in result["ranking"]] == [2, 1, 0]
    assert result["ranking"][0]["median_us"] == 31
    calls = requests(log)
    assert len(calls) == 15
    assert all(call["device"] == 1 and call["cell"] == "gdn-skill" for call in calls)
    assert [call["phase"] for call in calls[:5]] == ["warmup", "warmup", "sample", "sample", "sample"]


def test_check_uses_configured_scope_and_forwards_round(tmp_path: Path):
    process, result, log = run(tmp_path, "check", "--scope", "full", "--round", "2")
    assert process.returncode == 0
    assert result["cases"] == [0, 1, 2]
    assert requests(log) == [{
        "protocol_version": 1, "action": "check", "cell": "gdn-skill",
        "benchmark": "gdn", "device": 1, "cases": [0, 1, 2], "scope": "full", "round": 2,
    }]


def test_profile_medians_and_geometric_mean(tmp_path: Path):
    process, result, log = run(tmp_path, "profile", "--repeats", "3", "--round", "3")
    assert process.returncode == 0
    assert [row["median_us"] for row in result["cases"]] == [51, 31]
    assert round(result["geomean_us"], 6) == round((51 * 31) ** 0.5, 6)
    assert len(result["handles"]) == 6
    assert all(call["round"] == 3 for call in requests(log))


def test_compile_failure_counts_and_preserves_diagnostics(tmp_path: Path):
    process, result, _ = run(tmp_path, "check", mode="compile")
    assert process.returncode == 2
    assert result["status"] == "candidate_error"
    assert result["failure_type"] == "compile_error"
    assert result["handles"] == ["gz-a3:compile"]
    assert "invalid operands\nsource.py:17" in result["diagnostics"]


def test_infrastructure_and_invalid_protocol_are_discardable(tmp_path: Path):
    process, result, _ = run(tmp_path, "profile", mode="infra")
    assert process.returncode == 3
    assert result["status"] == "infrastructure_error"
    assert result["handles"] == ["gz-a3:infra"]
    other = tmp_path / "other"
    other.mkdir()
    process, result, _ = run(other, "check", mode="bad-json")
    assert process.returncode == 3
    assert "invalid backend response" in result["diagnostics"]


def test_bad_cell_or_benchmark_is_configuration_error(tmp_path: Path):
    config, env = setup(tmp_path)
    process = subprocess.run(
        ["python3", str(CLI), "--config", str(config), "--cell", "gdn-skill", "rank", "--benchmark", "bsa"],
        text=True, capture_output=True, env=env,
    )
    result = json.loads(process.stdout)
    assert process.returncode == 4
    assert result["status"] == "config_error"
    assert not (tmp_path / "requests.jsonl").exists()
