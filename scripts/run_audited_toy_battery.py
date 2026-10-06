#!/usr/bin/env python3
"""Launch the real three-agent, three-experiment, no-NPU acceptance battery."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

from audited_contract import AuditError
from audited_experiment import run_acceptance_battery
from audited_lifecycle import AuditedExperimentRunner
from audited_runtime import CodexInvoker

TASK = """# Toy task

`candidate.py` starts with `VALUE = 0`. In each experiment increment `VALUE`
by exactly one and run a local Python command that prints it. The three
increments are independent material changes. No external source, remote
target, device, or hardware is needed.
"""


def initialize_repo(repo: Path) -> None:
    """Create one isolated immutable-baseline repository."""
    if repo.exists():
        raise AuditError(f"toy repository must not already exist: {repo}")
    repo.mkdir(parents=True)
    commands = (
        ("git", "init", "-q", "-b", "main"),
        ("git", "config", "user.name", "Audited Host"),
        ("git", "config", "user.email", "host@experiment.invalid"),
    )
    for command in commands:
        subprocess.run(command, cwd=repo, check=True)
    (repo / "candidate.py").write_text("VALUE = 0\n")
    (repo / "candidate.manifest.json").write_text(
        json.dumps({"candidate": "candidate.py"}, sort_keys=True) + "\n"
    )
    subprocess.run(("git", "add", "."), cwd=repo, check=True)
    subprocess.run(("git", "commit", "-qm", "toy baseline"), cwd=repo, check=True)


def local_receipt(agent_id: str, number: int, candidate_hash: str,
                  manifest_hash: str) -> dict:
    """Return deterministic policy evidence without touching hardware."""
    handle = f"toy:{agent_id}:{number}:{candidate_hash[:12]}"
    value = float(number)
    return {
        "status": "ok", "handle": handle, "candidate_sha256": candidate_hash,
        "manifest_sha256": manifest_hash, "device": "local-no-npu",
        "samples_us": [value, value, value], "median_us": value,
        "policy": {
            "schema": "profiling-skill/controller-policy/v1",
            "selected_device": "local-no-npu",
            "admission_controls": [{
                "device": "local-no-npu", "status": "pass", "healthy": True,
                "idle": True, "warmed": True,
            }],
            "submission_candidate_sha256": candidate_hash,
            "submitted_handles": [handle], "observed_handles": [handle],
            "infra_retries": 0, "retry_budget": 1,
            "quarantined_devices": [], "quarantine_controls": {},
            "sample_count": 3, "variability_threshold": 0.25,
            "variability_ratio": 0.0, "confirmation_count": 0,
            "confirmation": None, "post_control": "stable",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path,
                        help="new directory for isolated repositories and state")
    parser.add_argument("--prompt", type=Path, default=(
        Path(__file__).parents[1] / "prompts" / "audited-three-experiment.md"
    ))
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--runtime", choices=("docker", "direct"), default="docker")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", default="python:3.10")
    parser.add_argument("--node-runtime", type=Path)
    parser.add_argument("--auth-home", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args(argv)
    if args.root.exists():
        parser.error(f"battery root must not already exist: {args.root}")
    args.root.mkdir(parents=True)
    task = args.root / "TASK.md"
    task.write_text(TASK)
    runners: dict[str, AuditedExperimentRunner] = {}
    invokers: list[CodexInvoker] = []
    try:
        for agent_id in ("agent-1", "agent-2", "agent-3"):
            repo = args.root / agent_id
            initialize_repo(repo)
            invoker = CodexInvoker(
                repo, codex=args.codex, timeout=args.timeout,
                auth_home=args.auth_home,
                state_dir=args.root / ".codex-state" / agent_id,
                runtime_mode=args.runtime, docker=args.docker, image=args.image,
                node_runtime=args.node_runtime, model=args.model,
                reasoning_effort=args.reasoning_effort, agent_id=agent_id,
            )
            invokers.append(invoker)
            controller = lambda number, candidate, manifest, agent_id=agent_id: (
                local_receipt(agent_id, number, candidate, manifest)
            )
            runners[agent_id] = AuditedExperimentRunner(
                repo, args.prompt, task, invoker, controller,
            )
        results = run_acceptance_battery("toy-acceptance", runners)
    except AuditError as error:
        parser.error(str(error))
    finally:
        for invoker in invokers:
            invoker.scrub_auth()
    print(json.dumps({agent: {
        "branch": result.branch, "session_id": result.session_id,
        "seed_commit": result.seed_commit, "commits": result.commits,
    } for agent, result in results.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
