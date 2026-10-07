#!/usr/bin/env python3
"""Run one audited agent or compose the three-agent acceptance battery."""

from __future__ import annotations

import argparse
import json
import shlex
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, Sequence

from audited_contract import AuditError
from audited_lifecycle import AuditedExperimentRunner, RunResult
from audited_runtime import CodexInvoker, CommandController
from audited_verifier import validate_branch


def run_acceptance_battery(
    run_id: str,
    runners: Mapping[str, AuditedExperimentRunner],
    *,
    verify: Callable[[Path], dict] = validate_branch,
) -> dict[str, RunResult]:
    """Run three isolated agents concurrently and verify each retained branch."""
    if len(runners) != 3:
        raise AuditError("acceptance battery requires exactly three agents")
    repos = [runner.repo.resolve() for runner in runners.values()]
    if len(set(repos)) != 3:
        raise AuditError("acceptance agents must use isolated repositories")
    contracts = {(runner.prompt_bytes, runner.task_bytes) for runner in runners.values()}
    if len(contracts) != 1:
        raise AuditError("acceptance prompt and task must be byte-identical")
    round_counts = {getattr(runner, "round_count", 3) for runner in runners.values()}
    if len(round_counts) != 1:
        raise AuditError("acceptance agents must use one immutable round count")

    def execute(item: tuple[str, AuditedExperimentRunner]) -> tuple[str, RunResult]:
        agent_id, runner = item
        result = runner.run(run_id, agent_id)
        round_count = getattr(runner, "round_count", 3)
        if result.status != "complete" or len(result.commits) != round_count:
            raise AuditError(
                f"acceptance agent {agent_id} did not complete {round_count} experiments"
            )
        checked = verify(runner.repo)
        if checked.get("status") != "valid":
            raise AuditError(f"acceptance agent {agent_id} failed offline verification")
        return agent_id, result

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = dict(pool.map(execute, runners.items()))
    if len({result.session_id for result in results.values()}) != 3:
        raise AuditError("acceptance agents did not retain isolated sessions")
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--controller", required=True,
                        help="target-neutral controller command")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--runtime", choices=("docker", "direct"), default="docker")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--image", default="python:3.10")
    parser.add_argument("--node-runtime", type=Path)
    parser.add_argument("--auth-home", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--rounds", type=int, default=3,
                        help="immutable number of host-directed experiment rounds")
    parser.add_argument("--max-candidate-repairs", type=int, choices=range(3), default=2,
                        help="repairs allowed inside one round after candidate failure")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    invoker = None
    try:
        invoker = CodexInvoker(
            args.repo, codex=args.codex, timeout=args.timeout,
            auth_home=args.auth_home, state_dir=args.state_dir,
            runtime_mode=args.runtime, docker=args.docker, image=args.image,
            node_runtime=args.node_runtime, model=args.model,
            reasoning_effort=args.reasoning_effort, agent_id=args.agent_id,
        )
        controller = CommandController(
            shlex.split(args.controller), args.repo, timeout=args.timeout,
        )
        result = AuditedExperimentRunner(
            args.repo, args.prompt, args.task, invoker, controller,
            round_count=args.rounds,
            max_candidate_repairs=args.max_candidate_repairs,
        ).run(args.run_id, args.agent_id, resume=args.resume)
    except AuditError as error:
        parser.error(str(error))
    finally:
        if invoker is not None:
            invoker.scrub_auth()
    print(json.dumps({
        "status": result.status, "branch": result.branch,
        "session_id": result.session_id, "seed_commit": result.seed_commit,
        "commits": result.commits,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
