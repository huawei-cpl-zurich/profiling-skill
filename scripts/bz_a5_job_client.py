#!/usr/bin/env python3
"""Production JSON job client for self-contained BZ-A5 Triton workloads."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
from pathlib import Path

from bz_a3_job_client import (
    BzA3JobClient,
    CommandResult,
    GlobalCplRemoteTransport,
    JobError,
    _sha,
    _user_home,
)


GLOBAL_CPL_REMOTE = Path(".agents/skills/remote-access/scripts/cpl-remote")
TARGET = "bz-a5"
RUNTIME = "cann91"


class A5GlobalCplRemoteTransport(GlobalCplRemoteTransport):
    """Global transport constrained to BZ-A5 and its named CANN 9.1 runtime."""

    def __init__(self, expected_sha256: str, remote_root: str, invoke=None,
                 *, executable: Path | None = None):
        kwargs = {"targets": {TARGET}}
        if invoke is not None:
            kwargs["invoke"] = invoke
        if executable is not None:
            kwargs["executable"] = executable
        super().__init__(expected_sha256, remote_root, **kwargs)

    def dispatch(self, target: str, device: int, runtime: str, operation: str,
                 script: str, timeout: int) -> str:
        if target != TARGET or runtime != RUNTIME:
            raise JobError("request_error", "A5 dispatch requires bz-a5/cann91")
        return super().dispatch(target, device, runtime, operation, script, timeout)


class BzA5JobClient(BzA3JobClient):
    """A5 specialization that embeds its payload in one `run --file` request."""

    PRODUCT = "a5"
    RUNTIME = RUNTIME
    TARGETS = {TARGET}

    def _execution_provenance(self, runtime: str) -> dict[str, str]:
        return {**super()._execution_provenance(runtime), "product": self.PRODUCT,
                "target": TARGET, "staging": "self-contained-run-file"}

    def _prepare_remote_script(self, target: str, archive: Path, request_sha: str,
                               run_root: str, workload_timeout: int, job: dict,
                               transport_timeout: int) -> str:
        del target, request_sha, transport_timeout
        encoded = base64.b64encode(archive.read_bytes()).decode("ascii")
        qroot = shlex.quote(run_root)
        extraction = f'''payload={qroot}/payload.tar
mkdir -p {qroot}
python - "$payload" <<'PY'
import base64
import pathlib
import sys
pathlib.Path(sys.argv[1]).write_bytes(base64.b64decode({encoded!r}, validate=True))
PY
'''
        return extraction + self._remote_script(
            f"{run_root}/payload.tar", _sha(archive), run_root,
            workload_timeout, job,
        )

    @staticmethod
    def _complete(response: CommandResult, handle: str | None, identity: dict,
                  placement: dict, request_sha: str, job: dict,
                  remote_root: str, placements_sha256: str,
                  execution_provenance: dict[str, str]) -> dict:
        result = BzA3JobClient._complete(
            response, handle, identity, placement, request_sha, job,
            remote_root, placements_sha256, execution_provenance,
        )
        result["product"] = "a5"
        return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--placements-json", type=Path, required=True)
    parser.add_argument("--cpl-remote-sha256", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--canary-interrupt-after-dispatch-once", type=Path)
    args = parser.parse_args()
    try:
        placements = json.loads(args.placements_json.read_text())
        here = Path(__file__).resolve().parent
        client = BzA5JobClient(
            A5GlobalCplRemoteTransport(args.cpl_remote_sha256, args.remote_root),
            args.state_dir, placements,
            runner=here / "a3_benchmark_runner.py", profiler=here / "profile_a3.py",
            batch_profiler=here / "batch_profile_a3.py", remote_root=args.remote_root,
            interrupt_after_dispatch=args.canary_interrupt_after_dispatch_once,
        )
        result = client.run(
            json.load(os.sys.stdin), args.timeout,
            operation_mode=os.environ.get("PROFILING_SKILL_CONTROLLER_MODE", "submit"),
            controller_request_sha256=os.environ.get(
                "PROFILING_SKILL_CONTROLLER_REQUEST_SHA256"),
        )
    except (OSError, ValueError, json.JSONDecodeError, JobError) as exc:
        failure = exc.failure_type if isinstance(exc, JobError) else "request_error"
        result = {"status": "infrastructure_error", "failure_type": failure,
                  "diagnostics": str(exc), "handle": None, "product": "a5"}
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
