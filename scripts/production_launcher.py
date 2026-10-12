#!/usr/bin/env python3
"""Shared process-isolation support for profiling behavioral agents.

Experiment campaign execution belongs to ``simple_experiment_runner.py``. It
owns the JSON prepare/analyze protocol, terminal gate, billing, retries, and
Git evidence. The behavioral-gate launcher still needs the small set of
Bubblewrap/Codex helpers below, but it must not expose a second campaign
controller or hardware interface.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


class LaunchError(RuntimeError):
    """The isolated agent process cannot be launched safely."""


class ProductionLauncher:
    """Provide the shared, controller-free Codex isolation primitives.

    This class is intentionally not a campaign runner. The only production
    experiment lifecycle is the simple runner's JSON phase contract. The
    behavioral profiling gate subclasses this helper for its independent
    blinded-evaluation workflow.
    """

    def __init__(
        self,
        *,
        codex: str = "codex",
        bwrap: str = "bwrap",
        auth_home: Path | None = None,
        resolv_conf: Path = Path("/etc/resolv.conf"),
    ):
        self.codex = Path(shutil.which(codex) or codex).resolve()
        self.bwrap = shutil.which(bwrap) or bwrap
        self.auth_home = (
            auth_home or Path(os.environ.get("CODEX_HOME", Path.home() / ".codex"))
        ).resolve()
        self.resolv_conf = resolv_conf
        if not Path(self.bwrap).exists() or not self.codex.exists():
            raise LaunchError("Bubblewrap and Codex executables are required")
        if not (self.auth_home / "auth.json").is_file():
            raise LaunchError("authenticated Codex state is unavailable")

    def _runtime_mount(self) -> Path:
        """Return the versioned Node prefix containing the Codex executable."""
        for parent in self.codex.parents:
            if (parent / "bin" / "node").is_file() and (
                parent / "lib" / "node_modules"
            ).is_dir():
                return parent
        return self.codex.parent

    def _resolver_mounts(self) -> list[str]:
        """Expose only the systemd-resolved file needed by /etc/resolv.conf."""
        if not self.resolv_conf.is_symlink():
            if not self.resolv_conf.is_file():
                raise LaunchError("host resolver configuration is unavailable")
            return []
        link = Path(os.readlink(self.resolv_conf))
        destination = link if link.is_absolute() else Path("/etc") / link
        destination = Path(os.path.normpath(destination))
        allowed = Path("/run/systemd/resolve")
        if not destination.is_relative_to(allowed):
            raise LaunchError(f"unsupported resolver target: {destination}")
        try:
            source = self.resolv_conf.resolve(strict=True)
        except (FileNotFoundError, RuntimeError) as error:
            raise LaunchError("host resolver target is unavailable") from error
        if not source.is_file():
            raise LaunchError(f"host resolver target is unavailable: {source}")
        command = ["--dir", "/run", "--dir", "/run/systemd", "--dir", str(allowed)]
        command += ["--ro-bind", str(source), str(destination)]
        return command

    @staticmethod
    def _session_id(output: str) -> str:
        """Extract the durable Codex thread id from JSONL process output."""
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started" and isinstance(
                event.get("thread_id"), str
            ):
                return event["thread_id"]
        raise LaunchError("Codex did not report a persistent thread id")
