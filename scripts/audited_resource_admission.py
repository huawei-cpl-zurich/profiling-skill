#!/usr/bin/env python3
"""Pinned BZ-A3 resource admission for audited campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import subprocess
from pathlib import Path
from typing import Callable


TARGETS = ("bz-a3-1", "bz-a3-2")
ADMISSION_SCHEMA = "profiling-skill/bz-a3-admission/v1"
GLOBAL_CPL_REMOTE = Path(".agents/skills/remote-access/scripts/cpl-remote")
REQUIRED_CAPABILITIES = ("run", "observe", "logs", "upload")


class AdmissionError(RuntimeError):
    """Pinned admission input or remote discovery is invalid."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise AdmissionError(f"{label} digest must be 64 lowercase hexadecimal characters")
    return value


def _user_home() -> Path:
    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def load_admission(path: Path, expected_sha256: str) -> list[dict]:
    """Load and validate one hash-pinned placement-provider snapshot."""
    expected = _require_sha256(expected_sha256, "admission")
    path = Path(path)
    if not path.is_file():
        raise AdmissionError(f"admission placement-provider input is unavailable: {path}")
    if file_sha256(path) != expected:
        raise AdmissionError("admission hash does not match the pinned input")
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AdmissionError(f"admission input is invalid JSON: {error}") from error
    if (not isinstance(document, dict) or set(document) != {"schema", "slots"}
            or document.get("schema") != ADMISSION_SCHEMA
            or not isinstance(document.get("slots"), list)):
        raise AdmissionError(f"admission input requires exact schema {ADMISSION_SCHEMA}")
    admitted, seen = [], set()
    for slot in document["slots"]:
        if (not isinstance(slot, dict)
                or set(slot) != {"target", "device", "healthy", "idle"}):
            raise AdmissionError("admission slot has an invalid shape")
        identity = (slot["target"], slot["device"])
        if (slot["target"] not in TARGETS
                or type(slot["device"]) is not int or slot["device"] < 0
                or type(slot["healthy"]) is not bool
                or type(slot["idle"]) is not bool
                or identity in seen):
            raise AdmissionError("admission slot identity is invalid or duplicated")
        seen.add(identity)
        admitted.append(dict(slot))
    return admitted


def _json_receipt(stdout: str) -> dict:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise AdmissionError("global cpl-remote returned no JSON receipt")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as error:
        raise AdmissionError("global cpl-remote trailing JSON receipt is invalid") from error
    if not isinstance(payload, dict):
        raise AdmissionError("global cpl-remote JSON receipt must be an object")
    for line in lines[:-1]:
        try:
            earlier = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(earlier, dict):
            raise AdmissionError("global cpl-remote JSON receipt is ambiguous")
    return payload


class CplRemoteResourcePool:
    """Intersect a pinned slot snapshot with approved global remote health."""

    def __init__(self, admission: Path, admission_sha256: str, *,
                 cpl_remote_sha256: str, timeout: int = 120,
                 invoke: Callable = subprocess.run):
        self.admission = Path(admission).resolve()
        self.admission_sha256 = _require_sha256(admission_sha256, "admission")
        self.cpl_remote_sha256 = _require_sha256(
            cpl_remote_sha256, "global cpl-remote")
        if type(timeout) is not int or timeout <= 0:
            raise AdmissionError("remote probe timeout must be a positive integer")
        self.timeout = timeout
        self.invoke = invoke
        self.cpl_remote = _user_home() / GLOBAL_CPL_REMOTE
        self._validate_client()

    def _validate_client(self) -> None:
        if (not self.cpl_remote.is_file()
                or not os.access(self.cpl_remote, os.X_OK)):
            raise AdmissionError("global cpl-remote is unavailable or not executable")
        if file_sha256(self.cpl_remote) != self.cpl_remote_sha256:
            raise AdmissionError("global cpl-remote hash does not match pinned input")

    def _probe(self, target: str, operation: str) -> bool:
        self._validate_client()
        try:
            result = self.invoke(
                [str(self.cpl_remote), "--json", operation, target],
                text=True, capture_output=True, timeout=self.timeout, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        if result.returncode:
            return False
        payload = _json_receipt(result.stdout)
        if payload.get("target") != target:
            raise AdmissionError("global cpl-remote receipt identity mismatch")
        if operation == "capabilities":
            capabilities = payload.get("capabilities")
            return (payload.get("state") == "available"
                    and isinstance(capabilities, dict)
                    and all(capabilities.get(name) is True
                            for name in REQUIRED_CAPABILITIES))
        return payload.get("state") == "completed"

    def admit(self) -> list[dict]:
        """Return snapshot slots whose target passes both approved probes."""
        slots = load_admission(self.admission, self.admission_sha256)
        available = {
            target for target in TARGETS
            if self._probe(target, "capabilities")
            and self._probe(target, "preflight")
        }
        return [dict(slot) for slot in slots if slot["target"] in available]
