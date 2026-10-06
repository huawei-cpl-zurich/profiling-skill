#!/usr/bin/env python3
"""Pinned BZ-A3 resource admission for audited campaigns."""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Callable


TARGETS = ("bz-a3-1", "bz-a3-2")
ADMISSION_SCHEMA = "profiling-skill/bz-a3-admission/v2"
GLOBAL_CPL_REMOTE = Path(".agents/skills/remote-access/scripts/cpl-remote")
REQUIRED_CAPABILITIES = ("run", "observe", "logs", "upload")
MAX_ADMISSION_AGE = timedelta(minutes=5)
MAX_CLOCK_SKEW = timedelta(seconds=30)


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


@dataclass(frozen=True)
class AdmissionSlot:
    target: str
    device: int
    healthy: bool
    idle: bool

    def as_dict(self) -> dict:
        return {"target": self.target, "device": self.device,
                "healthy": self.healthy, "idle": self.idle}


@dataclass(frozen=True)
class AdmissionReceipt:
    receipt_sha256: str
    provider_id: str
    allowlist_sha256: str
    generated_at: str
    expires_at: str
    slots: tuple[AdmissionSlot, ...]

    def as_slots(self) -> list[dict]:
        return [slot.as_dict() for slot in self.slots]


def slot_allowlist_sha256(slots: list[dict] | tuple[AdmissionSlot, ...]) -> str:
    identities = []
    for slot in slots:
        if isinstance(slot, AdmissionSlot):
            target, device = slot.target, slot.device
        elif isinstance(slot, dict):
            target, device = slot.get("target"), slot.get("device")
        else:
            raise AdmissionError("allowlist slots must be objects")
        if target not in TARGETS or type(device) is not int or device < 0:
            raise AdmissionError("allowlist slot identity is invalid")
        identities.append({"target": target, "device": device})
    identities.sort(key=lambda item: (item["target"], item["device"]))
    return hashlib.sha256(json.dumps(
        identities, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def cpl_remote_closure_sha256(launcher: Path) -> str:
    """Digest the executable launcher and the Python implementation it executes."""
    launcher = Path(launcher)
    dependencies = (launcher, launcher.with_name("cpl_remote.py"))
    digest = hashlib.sha256()
    for path in dependencies:
        if not path.is_file():
            raise AdmissionError(f"global cpl-remote dependency is unavailable: {path.name}")
        data = path.read_bytes()
        digest.update(path.name.encode() + b"\0")
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def _timestamp(value: object, label: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise AdmissionError(f"admission {label} must be a UTC RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise AdmissionError(f"admission {label} is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise AdmissionError(f"admission {label} must be UTC")
    return parsed


def load_admission(path: Path, expected_sha256: str, *,
                   now: datetime | None = None) -> AdmissionReceipt:
    """Load and validate one hash-pinned placement-provider snapshot."""
    expected = _require_sha256(expected_sha256, "admission")
    path = Path(path)
    if not path.is_file():
        raise AdmissionError(f"admission placement-provider input is unavailable: {path}")
    try:
        data = path.read_bytes()
    except OSError as error:
        raise AdmissionError(f"admission input cannot be read: {error}") from error
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise AdmissionError("admission hash does not match the pinned input")
    try:
        document = json.loads(data)
    except json.JSONDecodeError as error:
        raise AdmissionError(f"admission input is invalid JSON: {error}") from error
    required = {"schema", "provider", "generated_at", "expires_at", "slots"}
    if (not isinstance(document, dict) or set(document) != required
            or document.get("schema") != ADMISSION_SCHEMA
            or not isinstance(document.get("provider"), dict)
            or set(document["provider"]) != {"id", "allowlist_sha256"}
            or not isinstance(document.get("slots"), list)):
        raise AdmissionError(f"admission input requires exact schema {ADMISSION_SCHEMA}")
    provider_id = document["provider"]["id"]
    if (not isinstance(provider_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", provider_id)):
        raise AdmissionError("admission provider identity is invalid")
    declared_allowlist = _require_sha256(
        document["provider"].get("allowlist_sha256"), "admission allowlist")
    admitted: list[AdmissionSlot] = []
    seen = set()
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
        admitted.append(AdmissionSlot(**slot))
    if slot_allowlist_sha256(tuple(admitted)) != declared_allowlist:
        raise AdmissionError("admission allowlist identity does not match slots")
    generated = _timestamp(document["generated_at"], "generated_at")
    expires = _timestamp(document["expires_at"], "expires_at")
    current = now or datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise AdmissionError("admission validation clock must be timezone-aware")
    current = current.astimezone(UTC)
    if expires <= current:
        raise AdmissionError("admission receipt is expired")
    if generated > current + MAX_CLOCK_SKEW:
        raise AdmissionError("admission receipt is generated in the future")
    if expires <= generated or expires - generated > MAX_ADMISSION_AGE:
        raise AdmissionError("admission receipt validity exceeds the five-minute bound")
    return AdmissionReceipt(
        receipt_sha256=actual, provider_id=provider_id,
        allowlist_sha256=declared_allowlist,
        generated_at=document["generated_at"], expires_at=document["expires_at"],
        slots=tuple(admitted),
    )


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
                 provider_id: str, allowlist_sha256: str,
                 cpl_remote_closure_sha256: str, timeout: int = 120,
                 invoke: Callable = subprocess.run,
                 clock: Callable[[], datetime] = lambda: datetime.now(UTC)):
        self.admission = Path(admission).resolve()
        self.admission_sha256 = _require_sha256(admission_sha256, "admission")
        if (not isinstance(provider_id, str)
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", provider_id)):
            raise AdmissionError("pinned provider identity is invalid")
        self.provider_id = provider_id
        self.allowlist_sha256 = _require_sha256(allowlist_sha256, "pinned allowlist")
        self.cpl_remote_closure_sha256 = _require_sha256(
            cpl_remote_closure_sha256, "global cpl-remote closure")
        if type(timeout) is not int or timeout <= 0:
            raise AdmissionError("remote probe timeout must be a positive integer")
        self.timeout = timeout
        self.invoke = invoke
        self.clock = clock
        self.cpl_remote = _user_home() / GLOBAL_CPL_REMOTE
        self.last_receipt: AdmissionReceipt | None = None
        self._validate_client()

    def _validate_client(self) -> None:
        if (not self.cpl_remote.is_file()
                or not os.access(self.cpl_remote, os.X_OK)):
            raise AdmissionError("global cpl-remote is unavailable or not executable")
        if cpl_remote_closure_sha256(self.cpl_remote) != self.cpl_remote_closure_sha256:
            raise AdmissionError("global cpl-remote closure hash does not match pinned input")

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

    def admit_snapshot(self) -> AdmissionReceipt:
        """Return fresh provider evidence intersected with approved remote health."""
        self.last_receipt = None
        receipt = load_admission(
            self.admission, self.admission_sha256, now=self.clock())
        if receipt.provider_id != self.provider_id:
            raise AdmissionError("admission provider identity does not match pinned input")
        if receipt.allowlist_sha256 != self.allowlist_sha256:
            raise AdmissionError("admission allowlist identity does not match pinned input")
        available = {
            target for target in TARGETS
            if self._probe(target, "capabilities")
            and self._probe(target, "preflight")
        }
        snapshot = replace(
            receipt, slots=tuple(
                slot for slot in receipt.slots if slot.target in available))
        self.last_receipt = snapshot
        return snapshot

    def admit(self) -> list[dict]:
        """Return scheduler-compatible slots and retain their exact receipt."""
        return self.admit_snapshot().as_slots()
