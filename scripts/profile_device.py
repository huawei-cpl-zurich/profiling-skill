"""Small, dependency-free idle-device selector for retained profiling jobs.

The returned open lease must stay alive for the complete capture. Inventory
comes from npu-smi mapping, not card count: a card may contain several chips.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def bounded_run(argv, *, timeout, **kwargs):
    """Bound a subprocess and descendants without killing the retained job."""
    with subprocess.Popen(argv, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def discover_devices(mapping):
    devices = []
    for line in mapping.splitlines():
        fields = line.split()
        if len(fields) >= 5 and all(v.isdigit() for v in fields[:4]):
            card, chip, logical, physical = map(int, fields[:4])
            devices.append((card, chip, physical))
    if not devices or len({d[2] for d in devices}) != len(devices):
        raise RuntimeError("npu-smi mapping has no unique device inventory")
    return devices


def inspect_devices(table, inventory, *, min_free_mb=4096):
    """Parse the native status/process table; missing observations fail closed."""
    status, occupied = {}, set()
    pending = None
    processes = False
    for line in table.splitlines():
        if "Process id" in line and "Process name" in line:
            processes = True
        fields = [part.strip() for part in line.split("|")[1:-1]]
        if not fields:
            continue
        first = fields[0].split()
        if processes:
            if len(first) == 2 and all(v.isdigit() for v in first) and len(fields) > 1 and fields[1].isdigit():
                occupied.add(tuple(map(int, first)))
            continue
        if len(first) >= 2 and first[0].isdigit() and "Ascend" in first[1]:
            pending = (int(first[0]), fields[1] == "OK")
            continue
        if pending and len(first) == 2 and all(v.isdigit() for v in first) and len(fields) >= 3:
            chip, physical = map(int, first)
            values = fields[2].split()
            memory = re.search(r"(\d+)\s*/\s*(\d+)\s*$", fields[2])
            if values and values[0].isdigit() and memory:
                used, capacity = map(int, memory.groups())
                status[(pending[0], chip, physical)] = (
                    pending[1] and int(values[0]) == 0 and capacity - used >= min_free_mb
                )
            pending = None
    if not processes:
        raise RuntimeError("npu-smi status omitted process inventory")
    return [d for d in inventory if status.get(d, False) and d[:2] not in occupied]


def smi(*arguments):
    result = subprocess.run(["npu-smi", "info", *arguments], text=True,
                            capture_output=True, timeout=20)
    if result.returncode:
        raise RuntimeError("npu-smi query failed: " + result.stderr[-500:])
    return result.stdout


def wait_for_device(probe, *, poll_seconds=15, lease_root=None,
                    query=smi, sleep=time.sleep, notify=None):
    """Wait without consuming capture attempts, then probe only an idle chip.

    ``probe(env)`` returns bool or (bool, diagnostic). Query/parse failures are infrastructure
    errors, not evidence that every device is busy. OS flock releases on exit.
    """
    root = Path(lease_root) if lease_root else Path.home() / ".cpl-device-leases"
    root.mkdir(parents=True, exist_ok=True)
    notify = notify or (lambda value: print("DEVICE_WAIT=" + json.dumps(value), flush=True))
    while True:
        inventory = discover_devices(query("-m"))
        eligible = inspect_devices(query(), inventory)
        failures = []
        for card, chip, physical in eligible:
            lease = open(root / ("device-lease-%d.lock" % physical), "a+")
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            try:
                # Recheck after locking: another workload may have started since
                # the first status snapshot, even outside this lease protocol.
                if (card, chip, physical) not in inspect_devices(query(), inventory):
                    lease.close()
                    continue
                env = os.environ.copy()
                env["ASCEND_RT_VISIBLE_DEVICES"] = str(physical)
                env["ASCEND_DEVICE_ID"] = "0"
                result = probe(env)
                ok, detail = result if isinstance(result, tuple) else (result, "probe returned false")
                if ok:
                    notify({"state": "selected", "physical_id": physical, "logical_id": 0})
                    return physical, env, lease
                failures.append("%d: %s" % (physical, str(detail)[-1000:]))
            except BaseException:
                lease.close()
                raise
            lease.close()
        if failures:
            raise RuntimeError("idle-device runtime probes failed: " + "; ".join(failures))
        notify({"state": "waiting", "devices": len(inventory), "eligible": len(eligible)})
        sleep(poll_seconds)
