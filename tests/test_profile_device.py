import importlib.util
import fcntl
import os
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("profile_device", Path(__file__).parents[1] / "scripts/profile_device.py")
device = importlib.util.module_from_spec(spec)
spec.loader.exec_module(device)

MAPPING = "NPU ID Chip ID Chip Logic ID Chip Phy-ID Chip Name\n0 0 0 0 Ascend910\n7 1 15 15 Ascend910\n7 2 - - Mcu\n"


def table(*, busy=False, health="OK", usage=0, used=100):
    return (f"| 7 Ascend910 | {health} | 20 |\n"
            f"| 1 15 | bus | {usage} 0 / 0 {used} / 65536 |\n"
            "| NPU Chip | Process id | Process name |\n"
            + ("| 7 1 | 123 | python |\n" if busy else "| No running processes found in NPU 7 |\n"))


def test_inventory_discovers_second_chip_above_seven():
    assert device.discover_devices(MAPPING) == [(0, 0, 0), (7, 1, 15)]
    assert device.inspect_devices(table(), device.discover_devices(MAPPING)) == [(7, 1, 15)]


def test_real_two_chip_card_status_health_and_power_columns():
    inventory = device.discover_devices("3 0 6 6 Ascend910\n3 1 7 7 Ascend910")
    status = ("| 3 Ascend910 | OK | 163.7 47 0 / 0 |\n"
              "| 0 6 | 0000:91:00.0 | 0 0 / 0 3139 / 65536 |\n"
              "| 3 Ascend910 | OK | - 48 0 / 0 |\n"
              "| 1 7 | 0000:93:00.0 | 0 0 / 0 2890 / 65536 |\n"
              "| NPU Chip | Process id | Process name |\n"
              "| No running processes found in NPU 3 |\n")
    assert device.inspect_devices(status, inventory) == [(3, 0, 6), (3, 1, 7)]


@pytest.mark.parametrize("process_rows,expected", [
    ("", []),
    ("| No running processes found in NPU 8 |\n", []),
    ("| 1 1 | 123 | python |\n", [(1, 0, 2)]),
    ("| No running processes found in NPU 1 |\n", [(1, 0, 2), (1, 1, 3)]),
])
def test_process_coverage_is_required_for_each_card(process_rows, expected):
    inventory = device.discover_devices("1 0 2 2 Ascend910\n1 1 3 3 Ascend910\n2 0 4 4 Ascend910")
    status = ("| 1 Ascend910 | OK | 20 |\n| 0 2 | bus | 0 0/0 100/65536 |\n"
              "| 1 Ascend910 | OK | - |\n| 1 3 | bus | 0 0/0 100/65536 |\n"
              "| 2 Ascend910 | OK | 20 |\n| 0 4 | bus | 0 0/0 100/65536 |\n"
              "| NPU Chip | Process id | Process name |\n")
    assert device.inspect_devices(status + process_rows, inventory) == expected


@pytest.mark.parametrize("kwargs", [{"busy": True}, {"health": "Warning"}, {"usage": 1}, {"used": 64000}])
def test_occupancy_health_activity_memory_reject_before_probe(kwargs):
    assert device.inspect_devices(table(**kwargs), device.discover_devices(MAPPING)) == []


def test_busy_to_free_wait_never_probes_busy(tmp_path):
    sleeps, probes = [], []
    states = iter([table(busy=True), table(), table()])
    def query(*args):
        return MAPPING if args else next(states)
    physical, env, lease = device.wait_for_device(
        lambda env: probes.append(env["ASCEND_RT_VISIBLE_DEVICES"]) or True,
        query=query, sleep=sleeps.append, lease_root=tmp_path, notify=lambda value: None)
    assert physical == 15 and env["ASCEND_DEVICE_ID"] == "0"
    assert sleeps == [15] and probes == ["15"]
    lease.close()


def test_shared_lease_and_restart_release(tmp_path):
    lock = open(tmp_path / "device-lease-15.lock", "a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sleeps, probes = [], []
    def sleep(seconds):
        sleeps.append(seconds)
        lock.close()
    physical, _, lease = device.wait_for_device(
        lambda env: probes.append(env) or True,
        query=lambda *args: MAPPING if args else table(), sleep=sleep,
        lease_root=tmp_path, notify=lambda value: None)
    assert physical == 15 and sleeps == [15] and len(probes) == 1
    competing = open(tmp_path / "device-lease-15.lock", "a+")
    with pytest.raises(BlockingIOError):
        fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    lease.close()
    fcntl.flock(competing, fcntl.LOCK_EX | fcntl.LOCK_NB)
    competing.close()


def test_probe_failure_fails_clearly_and_releases_lease(tmp_path):
    probes, sleeps = [], []
    def probe(env):
        probes.append(env)
        return False, "torch import failed"
    with pytest.raises(RuntimeError, match="idle-device runtime probes failed: 15: torch import failed"):
        device.wait_for_device(probe, query=lambda *args: MAPPING if args else table(),
            sleep=sleeps.append, lease_root=tmp_path, notify=lambda value: None)
    assert sleeps == [] and len(probes) == 1
    with open(tmp_path / "device-lease-15.lock", "a+") as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_external_occupancy_race_rechecked_under_lock(tmp_path):
    statuses = iter([table(), table(busy=True), table(), table()])
    probes, sleeps = [], []
    _, _, lease = device.wait_for_device(lambda env: probes.append(env) or True,
        query=lambda *args: MAPPING if args else next(statuses), sleep=sleeps.append,
        lease_root=tmp_path, notify=lambda value: None)
    assert sleeps == [15] and len(probes) == 1
    lease.close()


@pytest.mark.parametrize("mapping,status", [("unrecognized", table()), (MAPPING, "missing process info")])
def test_malformed_observation_fails_clearly_without_probe(tmp_path, mapping, status):
    with pytest.raises(RuntimeError):
        device.wait_for_device(lambda env: pytest.fail("must not allocate"),
            query=lambda *args: mapping if args else status, lease_root=tmp_path)


def test_bounded_run_kills_descendants_without_harming_caller(tmp_path):
    child_pid = tmp_path / "child.pid"
    script = ("import subprocess,sys,time,pathlib\n"
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])\n"
              f"pathlib.Path({str(child_pid)!r}).write_text(str(p.pid))\n"
              "time.sleep(60)\n")
    with pytest.raises(subprocess.TimeoutExpired):
        device.bounded_run([sys.executable, "-c", script], timeout=0.4,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pid = int(child_pid.read_text())
    proc = Path(f"/proc/{pid}/stat")
    # A reparented zombie can await host init reaping but cannot execute NPU work.
    try:
        state = proc.read_text().split()[2]
    except (FileNotFoundError, ProcessLookupError):
        state = None  # The host reaped the dead child between lookup and read.
    assert state in (None, "Z")
    result = device.bounded_run([sys.executable, "-c", "print('alive')"],
        timeout=3, text=True, stdout=subprocess.PIPE)
    assert result.returncode == 0 and result.stdout == "alive\n"
