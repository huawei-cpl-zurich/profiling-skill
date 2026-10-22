"""Bounded per-chip idle selection; dead aggregate table entries cannot stall it."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import os
from pathlib import Path
import re
import subprocess
import time

from profile_device import bounded_run, discover_devices


def eligible(health, usages, processes):
    def number(name):
        match = re.search(r'^\s*' + re.escape(name) + r'\s*:\s*(\d+)\s*$', usages, re.M)
        return int(match.group(1)) if match else None
    return bool(re.search(r'^\s*Health Status\s*:\s*OK\s*$', health, re.M)
        and all(number(name) == 0 for name in
                ('Aicore Usage Rate(%)', 'Aivector Usage Rate(%)', 'NPU Utilization(%)'))
        and number('HBM Capacity(MB)') is not None
        and number('HBM Usage Rate(%)') is not None
        and number('HBM Capacity(MB)') * (100 - number('HBM Usage Rate(%)')) / 100 >= 4096
        and processes.strip() == 'No process in device.')


def acquire(probe, remaining, *, query=None, sleep=time.sleep, lease_root=None):
    def native_query(*args):
        result = bounded_run(['npu-smi', 'info', *args], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=min(8., remaining()))
        if result.returncode:
            raise RuntimeError('per-device query unavailable')
        return result.stdout
    query = query or native_query
    root = Path(lease_root) if lease_root else Path.home() / '.cpl-device-leases'
    root.mkdir(parents=True, exist_ok=True)
    def inspect(device):
        card, chip, _ = device
        try:
            args = ('-i', str(card), '-c', str(chip))
            return eligible(query('-t', 'health', *args), query('-t', 'usages', *args),
                            query('-t', 'proc-mem', *args))
        except (OSError, RuntimeError, subprocess.TimeoutExpired):
            return False
    while remaining() > 0:
        inventory = discover_devices(query('-m'))
        with ThreadPoolExecutor(max_workers=min(8, len(inventory))) as pool:
            free = [device for device, ok in zip(inventory, pool.map(inspect, inventory)) if ok]
        for device in free:
            lease = (root / ('device-lease-%d.lock' % device[2])).open('a+')
            try:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lease.close()
                continue
            try:
                if not inspect(device):
                    lease.close()
                    continue
                env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=str(device[2]), ASCEND_DEVICE_ID='0')
                print('DEVICE_PROBING=' + str(device[2]), flush=True)
                result = probe(env)
                ok, reason = result if isinstance(result, tuple) else (result, 'runtime probe failed')
                if ok:
                    print('DEVICE_SELECTED=' + str(device[2]), flush=True)
                    return device[2], env, lease
                raise RuntimeError('idle-device runtime probe unavailable')
            except BaseException:
                lease.close()
                raise
        print('DEVICE_WAIT eligible=' + str(len(free)), flush=True)
        sleep(min(5., remaining()))
    raise TimeoutError('role deadline expired waiting for an idle healthy device')
