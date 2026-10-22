import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from model_device import acquire, eligible

HEALTH = 'Health Status : OK\n'
USAGES = ('HBM Capacity(MB) : 65536\nHBM Usage Rate(%) : 4\n'
          'Aicore Usage Rate(%) : 0\nAivector Usage Rate(%) : 0\nNPU Utilization(%) : 0\n')
EMPTY = '\tNo process in device.\n'


def test_device_health_occupancy_and_missing_fields_fail_closed():
    assert eligible(HEALTH, USAGES, EMPTY)
    assert not eligible('Health Status : Critical\n', USAGES, EMPTY)
    assert not eligible(HEALTH, USAGES.replace('Aicore Usage Rate(%) : 0', 'Aicore Usage Rate(%) : 1'), EMPTY)
    assert not eligible(HEALTH, USAGES, 'Process 123 python')
    assert not eligible(HEALTH, USAGES.replace('HBM Capacity(MB) : 65536', ''), EMPTY)


def test_native_engine_usage_without_optional_aggregate_utilization():
    native = USAGES.replace('NPU Utilization(%) : 0\n', '')
    assert eligible(HEALTH, native, EMPTY)
    for missing in ('Aicore Usage Rate(%) : 0\n', 'Aivector Usage Rate(%) : 0\n'):
        assert not eligible(HEALTH, native.replace(missing, ''), EMPTY)
    for value in ('1', 'unknown'):
        assert not eligible(HEALTH, native + f'NPU Utilization(%) : {value}\n', EMPTY)


def test_native_three_column_selector_preserves_runtime_mapping(tmp_path):
    def query(*args):
        if args == ('-m',):
            return 'NPU ID Chip ID Chip Logic ID Chip Name\n7 0 7 Ascend 910B4-1\n7 1 - Mcu\n'
        return {'health': HEALTH,
                'usages': USAGES.replace('NPU Utilization(%) : 0\n', ''),
                'proc-mem': EMPTY}[args[1]]
    physical, env, lease = acquire(lambda env: env['ASCEND_RT_VISIBLE_DEVICES'] == '7',
                                   lambda: 100, query=query, lease_root=tmp_path)
    try:
        assert physical == 7 and env['ASCEND_DEVICE_ID'] == '0'
    finally:
        lease.close()


def test_targeted_selector_excludes_bad_chip_and_rechecks_under_lease(tmp_path):
    calls = []
    def query(*args):
        calls.append(args)
        if args == ('-m',):
            return '0 0 0 0 Ascend\n1 0 1 1 Ascend\n'
        kind, card = args[1], args[3]
        if card == '0':
            raise RuntimeError('unresponsive unhealthy chip')
        return {'health': HEALTH, 'usages': USAGES, 'proc-mem': EMPTY}[kind]
    physical, env, lease = acquire(lambda env: True, lambda: 100,
                                   query=query, lease_root=tmp_path)
    try:
        assert physical == 1 and env['ASCEND_RT_VISIBLE_DEVICES'] == '1'
        assert env['ASCEND_DEVICE_ID'] == '0'
        assert calls.count(('-t', 'proc-mem', '-i', '1', '-c', '0')) == 2
        assert () not in calls
    finally:
        lease.close()


def test_selector_waits_until_deadline_without_guessing_device(tmp_path):
    budget = [5]
    def query(*args):
        if args == ('-m',):
            return '0 0 0 0 Ascend\n'
        return {'health': HEALTH, 'usages': USAGES, 'proc-mem': 'Busy process'}[args[1]]
    def sleep(seconds):
        budget[0] -= seconds
    with pytest.raises(TimeoutError):
        acquire(lambda env: pytest.fail('must not probe busy device'), lambda: budget[0],
                query=query, sleep=sleep, lease_root=tmp_path)
