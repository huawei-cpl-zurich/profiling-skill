import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import a3_wait_timeline as study


def success(*args, **kwargs):
    return subprocess.CompletedProcess(args[0], 0, '{"correctness":true}\n', '')


def clean(*args):
    return {'clean_acquisition_gate_passed': True}


def test_manifest_is_device_independent():
    p = subprocess.run([sys.executable, str(ROOT / 'scripts/a3_wait_timeline.py'), '--manifest'],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    groups = json.loads(p.stdout)
    assert len(groups) == 6
    assert sum(map(len, groups.values())) == 12
    for group in groups.values():
        assert {c['placement'] for c in group} == {'overlap', 'early-wait'}
        assert {c['rounds'] for c in group} == {32}


def test_interleaves_all_samples_and_keeps_noisy_values(tmp_path):
    calls = []
    def capture(cmd, folder, env, metric, selector):
        calls.append((cmd[2], metric))
        return dict(metric=metric, basic_rows=[{'Duration(us)': str(1 + len(calls))}])
    result = study.run_pair(study.pairs()['mte2-vector'], '/kernel', tmp_path / 'pair', {}, 'soc',
                            simulate=clean, capture=capture, execute=success)
    assert result['status'] == 'complete'
    assert len(calls) == 22
    assert calls[:6] == [('0', 'BasicInfo'), ('1', 'BasicInfo'), ('1', 'BasicInfo'),
                          ('0', 'BasicInfo'), ('0', 'BasicInfo'), ('1', 'BasicInfo')]
    assert all(len(c['timing']['samples_us']) == 5 and c['timing']['low_confidence']
               for c in result['cases'].values())


def test_failed_simulation_still_completes_hardware_matrix(tmp_path):
    calls = []
    def simulate(*args):
        calls.append('simulation')
        return {'clean_acquisition_gate_passed': False, 'status': 'profiler_failed'}
    def capture(cmd, folder, env, metric, selector):
        calls.append(metric)
        return dict(metric=metric, basic_rows=[{'Duration(us)': '2'}])
    result = study.run_pair(study.pairs()['mode0/balanced'], '/kernel', tmp_path / 'pair', {}, 'soc',
                            include_simulation=True, simulate=simulate, capture=capture, execute=success)
    assert result['status'] == 'complete'
    assert len(result['cases']) == 2
    assert calls.count('BasicInfo') == 10
    assert calls.count('PipeUtilization') == calls.count('Memory') == 6
    assert calls[-2:] == ['simulation', 'simulation']
    assert all(c['simulation']['availability'] == 'unavailable' for c in result['cases'].values())


def test_correctness_failure_stops_before_simulation(tmp_path):
    def bad(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, '', 'incorrect')
    def simulate(*args):
        raise AssertionError('simulation before numerical check')
    result = study.run_pair(study.pairs()['vector-mte3'], '/kernel', tmp_path / 'pair', {}, 'soc',
                            simulate=simulate, execute=bad)
    assert result['status'] == 'failed'
    assert 'failed' in result['failure']['message']


def test_canary_checks_both_variants_and_hardware_discovery(tmp_path):
    calls = []
    def capture(cmd, folder, env, metric, selector):
        calls.append((cmd[2], metric))
        return dict(metric=metric, basic_rows=[{'Duration(us)': '2'}])
    result = study.run_pair(study.pairs()['mode2/imbalanced'], '/kernel', tmp_path / 'pair', {}, 'soc',
                            canary=True, capture=capture, execute=success)
    assert result['status'] == 'canary_passed'
    assert len(result['cases']) == 2
    assert calls == [('0', 'BasicInfo'), ('1', 'BasicInfo')]
    assert all(c['simulation']['status'] == 'not_requested' for c in result['cases'].values())


def test_explicit_memory_rejection_stops_only_memory_group(tmp_path):
    calls = []
    def capture(cmd, folder, env, metric, selector):
        calls.append(metric)
        return dict(metric=metric, availability='unsupported' if metric == 'Memory' else 'available',
                    basic_rows=[{'Duration(us)': '2'}])
    result = study.run_pair(study.pairs()['mte2-vector'], '/kernel', tmp_path / 'pair', {}, 'soc',
                            simulate=clean, capture=capture, execute=success)
    assert result['status'] == 'complete'
    assert calls.count('Memory') == 1
    assert calls.count('BasicInfo') == 10
    assert calls.count('PipeUtilization') == 6
    assert result['memory_availability'].startswith('unsupported')


def test_failed_capture_is_retained_not_classified_unsupported(tmp_path):
    def capture(*args):
        raise RuntimeError('invalid device memory')
    result = study.run_pair(study.pairs()['mte2-vector'], '/kernel', tmp_path / 'pair', {}, 'soc',
                            simulate=clean, capture=capture, execute=success)
    retained = json.loads((tmp_path / 'pair/summary.json').read_text())
    assert result == retained
    assert result['status'] == 'failed'
    assert 'invalid device memory' == result['failure']['message']


def test_controls_read_after_display_limit_and_bind_hashes(tmp_path):
    path = tmp_path / 'report/OPPROF_0/simulator/trace.json'
    path.parent.mkdir(parents=True)
    rows = [dict(ph='X', pid='core0.veccore0', tid='VECTOR', name='ADD', ts=i, dur=1)
            for i in range(300)]
    rows.append(dict(ph='X', pid='core0.veccore0', tid='SCALAR', name='WAIT_FLAG', ts=300, dur=2,
                     args={'pc_addr': '0x123', 'detail': 'MTE2 -> VECTOR'}))
    raw = json.dumps({'traceEvents': rows}).encode()
    path.write_bytes(raw)
    result = {'clean_acquisition_gate_passed': True,
              'simulation': {'sources': [{'name': 'OPPROF_0/simulator/trace.json',
                                          'sha256': hashlib.sha256(raw).hexdigest()}]}}
    evidence = study.instruction_evidence(result, tmp_path)
    assert evidence['controls'][0]['pc'] == '0x123'
    assert evidence['controls'][0]['pipeline'] == 'SCALAR'
    assert evidence['source_attribution'].startswith('unavailable')
    assert evidence['cross_core_overlap'].startswith('unavailable')
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='changed'):
        study.instruction_evidence(result, tmp_path)


def test_partial_capture_does_not_publish_control_interpretation(tmp_path):
    result = study.instruction_evidence({'clean_acquisition_gate_passed': False}, tmp_path)
    assert result['status'] == 'unavailable'


def test_simulation_worker_failure_cannot_promote_clean_result(tmp_path, monkeypatch):
    def execute(argv, output, env, **kwargs):
        target = Path(argv[argv.index('--output') + 1])
        target.mkdir()
        (target / 'result.json').write_text('{"clean_acquisition_gate_passed":true}')
        return subprocess.CompletedProcess(argv, 1, '', 'failed')
    monkeypatch.setattr(study.base, 'execute', execute)
    result = study.simulation('/kernel', study.pairs()['mte2-vector'][0], tmp_path / 'sim', {}, 'soc')
    assert not result['clean_acquisition_gate_passed']
    assert result['instruction_evidence']['status'] == 'unavailable'


def test_measurement_passes_explicit_watchdog_to_existing_interfaces(tmp_path, monkeypatch):
    calls = []
    def capture(*args, **kwargs):
        calls.append(kwargs)
        return {'metric': 'observed'}
    monkeypatch.setattr(study.base, 'capture', capture)
    monkeypatch.setattr(study, 'memory_capture', capture)
    for metric in ('BasicInfo', 'PipeUtilization', 'Memory'):
        assert study.measured_capture(['/kernel'], tmp_path, {}, metric, 'exact')['metric'] == 'observed'
    assert calls == [{'timeout': 300}] * 3


def test_simulator_exception_is_separate_and_hardware_checkpointed(tmp_path):
    def simulate(*args):
        raise subprocess.TimeoutExpired(['simulator'], 300)
    def capture(cmd, folder, env, metric, selector):
        return dict(metric=metric, basic_rows=[{'Duration(us)': '2'}])
    output = tmp_path / 'pair'
    result = study.run_pair(study.pairs()['mte2-vector'], '/kernel', output,
                            {'ASCEND_RT_VISIBLE_DEVICES': 'leased-physical'}, 'observed', canary=True,
                            include_simulation=True, simulate=simulate, capture=capture, execute=success)
    assert result['status'] == 'canary_passed'
    for cell in result['cases'].values():
        assert cell['simulation']['failure']['type'] == 'TimeoutExpired'
        capture_record = cell['captures'][0]
        assert capture_record['physical_device_mask'] == 'leased-physical'
        assert capture_record['command'][0] == '/kernel'
        assert capture_record['captured_at_unix'] > 0
        assert (Path(capture_record['artifact_root']) / 'completed.json').is_file()


def test_second_variant_incorrect_never_profiles_either(tmp_path):
    def execute(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, json.dumps({'correctness': cmd[2] == '0'}), '')
    def capture(*args):
        raise AssertionError('profile dispatched despite incorrect paired variant')
    result = study.run_pair(study.pairs()['mte2-vector'], '/kernel', tmp_path / 'pair', {},
                            canary=True, execute=execute, capture=capture)
    assert result['status'] == 'failed'
    assert not result['capture_order']


def test_cli_default_hardware_sweeps_without_soc(tmp_path, monkeypatch):
    class Lease:
        closed = False
        def close(self):
            self.closed = True
    lease = Lease()
    binary = tmp_path / 'native'
    binary.write_bytes(b'compiled-native')
    monkeypatch.setattr(study, 'build', lambda _: binary)
    monkeypatch.setattr(study.base, 'wait_for_device', lambda _: ('leased', {}, lease))
    original = study.run_pair
    def capture(cmd, folder, env, metric, selector):
        return dict(metric=metric, basic_rows=[{'Duration(us)': '3'}])
    def run_pair(*args, **kwargs):
        assert kwargs['include_simulation'] is False
        return original(*args, **kwargs, capture=capture, execute=success)
    monkeypatch.setattr(study, 'run_pair', run_pair)
    output = tmp_path / 'hardware'
    assert study.main(['--output', str(output), '--canary', '--family', 'mode2/imbalanced',
                       '--rounds', '1', '--rounds', '32']) == 0
    summary = json.loads((output / 'summary.json').read_text())
    assert len(summary['pairs']) == 2
    assert {c['case']['rounds'] for p in summary['pairs'].values() for c in p['cases'].values()} == {1, 32}
    assert all(len(p['capture_order']) == 2 for p in summary['pairs'].values())
    assert lease.closed


def test_cli_requested_simulation_requires_soc_before_build(tmp_path):
    with pytest.raises(SystemExit) as error:
        study.main(['--output', str(tmp_path / 'forbidden'), '--simulation'])
    assert error.value.code == 2
    assert not (tmp_path / 'forbidden').exists()
