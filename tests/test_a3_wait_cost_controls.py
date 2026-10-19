import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import a3_wait_cost_batch as controls


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = []

    def build(folder, *, source):
        folder.mkdir()
        exe = folder / 'native'
        exe.write_bytes(b'compiled self-contained source')
        return exe

    def execute(argv, folder, env, *, timeout):
        calls.append(('execute', argv))
        folder.mkdir(parents=True)
        record = dict(correctness=True, mismatches=0, warmup=3 if argv[-1] == '1' else 0,
                      launch_count=7 if argv[-1] == '1' else 1,
                      samples_us=[1, 2, 3, 4, 5, 6, 7] if argv[-1] == '1' else [])
        (folder / 'stdout.log').write_text(json.dumps(record))
        return subprocess.CompletedProcess(argv, 0, json.dumps(record), '')

    def capture(argv, folder, env, metric, selector):
        calls.append(('capture', argv, metric))
        assert argv[-1] == '0'  # The actual application never profiles timing/warmup loops.
        folder.mkdir(parents=True)
        (folder / 'retained.log').write_text(metric)
        return dict(metric=metric, selector=selector,
                    basic_rows=[{'Kernel Name': selector, 'Duration(us)': '2.5'}],
                    **({'availability': 'available'} if metric == 'Memory' else {}))

    monkeypatch.setattr(controls.a3_native_build, 'build', build)
    monkeypatch.setattr(controls.base, 'execute', execute)
    monkeypatch.setattr(controls.base, 'capture', capture)
    monkeypatch.setattr(controls, 'memory_capture',
                        lambda argv, folder, env, selector: capture(argv, folder, env, 'Memory', selector))
    return calls


def test_equal_work_oracle_and_sweep():
    assert len(controls.cases()) == 32
    for start in range(0, 32, 4):
        transfer, compute, serial, overlap = controls.cases()[start:start + 4]
        a, b = controls.expected(transfer), controls.expected(compute)
        assert controls.expected(serial) == controls.expected(overlap)
        assert controls.expected(serial) == [x + y for x, y in zip(a, b)]
        assert len(a) == transfer['n'] * 2
        assert len(set(a[:transfer['n']])) == 17
        assert set(b[compute['n']:]) == {compute['rounds'] * compute['repetitions']}


def test_capture_counts_interleaving_resume_and_provenance(tmp_path, fake_runtime):
    selected = controls.cases(True)
    output = tmp_path / 'run'
    result = controls.run(selected, output, {}, physical=7)
    assert len(result['capture_order']) == 44
    for cell in result['cells'].values():
        assert cell['event_timing']['median_us'] == 4
        assert cell['profile_timing']['samples_us'] == [2.5] * 5
        assert cell['event_evidence']['physical_device'] == 7
        assert len(cell['captures']) == 11
        assert cell['captures'][-1]['availability'] == 'available'
    assert result['capture_order'][0]['case'].endswith('/transfer')
    assert result['capture_order'][4]['case'].endswith('/compute')
    comparison = result['comparisons']['1024/1/1']
    assert comparison['paired_speedup'] == 1
    assert comparison['isolated_stage_sum_us'] == 8
    old_calls = list(fake_runtime)
    resumed = controls.run(selected, output, {}, physical=9, resume=True)
    assert fake_runtime == old_calls
    assert resumed['cells'][selected[0]['id']]['event_evidence']['physical_device'] == 7
    assert resumed['capture_order'][0]['physical_device'] == 7


def test_failure_stops_before_profiling_and_keeps_logs(tmp_path, fake_runtime, monkeypatch):
    execute = controls.base.execute

    def failed(*args, **kwargs):
        result = execute(*args, **kwargs)
        return subprocess.CompletedProcess(result.args, 1, result.stdout, 'runtime failure')

    monkeypatch.setattr(controls.base, 'execute', failed)
    with pytest.raises(RuntimeError, match='workload process failed'):
        controls.run(controls.cases(True), tmp_path / 'run', {}, physical=0)
    assert not any(c[0] == 'capture' for c in fake_runtime)
    assert list((tmp_path / 'run').rglob('stdout.log'))


def test_memory_rejection_stops_further_replays_across_groups(tmp_path, fake_runtime, monkeypatch):
    capture = controls.memory_capture

    def unsupported(*args):
        return dict(capture(*args), availability='unsupported', diagnostic='Memory unsupported')

    monkeypatch.setattr(controls, 'memory_capture', unsupported)
    selected = controls.cases(True) + controls.cases()[4:8]
    result = controls.run(selected, tmp_path / 'run', {}, physical=0)
    assert sum(c[0] == 'capture' and c[2] == 'Memory' for c in fake_runtime) == 1
    assert sum(c[0] == 'capture' and c[2] == 'BasicInfo' for c in fake_runtime) == 40
    assert sum(c[0] == 'capture' and c[2] == 'PipeUtilization' for c in fake_runtime) == 24
    assert result['memory_availability'] == 'unsupported'
    assert all(c['memory_availability'] == 'unsupported' for c in result['cells'].values())
    assert sum(e.get('availability') == 'unsupported' for c in result['cells'].values()
               for e in c['captures']) == 1
    assert all(len(c['profile_timing']['samples_us']) == 5 for c in result['cells'].values())
    old_calls = list(fake_runtime)
    controls.run(selected, tmp_path / 'run', {}, physical=0, resume=True)
    assert fake_runtime == old_calls


def test_memory_runtime_failure_is_not_capability_rejection(tmp_path, fake_runtime, monkeypatch):
    def failed(*args):
        raise RuntimeError('workload runtime failure')

    monkeypatch.setattr(controls, 'memory_capture', failed)
    with pytest.raises(RuntimeError, match='workload runtime failure'):
        controls.run(controls.cases(True), tmp_path / 'run', {}, physical=0)
    result = json.loads((tmp_path / 'run/summary.json').read_text())
    assert 'memory_availability' not in result


def test_bad_event_samples_are_not_zero_filled(tmp_path, fake_runtime, monkeypatch):
    execute = controls.base.execute

    def invalid(*args, **kwargs):
        result = execute(*args, **kwargs)
        record = json.loads(result.stdout)
        if record['samples_us']:
            record['samples_us'][0] = 0
        return subprocess.CompletedProcess(result.args, 0, json.dumps(record), '')

    monkeypatch.setattr(controls.base, 'execute', invalid)
    with pytest.raises(ValueError, match='finite and positive'):
        controls.run(controls.cases(True), tmp_path / 'run', {}, physical=0)


def test_resume_rejects_source_config_binary_and_evidence_changes(tmp_path, fake_runtime):
    output = tmp_path / 'run'
    controls.run(controls.cases(True), output, {}, physical=0)
    with pytest.raises(RuntimeError, match='configuration/source mismatch'):
        controls.run(controls.cases(), output, {}, physical=0, resume=True)
    exe = output / 'build/native'
    exe.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='executable changed'):
        controls.run(controls.cases(True), output, {}, physical=0, resume=True)
    exe.write_bytes(b'compiled self-contained source')
    next(output.rglob('retained.log')).write_text('changed')
    with pytest.raises(RuntimeError, match='evidence changed'):
        controls.run(controls.cases(True), output, {}, physical=0, resume=True)


def test_cli_rejects_invalid_group_before_device_probe(tmp_path):
    result = subprocess.run([sys.executable, str(Path(controls.__file__)), '--output', str(tmp_path),
                             '--canary', '--group', '8192/32/16'], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'unknown or excluded group' in result.stderr


@pytest.mark.parametrize('noisy_mode', ['serialized', 'overlap', 'transfer', None])
def test_pair_confidence_preserves_medians_and_noise(noisy_mode):
    cells = {}
    originals = {}
    for case in controls.cases(True):
        samples = [5.94, 27.24, 7.36, 5.76, 5.88, 5.2, 29.46] \
            if case['mode'] == noisy_mode else [6.0] * 7
        cells[case['id']] = dict(case=case, event_timing=controls.base.summarize(samples))
        originals[case['id']] = list(samples)
    comparison = controls.conclusions(cells)['1024/1/1']
    noisy_pair = noisy_mode in ('serialized', 'overlap')
    assert comparison['low_confidence'] is noisy_pair
    assert (comparison['status'] == 'inconclusive_noisy_event_timing') is noisy_pair
    assert comparison['event_low_confidence'] == {m: m == noisy_mode for m in controls.MODES}
    assert comparison['paired_speedup'] == comparison['event_medians_us']['serialized'] / \
        comparison['event_medians_us']['overlap']
    assert {key: cell['event_timing']['samples_us'] for key, cell in cells.items()} == originals


def test_actual_kernel_event_lifecycle_and_outputs_with_cpu_stubs(tmp_path):
    compiler = shutil.which('g++')
    if compiler is None:
        pytest.skip('g++ unavailable for CPU-only kernel contract test')
    (tmp_path / 'acl').mkdir()
    (tmp_path / 'acl/acl.h').write_text('')
    (tmp_path / 'kernel_operator.h').write_text(r'''
#pragma once
#include <cassert>
#include <algorithm>
#include <vector>
#define __global__
#define __aicore__
#define __gm__
using GM_ADDR = unsigned char*;
namespace AscendC {
using TEventID = int;
enum class HardEvent { V_MTE2, MTE2_V, V_MTE3 };
enum class TPosition { VECCALC };
enum { PIPE_V, PIPE_ALL };
inline bool live[3] = {};
inline int pending[3] = {};
template<class T> struct LocalTensor {
    T* p;
    LocalTensor operator[](int n) const { return {p+n}; }
};
template<class T> struct GlobalTensor {
    T* p;
    void SetGlobalBuffer(T* pointer, int) { p=pointer; }
    GlobalTensor operator[](int n) const { return {p+n}; }
};
template<TPosition> struct TBuf {
    std::vector<float> values;
    template<class T> LocalTensor<T> Get() { return {values.data()}; }
};
struct TPipe {
    template<class B> void InitBuffer(B& b, int bytes) { b.values.resize(bytes/4); }
    template<HardEvent E> TEventID AllocEventID() {
        int e=int(E); assert(!live[e]); live[e]=true; return e+3;
    }
    template<HardEvent E> void ReleaseEventID(TEventID id) {
        int e=int(E); assert(id==e+3 && live[e] && pending[e]==0); live[e]=false;
    }
    ~TPipe() { for(int e=0;e<3;++e) assert(!live[e] && pending[e]==0); }
};
template<HardEvent E> void SetFlag(TEventID id) {
    int e=int(E); assert(id==e+3 && live[e] && pending[e]==0); ++pending[e];
}
template<HardEvent E> void WaitFlag(TEventID id) {
    int e=int(E); assert(id==e+3 && live[e] && pending[e]==1); --pending[e];
}
template<int P> void PipeBarrier() {}
inline void Duplicate(LocalTensor<float> a, float value, int n) { std::fill_n(a.p,n,value); }
inline void Adds(LocalTensor<float> a, LocalTensor<float> b, float value, int n) {
    for(int i=0;i<n;++i) a.p[i]=b.p[i]+value;
}
template<class D,class S> void DataCopy(D a, S b, int n) { std::copy_n(b.p,n,a.p); }
}
''')
    source = Path(controls.__file__).resolve().parents[1] / 'benchmarks/a3_wait_cost_control.asc'
    program = tmp_path / 'contract.cpp'
    program.write_text('#define A3_WAIT_COST_KERNEL_ONLY\n#include ' + json.dumps(str(source)) + r'''
int main() {
    int cases=0;
    for(int n: {1024,8192}) for(int rounds: {1,32}) for(int reps: {1,16})
    for(int mode=0;mode<4;++mode) {
        std::vector<float> x(n), y(2*n,-999);
        for(int i=0;i<n;++i) x[i]=float(i%17);
        a3_wait_cost_probe((GM_ADDR)x.data(),(GM_ADDR)y.data(),mode,n,rounds,reps);
        for(int i=0;i<n;++i) {
            if(y[i]!=(mode==1?0:x[i]) || y[n+i]!=(mode==0?0:rounds*reps)) return 1;
        }
        ++cases;
    }
    std::printf("%d\n",cases);
}
''')
    executable = tmp_path / 'contract'
    built = subprocess.run([compiler, '-std=c++17', '-I', str(tmp_path), str(program), '-o',
                            str(executable)], capture_output=True, text=True, timeout=60)
    assert built.returncode == 0, built.stderr
    ran = subprocess.run([str(executable)], capture_output=True, text=True, timeout=30)
    assert ran.returncode == 0, ran.stderr
    assert ran.stdout.strip() == '32'
