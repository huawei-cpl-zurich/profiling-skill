from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_actual_native_control_event_lifecycles(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('CPU C++ compiler unavailable')
    (tmp_path / 'kernel_operator.h').write_bytes((ROOT / 'tests/fixtures/a3-event-lifecycle.hpp').read_bytes())
    (tmp_path / 'acl').mkdir()
    (tmp_path / 'acl/acl.h').write_text('')
    # Compile the actual device function bodies, excluding the ACL host launch
    # syntax unsupported by a CPU compiler. This tests API ownership, not NPU math.
    kernels = (ROOT / 'benchmarks/a3_sync_control.asc').read_text().split('#define ACL_CHECK', 1)[0]
    driver = r'''
int main() {
    for (int rounds : {1, 32}) for (int placement : {0, 1, 2})
        for (int family : {0, 1}) a3_intra_probe(nullptr, nullptr, family, placement, rounds);
    for (int rounds : {1, 32}) for (int placement : {0, 1}) for (int imbalanced : {0, 1}) {
        cpu_aic = false;
        for (cpu_block = 0; cpu_block < 2; ++cpu_block)
            a3_inter0_probe(nullptr, placement, rounds, imbalanced);
        for (bool aic : {false, true}) {
            cpu_aic = aic;
            for (cpu_block = 0; cpu_block < 4; ++cpu_block)
                a3_inter2_probe(nullptr, nullptr, placement, rounds, imbalanced);
        }
    }
    assert(AscendC::allocations > 0 && AscendC::allocations == AscendC::releases);
    assert(AscendC::peak[0] >= 2 && AscendC::peak[1] >= 2 && AscendC::cross_calls > 0);
}
'''
    source = tmp_path / 'control.cpp'
    source.write_text(kernels + driver)
    binary = tmp_path / 'control'
    compiled = subprocess.run([compiler, '-std=c++17', '-I', str(tmp_path), str(source), '-o', str(binary)],
                              capture_output=True, text=True, timeout=30)
    assert compiled.returncode == 0, compiled.stderr
    result = subprocess.run([str(binary)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
