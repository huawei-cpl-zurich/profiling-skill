import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))


def test_manifest_and_commands_cover_both_implementations():
    from a3_profile_small_matmul import cases, command
    manifest = cases()
    assert len(manifest) == 12
    assert len({c['id'] for c in manifest}) == 12
    for c in manifest:
        argv = command('/tmp/native', c, Path('/tmp/artifacts'))
        if c['implementation'] == 'native':
            assert argv == ['/tmp/native', str(c['m']), str(c['n']), str(c['k'])]
        else:
            assert '--triton-case' in argv
    result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/a3_profile_small_matmul.py'),
                             '--manifest'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == manifest


def test_layouts_are_bijections_and_match_documented_fractals():
    from a3_profile_small_matmul import offset
    for layout in ('ZZ', 'ZN', 'NZ'):
        addresses = [offset(r, c, 32, 64, layout) for r in range(32) for c in range(64)]
        assert sorted(addresses) == list(range(32 * 64))
    assert [offset(0, c, 32, 64, 'ZZ') for c in (0, 1, 16)] == [0, 1, 256]
    assert [offset(r, 0, 32, 64, 'ZN') for r in (0, 1, 16)] == [0, 1, 1024]
    assert [offset(r, 0, 32, 64, 'NZ') for r in (0, 1, 16)] == [0, 16, 256]


def test_packed_cpu_product_uses_asymmetric_inputs():
    from a3_profile_small_matmul import offset, values, reference
    m, n, k = 16, 64, 32
    a, b = values(m, k, 3), values(k, n, 7)
    ap, bp = [0.] * (m * k), [0.] * (k * n)
    for r in range(m):
        for c in range(k):
            ap[offset(r, c, m, k, 'ZZ')] = a[r * k + c]
    for r in range(k):
        for c in range(n):
            bp[offset(r, c, k, n, 'ZN')] = b[r * n + c]
    actual = [sum(ap[offset(r, q, m, k, 'ZZ')] * bp[offset(q, c, k, n, 'ZN')]
                  for q in range(k)) for r in range(m) for c in range(n)]
    assert actual == reference(a, b, m, n, k)
    assert len(set(actual)) > 8


def test_triton_case_profiles_through_real_process_and_csv_boundary(tmp_path):
    import os
    sys.path.insert(0, str(ROOT / 'scripts'))
    from a3_memory_batch import run_case
    from a3_profile_small_matmul import cases
    profiler = tmp_path / 'msprof'
    profiler.write_text('#!' + sys.executable + '\n'
        'import sys\nfrom pathlib import Path\n'
        'args=sys.argv[1:]\n'
        'selector=next((a.split("=",1)[1] for a in args if a.startswith("--kernel-name=")),None)\n'
        'if selector not in (None,"a3_small_dot"): raise SystemExit(3)\n'
        'out=Path(next(a.split("=",1)[1] for a in args if a.startswith("--output=")))\n'
        'out.mkdir(parents=True)\n'
        '(out/"OpBasicInfo.csv").write_text("Op Name,Op Type,Task Duration(us),Block Dim\\na3_small_dot,cube,2.5,1\\n")\n'
        'if "--aic-metrics=PipeUtilization" in args:\n'
        ' (out/"PipeUtilization.csv").write_text("Op Name,Block Id,mac_ratio\\na3_small_dot,0,0.25\\n")\n'
        'print("Profiling running finished. All task success.")\n')
    profiler.chmod(0o700)
    env = dict(os.environ, PATH=str(tmp_path) + os.pathsep + os.environ['PATH'])
    case = next(c for c in cases() if c['id'] == 'triton/16x16x16')
    result = run_case([sys.executable, '-c', 'print(\'{"correctness":true}\')'],
                      tmp_path / 'case', env, case['selector'])
    assert result['selector'] == 'a3_small_dot'
    assert result['timing']['samples_us'] == [2.5] * 5
    assert len(result['pipe_captures']) == 3
    assert all(capture['pipe_rows'][0]['Op Name'] == 'a3_small_dot'
               for capture in result['pipe_captures'])
