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
