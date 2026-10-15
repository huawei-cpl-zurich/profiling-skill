import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'benchmarks'))


def test_controls_have_equivalent_placement_groups():
    from a3_profile_sync import cases
    selected = cases()
    assert len({case['id'] for case in selected}) == len(selected)
    for family in ('mte2-vector', 'vector-mte3'):
        for rounds in (1, 32):
            group = [c for c in selected if c['family'] == family and c['rounds'] == rounds]
            assert {c['placement'] for c in group} == {'overlap', 'early-wait', 'late-set'}
            assert len({c['expected_increment'] for c in group}) == 1


def test_manifest_does_not_import_device_dependencies():
    import json
    for name in ('a3_profile_sync',):
        result = subprocess.run([sys.executable, str(ROOT / 'benchmarks' / (name + '.py')),
                                 '--manifest'], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout)


def test_compile_failure_is_retained(tmp_path):
    from a3_native_build import build
    fake = tmp_path / 'compiler'
    fake.write_text('#!/bin/sh\necho compiler-error >&2\nexit 3\n')
    fake.chmod(0o700)
    with pytest.raises(RuntimeError, match='compilation failed'):
        build(tmp_path / 'out', compiler=str(fake))
    assert 'compiler-error' in (tmp_path / 'out' / 'build.stderr').read_text()
    assert (tmp_path / 'out' / 'build.json').is_file()


def test_compile_custom_source_and_extra_flags(tmp_path):
    import json
    from a3_native_build import build
    source = tmp_path / 'custom.asc'
    source.write_text('custom source')
    fake = tmp_path / 'compiler'
    fake.write_text('#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n'
                    'Path(sys.argv[sys.argv.index("-o")+1]).write_bytes(b"binary")\n')
    fake.chmod(0o700)
    executable = build(tmp_path / 'out', source=source, compiler=str(fake),
                       extra_flags=('--sanitizer',))
    assert executable.read_bytes() == b'binary'
    identity = json.loads((tmp_path / 'out' / 'build.json').read_text())
    assert str(source) in identity['command']
    assert '--sanitizer' in identity['command']
    assert len(identity['binary_sha256']) == 64
    assert len(identity['source_sha256']) == 64


def test_missing_compiler_retains_diagnostic(tmp_path):
    import json
    from a3_native_build import build
    with pytest.raises(RuntimeError, match='compilation failed'):
        build(tmp_path / 'out', compiler=str(tmp_path / 'absent'))
    assert json.loads((tmp_path / 'out' / 'build.json').read_text())['failure']['type'] == 'FileNotFoundError'


def test_commands_cover_cross_core_modes_without_late_unmatched_waits():
    from a3_profile_sync import cases, command
    for case in cases():
        argv = command('/tmp/control', case)
        assert argv[0] == '/tmp/control'
        assert len(argv) == 5
        assert int(argv[3]) == case['rounds']
        if case['family'].startswith('mode'):
            assert argv[1] in ('2', '3')
            assert argv[2] in ('0', '1')
