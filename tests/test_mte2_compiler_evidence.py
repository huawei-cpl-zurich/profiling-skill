import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('evidence', ROOT / 'scripts/mte2_compiler_evidence.py')
evidence = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evidence)


def manifest(tmp_path, exit_code=0, invocations=True):
    root = tmp_path / 'capture'
    root.mkdir()
    inp = root / 'input.mlir'
    inp.write_text('module {\n scf.for %i\n %v = memref.load %x\n memref.copy %x, %y\n}\n')
    out = root / 'kernel.o'
    out.write_bytes(b'\x00binary')
    err = root / 'stderr.txt'
    err.write_text('retained compiler diagnostics\n')
    data = dict(reference_exit_code=exit_code, reference_stdout=str(err),
                reference_stderr=str(err), real_tools={'bishengir-compile': '/owned/compiler'},
                wrappers={'bishengir-compile': '/owned/wrapper'}, triton_dumps=[],
                invocations=[dict(tool='bishengir-compile', id='actual', argv=['compiler', '-O2'],
                                  input_copy=str(inp), output_copy=str(out),
                                  exit_code=exit_code, stdout=str(err), stderr=str(err))]
                if invocations else [])
    path = root / 'capture-manifest.json'
    path.write_text(json.dumps(data))
    return path


def test_inspect_extracts_actual_operations_and_hashes_binary(tmp_path):
    path = manifest(tmp_path)
    result = evidence.inspect(path)
    assert result['status'] == 'captured'
    assert result['invocations'][0]['argv'] == ['compiler', '-O2']
    artifact = result['artifacts'][0]
    assert artifact['operations']['loads'][0]['line'] == 3
    assert artifact['operations']['copies'][0]['text'] == 'memref.copy %x, %y'
    assert any(item['kind'] == 'compiler_output' and 'operations' not in item
               for item in result['artifacts'])
    assert result['later_artifact_status'] == 'not_observed'


@pytest.mark.parametrize('failure', ['no_invocation', 'missing_input', 'failed_compiler'])
def test_exit_zero_is_not_enough(tmp_path, failure):
    path = manifest(tmp_path, exit_code=7 if failure == 'failed_compiler' else 0,
                    invocations=failure != 'no_invocation')
    if failure == 'missing_input':
        (path.parent / 'input.mlir').unlink()
    result = evidence.inspect(path)
    assert result['status'] == 'failed'
    assert result['errors']
    if failure == 'failed_compiler':
        assert result['invocations'][0]['stderr']['text'] == 'retained compiler diagnostics\n'


def test_inspect_retains_assembly_and_separates_late_evidence(tmp_path):
    path = manifest(tmp_path)
    assembly = tmp_path / 'assembly'
    assembly.mkdir()
    (assembly / 'ttir.txt').write_text('%v = tt.load %x\n')
    (assembly / 'llir.txt').write_text('define void @kernel() { ret void }')
    result = evidence.inspect(path, assembly)
    assert result['later_artifact_status'] == 'observed_not_hardware_schedule'
    assert {item.get('stage') for item in result['artifacts']} >= {'ttir', 'llir'}


def test_real_helper_capture_preserves_shared_runtime_and_error(tmp_path, monkeypatch):
    from types import SimpleNamespace
    package = tmp_path / 'shared/triton'
    binary = package / 'backends/ascend/bishengir/bin/bishengir-compile'
    binary.parent.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    binary.write_text('#!/bin/sh\ncp "$1" "$3"\nprintf "compile error detail" >&2\nexit 7\n')
    binary.chmod(0o755)
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(__file__=str(package / '__init__.py')))
    monkeypatch.setattr(evidence.adapter.shutil, 'which', lambda name: None)
    tla = os.environ.get('TLA_INTEGRATION_TESTS_ROOT')
    helper = (Path(tla) / 'scripts/capture-triton-compiler-artifacts.sh' if tla else
              (Path.home() / '.agents/skills/mlir-triton-dump/SKILL.md').resolve().parents[2]
              / 'vendor/tla-integration-tests/scripts/capture-triton-compiler-artifacts.sh')
    if not helper.is_file():
        pytest.skip('managed helper not available; supply it through test environment')
    source = tmp_path / 'workload.py'
    source.write_text('import subprocess, triton\nfrom pathlib import Path\n'
                      'p=Path(triton.__file__).parent/"backends/ascend/bishengir/bin/bishengir-compile"\n'
                      'i=Path("input.mlir"); i.write_text("memref.load %x")\n'
                      'raise SystemExit(subprocess.call([str(p),str(i),"-o","output.o"]))\n')
    original = hashlib.sha256(binary.read_bytes()).hexdigest()
    monkeypatch.chdir(tmp_path)
    output = tmp_path / 'job'
    result = evidence.capture(helper, [sys.executable, str(source)], output, [source],
                              {'target': 'test-only', 'specialization': {}, 'options': {}})
    assert result['status'] == 'failed'
    assert result['reference_exit_code'] == 7
    assert result['invocations'][0]['stderr']['text'] == 'compile error detail'
    assert hashlib.sha256(binary.read_bytes()).hexdigest() == original
    assert (output / 'evidence.json').is_file()
    assert result['source_files'][0]['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_cli_inspection_nonzero_and_complete_error_artifact(tmp_path):
    path = manifest(tmp_path, exit_code=2)
    output = tmp_path / 'summary.json'
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/mte2_compiler_evidence.py'),
                             'inspect', '--manifest', str(path), '--output', str(output)],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert json.loads(output.read_text())['reference_exit_code'] == 2


def test_control_manifest_without_triton_or_device():
    result = subprocess.run([sys.executable, str(ROOT / 'benchmarks/mte2_compiler_controls.py'),
                             '--manifest'], capture_output=True, text=True, check=True)
    cases = json.loads(result.stdout)['cases']
    assert len(cases) == 4
    assert cases[0]['specialization']['PREDICATE'] is False
    assert cases[1]['specialization']['PREDICATE'] is True
    assert cases[2]['specialization'] == cases[3]['specialization']
    assert [case['options'] for case in cases[2:]] == [{'num_stages': 1}, {'num_stages': 2}]
