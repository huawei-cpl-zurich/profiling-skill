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


def test_suffixless_copied_llvm_input_uses_actual_manifest_binding(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    original = Path(data['invocations'][0]['input_copy'])
    copied = original.with_name('captured-input')
    original.rename(copied)
    before = copied.read_bytes()
    invocation = data['invocations'][0]
    invocation.update(input_copy=str(copied), input_arg='/compile/kernel.ll',
                      argv=['bisheng', '/compile/kernel.ll', '/linked/library.mlir'])
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['later_artifact_status'] == 'observed_not_hardware_schedule'
    assert result['artifacts'][0]['stage'] == 'llvm'
    assert result['invocations'][0]['input_arg'] == '/compile/kernel.ll'
    assert copied.read_bytes() == before
    invocation.update(input_arg='/compile/kernel.ttadapter.mlir',
                      argv=['bisheng', '/compile/kernel.ttadapter.mlir', '/linked/library.ll'])
    path.write_text(json.dumps(data))
    assert evidence.inspect(path)['later_artifact_status'] == 'not_observed'


@pytest.mark.parametrize('tool,bound,flags,role', [
    ('bishengir-compile', '/compile/kernel.ttadapter.mlir', ['--target=Ascend910_9382'], 'adapter'),
    ('hivmc', '/compile/module.hivm.opt.mlir', ['--enable-static-bare-ptr=true'], 'hivm'),
    ('bisheng', '/compile/kernel.ll', ['--cce-aicore-only', '-O2'], 'device_llvm'),
    ('bisheng', '/compile/kernel.bc', ['--cce-aicore-arch=dav-c220-vec'], 'device_llvm'),
    ('bisheng', '/compile/kernel.host.ll', ['-x', 'ir', '--shared'], 'host_llvm'),
    ('bisheng', '/compile/kernel.host.ll', ['-O2'], 'unknown'),
    ('unreviewed-compiler', '/compile/kernel.ttadapter.mlir', [], 'unknown'),
    ('bisheng', '/compile/kernel.ll', ['--shared', '--cce-aicore-only'], 'unknown'),
])
def test_stage_index_uses_bound_consumer_roles_without_modifying_artifacts(
        tmp_path, tool, bound, flags, role):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    invocation = data['invocations'][0]
    original = Path(invocation['input_copy'])
    copied = original.with_name('captured-input')
    original.rename(copied)
    before = copied.read_bytes()
    invocation.update(tool=tool, input_arg=bound, input_copy=str(copied),
                      argv=[tool, bound, *flags, '-o', 'kernel.o'])
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    entry, = result['stage_index']
    assert result['status'] == 'captured'
    assert entry['role'] == role
    assert entry['input_arg'] == bound
    assert entry['argv'] == invocation['argv']
    assert entry['invocation_id'] == 'actual'
    assert entry['provenance'] == 'capture_manifest_consumer_binding'
    assert entry['path'] == str(copied)
    assert entry['sha256'] == hashlib.sha256(before).hexdigest()
    assert entry['bytes'] == len(before)
    assert copied.read_bytes() == before
    # The additive index must not rename legacy artifact stages or alter hashes.
    assert result['artifacts'][0]['sha256'] == entry['sha256']
    expected = 'llvm' if bound.endswith('.ll') else ('llvm_bitcode' if bound.endswith('.bc')
                                                  else Path(bound).stem)
    assert result['artifacts'][0]['stage'] == expected


def test_stage_index_does_not_promote_filenames_or_linked_inputs(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    invocation = data['invocations'][0]
    invocation.update(tool='bisheng', input_arg='/compile/early.ttadapter.mlir',
                      argv=['bisheng', '/compile/early.ttadapter.mlir',
                            '/linked/device.ll', '--cce-aicore-only'])
    # A misleading copy filename must not override its actual compiler binding.
    misleading = path.parent / 'kernel.host.ll'
    Path(invocation['input_copy']).rename(misleading)
    invocation['input_copy'] = str(misleading)
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['stage_index'][0]['role'] == 'unknown'
    invocation['input_arg'] = '/not-in-argv/kernel.ll'
    path.write_text(json.dumps(data))
    assert evidence.inspect(path)['stage_index'][0]['role'] == 'unknown'


def test_stage_index_retains_failed_compiler_binding_but_not_queries(tmp_path):
    path = manifest(tmp_path, exit_code=7)
    data = json.loads(path.read_text())
    data['invocations'][0].update(tool='hivmc', input_arg='/compile/module.mlir',
                                  argv=['hivmc', '/compile/module.mlir'])
    data['invocations'].append(dict(id='query', tool='bisheng',
                                    argv=['bisheng', '--version'], exit_code=0))
    path.write_text(json.dumps(data))
    output = tmp_path / 'evidence.json'
    proc = subprocess.run([sys.executable, str(ROOT / 'scripts/mte2_compiler_evidence.py'),
                           'inspect', '--manifest', str(path), '--output', str(output)],
                          capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert proc.returncode == 2
    assert result['status'] == 'failed'
    assert len(result['stage_index']) == 1
    assert result['stage_index'][0]['role'] == 'hivm'
    assert result['invocations'][0]['exit_code'] == 7
    assert result['invocations'][1]['kind'] == 'tool_query'
    Path(data['invocations'][0]['input_copy']).unlink()
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['status'] == 'failed'
    assert result['stage_index'] == []
    assert any('missing actual compiler input' in error for error in result['errors'])


@pytest.mark.parametrize('query', ['--help', '-h', '--version', '-version',
                                  '-print-targets', '--print-targets'])
def test_actual_compile_may_include_inputless_backend_queries(tmp_path, query):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['invocations'].append(dict(id='query', tool='bisheng',
                                    argv=['bisheng', query], exit_code=0,
                                    input_copy=None, output_copy=None))
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['status'] == 'captured'
    assert result['invocations'][1]['kind'] == 'tool_query'
    assert result['invocations'][1]['argv'] == ['bisheng', query]


def test_queries_alone_and_compile_arguments_are_not_compilation_evidence(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['invocations'] = [dict(id='query', tool='bisheng', argv=['bisheng', '--help'],
                               exit_code=0, input_copy=None)]
    path.write_text(json.dumps(data))
    assert evidence.inspect(path)['status'] == 'failed'
    data['invocations'][0]['argv'].append('missing.mlir')
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['status'] == 'failed'
    assert result['invocations'][0]['kind'] == 'compiler'
    assert any('missing actual compiler input' in error for error in result['errors'])


def test_failed_optional_query_does_not_discard_successful_actual_compile(tmp_path):
    path = manifest(tmp_path)
    data = json.loads(path.read_text())
    data['invocations'].append(dict(id='query', tool='hivmc', argv=['hivmc', '--version'],
                                    exit_code=1, input_copy=None))
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['status'] == 'captured'
    assert result['invocations'][1]['exit_code'] == 1
    assert result['invocations'][1]['warning']
    data['invocations'][1]['argv'] = ['hivmc', '--unknown-option']
    path.write_text(json.dumps(data))
    result = evidence.inspect(path)
    assert result['status'] == 'failed'
    assert any('compiler failed' in item for item in result['errors'])


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


def test_control_run_emits_boolean_correctness_contract(tmp_path, monkeypatch, capsys):
    from types import SimpleNamespace
    loader = importlib.util.spec_from_file_location('controls', ROOT / 'benchmarks/mte2_compiler_controls.py')
    controls = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(controls)
    class Tensor:
        def to(self, *args): return self
        def numel(self): return 1024
        def __mod__(self, other): return self
        def __eq__(self, other): return self
        def cpu(self): return self
    checks = []
    torch = SimpleNamespace(manual_seed=lambda _: None, float32='fp32',
        randn=lambda *a, **k: Tensor(), where=lambda *a: Tensor(),
        arange=lambda *a, **k: Tensor(), empty_like=lambda *a: Tensor(),
        npu=SimpleNamespace(synchronize=lambda: None),
        testing=SimpleNamespace(assert_close=lambda *a, **k: checks.append(True)))
    compiled = SimpleNamespace(asm={'ttir': '%v = tt.load %x'}, metadata='actual metadata')
    class Kernel:
        def __getitem__(self, grid):
            return lambda *a, **k: compiled
    triton = SimpleNamespace(__version__='test', cdiv=lambda n, b: (n+b-1)//b,
        runtime=SimpleNamespace(driver=SimpleNamespace(active=SimpleNamespace(
            get_current_target=lambda: 'test target'))))
    monkeypatch.setitem(sys.modules, 'torch', torch)
    monkeypatch.setitem(sys.modules, 'torch_npu', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setattr(controls, 'kernels', lambda: (Kernel(), Kernel()))
    controls.run('mask', 'before', tmp_path / 'assembly')
    result = json.loads(capsys.readouterr().out)
    assert result['correctness'] is True
    assert checks == [True]
    assert json.loads((tmp_path / 'assembly/runtime.json').read_text()) == result
