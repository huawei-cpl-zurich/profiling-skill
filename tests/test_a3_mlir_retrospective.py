import importlib.util
import json
from pathlib import Path
import pytest


ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location('audit', ROOT / 'scripts/a3_mlir_retrospective.py')
audit = importlib.util.module_from_spec(loader)
loader.loader.exec_module(audit)


def cell(exp='loop-bound', variant='before', **extra):
    return dict(arguments=['--experiment', exp, '--sweep', 'test', '--variant', variant],
                sweep_parameters={'n': 256, 'work': 64}, kind='hypothesis',
                variant=variant, id=exp + '/' + variant, kernel_names=['a3_vector_probe'], **extra)


def test_plan_deduplicates_same_specialization_and_retains_cells():
    a, b = cell(), cell()
    b['id'] = 'another'
    run = {'source_commit': 'frozen', 'expected_cells': 2,
           'pairs': [{'id': 'one', 'cells': [a]}, {'id': 'two', 'cells': [b]}]}
    result = audit.make_plan(run)
    assert len(result['cells']) == 2
    assert len(result['specializations']) == 1


def test_loop_constants_and_schedule_options():
    assert audit.specializations(cell())[0]['constants']['loops'] == 256
    assert audit.specializations(cell(variant='after'))[0]['constants']['loops'] == 64
    assert audit.specializations(cell('schedule-knobs', 'after'))[0]['explicit_options'] == {
        'num_warps': 8, 'num_stages': 2}


def test_amplification_runtime_value_is_not_specialization():
    a = cell('load-masking')
    a['arguments'] += ['--repetitions', '32']
    b = dict(a, arguments=a['arguments'][:-1] + ['128'])
    assert audit.specializations(a) == audit.specializations(b)
    assert audit.specializations(a)[0]['signature']['repeat_ptr'] == '*i32'


def test_split_selector_coverage_and_duplicate_cells():
    a = cell('fused-vs-split', 'after')
    a['kind'] = 'timeline'
    names = ['a3_phase_copy', 'a3_phase_compute', 'a3_phase_store']
    a['kernel_names'] = names
    run = {'source_commit': 'x', 'expected_cells': 1, 'pairs': [{'id': 'x', 'cells': [a]}]}
    assert len(audit.make_plan(run)['specializations']) == 3
    a['kernel_names'] = ['incorrect']
    with pytest.raises(ValueError, match='selector'):
        audit.make_plan(run)


def test_hash_verification_rejects_changed_original(tmp_path):
    path = tmp_path / 'ttir'
    path.write_text('original')
    import hashlib
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert audit.verify_original(path, sha)['provenance'] == 'original_retained_artifact'
    path.write_text('changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        audit.verify_original(path, sha)


def test_launch_guard_blocks_and_restores_on_error():
    class JIT:
        def run(self): return 'launched'
        def warmup(self): return 'warmup'
    fn = JIT()
    with audit.forbid_launch(JIT):
        with pytest.raises(RuntimeError, match='forbidden'):
            fn.run()
        with pytest.raises(RuntimeError, match='forbidden'):
            fn.warmup()
    assert fn.run() == 'launched'


def test_operation_evidence_keeps_lines_not_pipe_claims():
    evidence = audit.inspect_ir('scf.for %x\n  %v = tt.load %p\n  tt.store %q, %v\n')
    assert evidence['loads'] == [{'line': 2, 'text': '%v = tt.load %p'}]
    assert evidence['matrix'] == []


def test_original_transfer_verification(tmp_path):
    import base64
    import gzip
    import hashlib
    data = b'scf.for %x'
    record = {'status': 'verified_original', 'sha256': hashlib.sha256(data).hexdigest(),
              'bytes': len(data), 'gzip_base64': base64.b64encode(gzip.compress(data)).decode()}
    result = audit.import_originals([record], tmp_path)
    assert Path(result[0]['local_path']).read_bytes() == data
    assert result[0]['operations']['loops'][0]['line'] == 1
    record['sha256'] = 'invalid'
    with pytest.raises(ValueError, match='hash or length'):
        audit.import_originals([record], tmp_path)


def test_isolated_capture_never_replaces_shared_tool(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    package = tmp_path / 'shared' / 'triton'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('')
    binary_dir = package / 'backends' / 'ascend' / 'bishengir' / 'bin'
    binary_dir.mkdir(parents=True)
    compiler = binary_dir / 'bishengir-compile'
    compiler.write_text('#!/bin/sh\nprintf original\n')
    compiler.chmod(0o755)
    helper = tmp_path / 'helper.sh'
    helper.write_text('''#!/bin/bash
set -eu
while [ "$1" != "--real-tool" ]; do shift; done
owned="${2#*=}"
mv "$owned" "$owned.backup"
printf '#!/bin/sh\\nprintf shim' > "$owned"
"$owned.backup"
mv "$owned.backup" "$owned"
''')
    monkeypatch.setitem(sys.modules, 'triton', SimpleNamespace(__file__=str(package / '__init__.py')))
    monkeypatch.setattr(audit.shutil, 'which', lambda name: None)
    original = compiler.read_bytes()
    result = audit.isolated_capture(helper, ['unused'], tmp_path / 'isolated')
    assert result['exit_code'] == 0
    assert compiler.read_bytes() == original
    assert (tmp_path / 'isolated/runtime/triton/backends/ascend/bishengir/bin/bishengir-compile').read_bytes() == original


def test_dispatch_constants_match_actual_original_host_dispatch(monkeypatch):
    """Exercise real _launch with inert kernels, never a device or JIT launch."""
    from types import SimpleNamespace
    load = importlib.util.spec_from_file_location('original_probe', ROOT / 'benchmarks/a3_profile_microbench.py')
    original = importlib.util.module_from_spec(load)
    load.loader.exec_module(original)
    captured = []
    class InertKernel:
        def __init__(self, name): self.name = name
        def __getitem__(self, grid):
            def record(*args, **kwargs): captured.append((self.name, kwargs))
            return record
    names = ('a3_vector_probe', 'a3_head_rank_probe', 'a3_key_tile_probe',
             'a3_row_reduce_probe', 'a3_empty_skip_probe', 'a3_dot_probe',
             'a3_phase_probe', 'a3_phase_copy', 'a3_phase_compute', 'a3_phase_store')
    for name in names:
        monkeypatch.setattr(original, name, InertKernel(name))
    monkeypatch.setattr(original, 'triton', SimpleNamespace(cdiv=lambda n, b: (n+b-1)//b))
    for experiment in original.experiments():
        for sweep in experiment['sweeps']:
            for variant in ('before', 'after'):
                captured.clear()
                inputs = original.generate_inputs(experiment['id'], sweep['id'])
                output = original.reference_output(experiment['id'], variant, inputs)
                original._launch(experiment['id'], variant, inputs, output)
                record = dict(arguments=['--experiment', experiment['id'], '--sweep', sweep['id'],
                                         '--variant', variant], sweep_parameters=sweep,
                              kind='timeline' if experiment in original._manifest()['timeline_probes'] else 'hypothesis')
                expected = audit.specializations(record)
                assert [name for name, kwargs in captured] == [item['kernel'] for item in expected]
                for (_, kwargs), item in zip(captured, expected):
                    actual = dict(kwargs)
                    # Default constexpr is deliberately absent from the original launch.
                    if item['kernel'] == 'a3_vector_probe':
                        actual.setdefault('volatile_loads', False)
                    assert actual == dict(item['constants'], **item['explicit_options'])


def test_summary_preserves_missing_and_failed_cells():
    spec = audit.specializations(cell())[0]
    plan = {'source_commit': 'frozen', 'cells': [{'id': 'one', 'specializations': ['key']},
                                               {'id': 'two', 'specializations': ['key2']}],
            'specializations': {'key': spec, 'key2': spec}}
    batch = {'specializations': {'key': {'status': 'compile_failed', 'error': 'compiler error'}}}
    report = audit.summarize(plan, [], batch)
    assert report['cells']['one']['compiled'][0]['error'] == 'compiler error'
    assert report['cells']['two']['compiled'][0]['status'] == 'not_compiled'
