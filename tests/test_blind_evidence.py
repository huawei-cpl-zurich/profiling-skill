import importlib.util
import json
from pathlib import Path
import tarfile

import pytest

SPEC = importlib.util.spec_from_file_location(
    'blind_evidence', Path(__file__).parents[1] / 'scripts/blind_evidence.py')
evidence = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evidence)


@pytest.mark.parametrize('state', ['completed', 'failed'])
def test_native_archive_extraction_keeps_diagnostics_not_trusted_host(tmp_path, state):
    root = tmp_path / 'remote'
    output = root / 'results'
    output.mkdir(parents=True)
    (root / 'task.json').write_text(json.dumps(dict(language='ascendc', native_wait_kind=2)))
    build = root / 'build' / 'source-hash'
    build.mkdir(parents=True)
    secret = 'TRUSTED_SEEDED_HOST_ORACLE_DO_NOT_PUBLISH'
    for name in ('native_host.asci', 'native_host.ll', 'native_host.mlir',
                 'native_host.s', 'native_host.bc', 'build.json', 'native_host.disasm'):
        (build / name).write_text(secret)
    # Neither a device-looking filename nor a clean-looking IR body proves
    # its translation-unit role. Keep it remote until that provenance exists.
    (build / 'candidate-vec.ll').write_text('define void @task_kernel() {}')
    compiler = root / 'compiler-evidence' / 'candidate'
    compiler.mkdir(parents=True)
    (compiler / 'device.mlir').write_text('module { func.func @task_kernel() }')
    (compiler / 'host.ll').write_text(secret)
    (compiler / 'manifest.json').write_text(secret)
    (build / 'build.stderr').write_text('candidate.asc:7: error: unknown vector operation')
    (build / 'build.stdout').write_text('compiler invocation completed')
    (output / 'check.stderr').write_text('native compilation failed; see build.stderr')
    (output / 'profile.csv').write_text('Kernel Name,Duration(us)\ntask_kernel,2\n')
    summary = dict(state=state)
    archive = evidence.compact_evidence(root, output, summary)
    downloaded = tmp_path / 'agent-evidence'
    with tarfile.open(archive) as stream:
        stream.extractall(downloaded, filter='data')
    files = {str(p.relative_to(downloaded)): p.read_text() for p in downloaded.rglob('*') if p.is_file()}
    assert set(files) == {'results/summary.json', 'results/check.stderr', 'results/profile.csv',
                          'build/source-hash/build.stderr', 'build/source-hash/build.stdout'}
    assert all(secret not in body for body in files.values())
    assert 'error: unknown vector operation' in files['build/source-hash/build.stderr']
    exported = json.loads(files['results/summary.json'])
    assert exported['state'] == state
    assert exported['compiler_evidence']['available'] is False
    assert exported['compiler_evidence']['role'] == 'unproven_native_translation_unit'
    assert (build / 'native_host.asci').read_text() == secret  # retained remotely
    assert (compiler / 'device.mlir').exists()


def test_triton_kernel_compiler_evidence_still_retrievable(tmp_path):
    output = tmp_path / 'results'
    output.mkdir()
    (tmp_path / 'task.json').write_text(json.dumps(dict(language='triton')))
    compiler = tmp_path / 'compiler-evidence' / 'candidate'
    compiler.mkdir(parents=True)
    (compiler / 'kernel.mlir').write_text('module { func.func @candidate_kernel() }')
    archive = evidence.compact_evidence(tmp_path, output, dict(state='completed'))
    with tarfile.open(archive) as stream:
        assert stream.extractfile('compiler-evidence/candidate/kernel.mlir').read().startswith(b'module')
