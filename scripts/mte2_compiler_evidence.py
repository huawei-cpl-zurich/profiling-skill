#!/usr/bin/env python3
"""Capture real compiler consumers with the managed helper, not a timing run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import a3_mlir_retrospective as adapter


def file_record(path, kind, stage=None):
    path = Path(path)
    data = path.read_bytes()
    result = {'path': str(path), 'kind': kind, 'bytes': len(data),
              'sha256': hashlib.sha256(data).hexdigest()}
    if stage:
        result['stage'] = stage
    # Compiler outputs may be binary. Do not decode arbitrary object files.
    if kind != 'compiler_output' and path.suffix != '.bc':
        try:
            text = data.decode('utf-8')
        except UnicodeDecodeError:
            return result
        if '\0' not in text:
            result['operations'] = {key: value[:20] for key, value in adapter.inspect_ir(text).items()}
    return result


def log_record(value):
    if not value or not Path(value).is_file():
        return {'path': value, 'missing': True}
    record = file_record(value, 'log')
    record.pop('operations', None)
    # Complete error text stays on disk; compact summaries point to it if large.
    text = Path(value).read_text(errors='replace')
    record['text'] = text[:8192]
    record['truncated'] = len(text) > 8192
    return record


def inspect(manifest_path, assembly_root=None):
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    errors, artifacts, invocations = [], [], []
    real_compiles = 0
    query_flags = {'--help', '-h', '--version', '-version', '-print-targets', '--print-targets'}
    if manifest['reference_exit_code'] != 0:
        errors.append('reference command failed')
    if not manifest.get('invocations'):
        errors.append('no actual compiler consumer invocation captured')
    for invocation in manifest.get('invocations', []):
        argv = invocation.get('argv', [])
        query = len(argv) == 2 and argv[1] in query_flags
        item = {key: invocation.get(key) for key in
                ('id', 'tool', 'argv', 'real_path', 'exit_code', 'duration_seconds')}
        item['kind'] = 'tool_query' if query else 'compiler'
        item.update({key: log_record(invocation.get(key)) for key in ('stdout', 'stderr')})
        invocations.append(item)
        if invocation.get('exit_code') != 0:
            errors.append('compiler failed: ' + str(invocation.get('id')))
        for key, kind in (('input_copy', 'compiler_input'), ('output_copy', 'compiler_output')):
            value = invocation.get(key)
            if value and Path(value).is_file():
                artifacts.append(file_record(value, kind))
                if key == 'input_copy' and not query:
                    real_compiles += 1
            elif key == 'input_copy' and not query:
                errors.append('missing actual compiler input: ' + str(invocation.get('id')))
    if not real_compiles:
        errors.append('no actual compiler invocation with a retained input')
    # Raw dumps preserve available stages beyond the helper's convenience adapter copy.
    roots = [manifest_path.parent / 'triton-dump']
    if assembly_root:
        roots.append(Path(assembly_root))
    for root in roots:
        if root.is_dir():
            for path in sorted(root.rglob('*')):
                if path.is_file() and path.suffix in ('.mlir', '.ll', '.ttir', '.ttgir', '.txt'):
                    artifacts.append(file_record(path, 'runtime_assembly', path.stem))
    for value in manifest.get('triton_dumps', []):
        if Path(value).is_file():
            artifacts.append(file_record(value, 'adapter_ir', 'ttadapter'))
    later = any(item.get('stage', '').lower() in ('llir', 'llvm', 'kernel.ll', 'ptx', 'amdgcn')
                or Path(item['path']).suffix == '.ll' for item in artifacts)
    runtime = Path(assembly_root) / 'runtime.json' if assembly_root else None
    runtime_metadata = ({'file': file_record(runtime, 'runtime_metadata'),
                         'value': json.loads(runtime.read_text())}
                        if runtime and runtime.is_file() else None)
    return {'schema_version': 1, 'status': 'failed' if errors else 'captured',
            'errors': errors, 'capture_manifest': file_record(manifest_path, 'manifest'),
            'reference_exit_code': manifest['reference_exit_code'],
            'reference_stdout': log_record(manifest.get('reference_stdout')),
            'reference_stderr': log_record(manifest.get('reference_stderr')),
            'real_tools': manifest.get('real_tools'), 'wrappers': manifest.get('wrappers'),
            'invocations': invocations, 'artifacts': artifacts,
            'runtime_metadata': runtime_metadata,
            'later_artifact_status': 'observed_not_hardware_schedule' if later else 'not_observed',
            'limitation': 'IR operations are not executed instruction counts or pipe assignments; '
                          'early IR does not establish final synchronization or hardware scheduling.'}


def capture(helper, command, output, sources, identity, assembly_root=None):
    if not command or not sources:
        raise ValueError('command and at least one workload source are required')
    if not all(key in identity for key in ('target', 'specialization', 'options')):
        raise ValueError('identity must explicitly name target, specialization and options')
    before = [file_record(path, 'workload_source') for path in sources]
    output = Path(output).resolve()
    isolation = adapter.isolated_capture(helper, command, output)
    (output / 'isolation.json').write_text(json.dumps(isolation, indent=2) + '\n')
    result = inspect(isolation['capture_manifest'], assembly_root)
    result.update(command=command, identity=identity, identity_provenance='caller_declared',
                  source_files=before, helper=file_record(helper, 'capture_helper'),
                  copied_tools_sha256=isolation['copied_tools_sha256'])
    implementation = Path(helper).with_name('compiler_capture_helper.py')
    if implementation.is_file():
        result['capture_implementation'] = file_record(implementation, 'helper_implementation')
    if before != [file_record(path, 'workload_source') for path in sources]:
        result['errors'].append('workload source changed during capture')
        result['status'] = 'failed'
    result['measurement_policy'] = ('Forced compile/debug acquisition is separate from timing. '
        'Match source, specialization, options and target before comparing; compare output hashes '
        'when available rather than assuming debug and timed binaries are identical.')
    (output / 'evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='verb', required=True)
    run = sub.add_parser('capture')
    run.add_argument('--helper', required=True)
    run.add_argument('--source', action='append', required=True)
    run.add_argument('--identity', required=True, help='JSON with target/specialization/options')
    run.add_argument('--output', required=True, help='fresh job-owned directory')
    run.add_argument('--assembly-root')
    run.add_argument('command', nargs=argparse.REMAINDER)
    view = sub.add_parser('inspect')
    view.add_argument('--manifest', required=True)
    view.add_argument('--assembly-root')
    view.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.verb == 'capture':
        command = args.command[1:] if args.command[:1] == ['--'] else args.command
        result = capture(args.helper, command, args.output, args.source,
                         json.loads(Path(args.identity).read_text()), args.assembly_root)
    else:
        result = inspect(args.manifest, args.assembly_root)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'errors': result['errors']}))
    return 0 if result['status'] == 'captured' else 2


if __name__ == '__main__':
    raise SystemExit(main())
