#!/usr/bin/env python3
"""Compile-only reconstruction of recorded A3 probes; never launch old kernels.

Plans describe the source launch contract, not recovered JIT cache metadata.
Recompilations remain unverified until exact retained IR hashes match. Compiler
targets and specialization attributes must be supplied explicitly by the caller.
"""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import gzip
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def specializations(cell, manifest=None):
    """Translate the frozen source's _launch dispatch, without importing Triton."""
    args = dict(zip(cell['arguments'][::2], cell['arguments'][1::2]))
    exp, after = args['--experiment'], args['--variant'] == 'after'
    parameters = cell.get('sweep_parameters')
    if parameters is None:
        if manifest is None:
            raise ValueError('amplified dispatch requires frozen hypothesis manifest')
        hypothesis = next(item for item in manifest['hypotheses'] if item['id'] == exp)
        parameters = next(item for item in hypothesis['sweeps'] if item['id'] == args['--sweep'])
    n, work = (parameters[key] for key in ('n', 'work'))
    def spec(kernel, pointers, constants, options=None):
        signature = {name: dtype for name, dtype in pointers}
        return {'kernel': kernel, 'signature': signature, 'constants': constants,
                'explicit_options': options or {},
                'provenance': 'unverified_reconstruction',
                'uncertainty': 'original target/options/alignment metadata not retained'}
    vector = [('x', '*fp32'), ('selected', '*i1'), ('work_ptr', '*i64'), ('output', '*fp32')]
    if '--repetitions' in args:
        mode = {'scalar-mask-decision': (2, 3), 'load-masking': (7, 4),
                'sparse-mode-control': (5, 6)}[exp][after]
        # repetitions is a runtime pointer value, not a compile constant.
        return [spec('a3_amplified_probe', vector + [('repeat_ptr', '*i32')],
                     {'n': n, 'work': work, 'mode': mode})]
    if exp in ('row-reduction-vs-dot', 'query-reuse'):
        pointers = [(key, '*fp32') for key in ('lhs', 'rhs', 'output')]
        if exp == 'row-reduction-vs-dot' and not after:
            return [spec('a3_row_reduce_probe', pointers, {'size': n})]
        rows = (n if exp == 'row-reduction-vs-dot' else 4) if after else 1
        return [spec('a3_dot_probe', pointers, {'size': n, 'rows_per_program': rows})]
    if exp == 'head-rank':
        return [spec('a3_head_rank_probe', [('x', '*fp32'), ('flags', '*i32'),
                     ('rank_ptr', '*i64'), ('output', '*fp32')],
                     {'n': n, 'heads': work, 'block': 128, 'direct': after})]
    if exp == 'key-tile-size':
        return [spec('a3_key_tile_probe', [('x', '*fp32'), ('output', '*fp32')],
                     {'n': n, 'block': 128, 'key_tile': 128 if after else 64})]
    if exp == 'empty-tile-skip' and after:
        return [spec('a3_empty_skip_probe', [('x', '*fp32'), ('selected', '*i1'),
                     ('output', '*fp32')], {'n': n, 'block': 128})]
    if cell['kind'] == 'timeline':
        if exp == 'fused-vs-split' and after:
            return [spec(name, [('x', '*fp32'), ('output', '*fp32')],
                         {'n': n, 'block': 128})
                    for name in ('a3_phase_copy', 'a3_phase_compute', 'a3_phase_store')]
        phase = {'reversed-phases': int(after), 'repeated-phases': 2 * after,
                 'serialized-vs-overlapped': 3 if after else 4}.get(exp, 0)
        loops = work * (4 if exp == 'duration-scaled-phases' and after else 1)
        return [spec('a3_phase_probe', [(key, '*fp32') for key in ('x', 'scratch', 'output')],
                     {'n': n, 'block': 128, 'compute_loops': loops, 'phase_mode': phase})]
    mode = {'loop-bound': 1, 'scalar-mask-decision': 3 if after else 2,
            'load-masking': 4 if after else 7, 'sparse-mode-control': 6 if after else 5,
            'empty-tile-skip': 6}.get(exp, 0)
    options = {'num_warps': 8 if after else 4, 'num_stages': 2 if after else 1} \
        if exp == 'schedule-knobs' else {}
    return [spec('a3_vector_probe', vector,
                 {'n': n, 'work': work, 'block': 128, 'mode': mode,
                  'loops': (work if after else 256) if exp == 'loop-bound' else 1,
                  'volatile_loads': False}, options)]


def make_plan(run, manifest=None):
    cells, unique = [], {}
    for pair in run['pairs']:
        for cell in pair['cells']:
            specs = specializations(cell, manifest)
            if [item['kernel'] for item in specs] != cell['kernel_names']:
                raise ValueError(f"dispatch selector mismatch: {cell['id']}")
            ids = []
            for item in specs:
                key = digest(item)
                unique[key] = item
                ids.append(key)
            cells.append({'id': pair['id'] + '/' + cell['variant'], 'specializations': ids})
    if len(cells) != run['expected_cells'] or len({cell['id'] for cell in cells}) != len(cells):
        raise ValueError('cell coverage mismatch or duplicate identity')
    return {'source_commit': run['source_commit'], 'cells': cells, 'specializations': unique}


def verify_original(path, expected):
    data = Path(path).read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ValueError(f'original artifact hash mismatch: {path}')
    return {'path': str(path), 'sha256': actual, 'bytes': len(data),
            'provenance': 'original_retained_artifact'}


def original_requests(run, pairs_root):
    """Select only named discovery artifacts, never recursively scan raw profiles."""
    requests = []
    for pair in run['pairs']:
        for cell in pair['cells']:
            if cell['kind'] != 'amplified':
                continue
            evidence_path = Path(pairs_root) / pair['id'] / cell['variant'] / 'discovery-attempt-0.json'
            evidence = json.loads(evidence_path.read_text())
            artifacts = evidence['normal_run']['workload_result']['compiler']
            prefix = hashlib.sha256((evidence['dispatch_key'] + '\0basic').encode()).hexdigest()[:20]
            for artifact in artifacts:
                requests.append(dict(artifact, cell=pair['id'] + '/' + cell['variant'],
                                     retained_prefix=prefix))
    return requests


def inspect_ir(text):
    """Index actual operation lines, not estimates of machine-pipe usage."""
    groups = {'loops': ('scf.for', 'scf.while', 'cf.cond_br'),
              'loads': ('tt.load', 'memref.load', 'vector.transfer_read'),
              'stores': ('tt.store', 'memref.store', 'vector.transfer_write'),
              'matrix': ('tt.dot', 'hivm.hir.mmad', 'linalg.matmul'),
              'synchronization': ('set_flag', 'wait_flag', 'barrier', 'sync_block'),
              'copies': ('hivm.hir.load', 'hivm.hir.store', 'memref.copy')}
    return {group: [{'line': n, 'text': line.strip()} for n, line in enumerate(text.splitlines(), 1)
                    if any(token in line for token in tokens)]
            for group, tokens in groups.items()}


def import_originals(records, output):
    """Materialize only hash-verified compact text evidence from retained jobs."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for record in records:
        if record['status'] != 'verified_original':
            results.append(record)
            continue
        data = gzip.decompress(base64.b64decode(record['gzip_base64'], validate=True))
        if hashlib.sha256(data).hexdigest() != record['sha256'] or len(data) != record['bytes']:
            raise ValueError('transferred original hash or length mismatch')
        path = output / (record['sha256'] + '.mlir')
        path.write_bytes(data)
        results.append({key: value for key, value in record.items() if key != 'gzip_base64'} |
                       {'local_path': str(path), 'provenance': 'original_retained_artifact',
                        'operations': inspect_ir(data.decode())})
    return results


@contextmanager
def forbid_launch(jit_class):
    """A runtime guard: accidental warmup/run is a hard error, not a launch."""
    saved = {name: getattr(jit_class, name) for name in ('run', 'warmup')}
    def forbidden(*args, **kwargs):
        raise RuntimeError('kernel execution is forbidden in retrospective audit')
    try:
        for name in saved:
            setattr(jit_class, name, forbidden)
        yield
    finally:
        for name, value in saved.items():
            setattr(jit_class, name, value)


def compile_one(spec, config, source, output):
    import triton
    from triton.compiler import ASTSource
    from triton.compiler.compiler import AttrsDescriptor
    from triton.backends.compiler import GPUTarget
    from triton.runtime.jit import JITFunction
    sys.path.insert(0, str(Path(source) / 'benchmarks'))
    module = 'a3_profile_amplified' if spec['kernel'] == 'a3_amplified_probe' else 'a3_profile_microbench'
    fn = getattr(importlib.import_module(module), spec['kernel'])
    target = GPUTarget(**config['target'])
    options = dict(config['options'], **spec['explicit_options'])
    # attrs is caller-provided compiler metadata, never silently reconstructed.
    attrs_config = config['attrs']
    if attrs_config == 'aligned-pointers-reconstruction':
        attrs_config = {'cls': 'AscendAttrsDescriptor', 'arg_properties': {
            'tt.divisibility': [fn.arg_names.index(name) for name in spec['signature']]}}
    attrs = AttrsDescriptor.from_dict(attrs_config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with forbid_launch(JITFunction):
        compiled = triton.compile(ASTSource(fn, signature=spec['signature'],
                                            constants=spec['constants'], attrs=attrs),
                                  target=target, options=options)
    artifacts = []
    for stage, text in compiled.asm.items():
        if not isinstance(text, str):
            continue
        path = output / (stage + '.txt')
        path.write_text(text)
        artifacts.append({'stage': stage, 'path': str(path),
                          'sha256': hashlib.sha256(text.encode()).hexdigest(),
                          'operations': inspect_ir(text)})
    return {'provenance': 'unverified_reconstruction', 'spec': spec, 'config': config,
            'effective_attrs': attrs_config,
            'triton_module_version': triton.__version__, 'artifacts': artifacts,
            'limitation': 'exact source dispatch; original specialization metadata not recovered'}


def isolated_capture(helper, command, output):
    """Copy runtime Python and selected executable consumers before shimming.

    Non-Python runtime resources are read-only symlinks. Every selected compiler
    executable itself is copied, and the helper only replaces job-owned paths.
    No TRITON_*_PATH overrides or shared compiler directory mutations are used.
    """
    import triton
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    package = Path(triton.__file__).parent
    clone = output / 'runtime' / 'triton'
    def resource(source, destination):
        if source.endswith(('.py', '.pyi')):
            return shutil.copy2(source, destination)
        os.symlink(str(Path(source).resolve()), destination)
        return destination
    shutil.copytree(package, clone, copy_function=resource,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    env = os.environ.copy()
    env['PYTHONPATH'] = str(clone.parent) + os.pathsep + env.get('PYTHONPATH', '')
    owned_bin = output / 'tools' / 'bin'
    owned_bin.mkdir(parents=True)
    tools = {}
    for name in ('bishengir-compile', 'bishengir-compile-a5', 'hivmc', 'hivmc-a5', 'bisheng'):
        # The backend prefers its bundled compiler over PATH.
        bundled = package / 'backends' / 'ascend' / 'bishengir' / 'bin' / name
        visible = str(bundled) if bundled.is_file() else shutil.which(name)
        if visible is None:
            continue
        destination = (clone / bundled.relative_to(package)) if bundled.is_file() else owned_bin / name
        if destination.is_symlink():
            destination.unlink()
        shutil.copy2(visible, destination)
        tools[name] = str(destination)
    env['PATH'] = str(owned_bin) + os.pathsep + env['PATH']
    argv = ['bash', str(helper), '--output-dir', str(output / 'capture')]
    for name, path in tools.items():
        argv += ['--wrap', name, '--real-tool', name + '=' + path]
    if not tools:
        raise RuntimeError('no compiler consumers visible after runtime activation')
    for path in tools.values():
        if not Path(path).resolve().is_relative_to(output):
            raise RuntimeError('capture would shim an unowned compiler')
    before = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in tools.items()}
    result = subprocess.run(argv + ['--'] + command, env=env, check=False)
    return {'exit_code': result.returncode, 'copied_tools_sha256': before,
            'capture_manifest': str(output / 'capture' / 'capture-manifest.json')}


def summarize(plan, originals, batch=None):
    original_cells = {}
    for item in originals:
        original_cells.setdefault(item['cell'], []).append({
            key: item[key] for key in ('stage', 'sha256', 'bytes', 'provenance', 'operations')
            if key in item})
    results = {}
    for cell in plan['cells']:
        compiled = []
        for key in cell['specializations']:
            result = (batch or {}).get('specializations', {}).get(key, {})
            compiled.append({'key': key, 'kernel': plan['specializations'][key]['kernel'],
                             'status': result.get('status', 'not_compiled'),
                             'provenance': result.get('provenance', 'unverified_reconstruction'),
                             'error': result.get('error'),
                             'artifacts': result.get('artifacts', [])})
        results[cell['id']] = {'originals': original_cells.get(cell['id'], []), 'compiled': compiled}
    return {'source_commit': plan['source_commit'], 'cells': results,
            'limitation': 'operation lines are compiler evidence, not counts of executed instructions or pipe time'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--run', required=True)
    plan.add_argument('--manifest', required=True)
    plan.add_argument('--output', required=True)
    for verb in ('compile', 'compile-all'):
        compile_parser = sub.add_parser(verb)
        for name in ('plan', 'config', 'source', 'output'):
            compile_parser.add_argument('--' + name, required=True)
        if verb == 'compile':
            compile_parser.add_argument('--key', required=True)
    originals = sub.add_parser('import-originals')
    originals.add_argument('--records', required=True)
    originals.add_argument('--output', required=True)
    capture = sub.add_parser('capture')
    capture.add_argument('--helper', required=True)
    capture.add_argument('--output', required=True)
    capture.add_argument('argv', nargs=argparse.REMAINDER)
    summary = sub.add_parser('summarize')
    for name in ('plan', 'originals', 'output'):
        summary.add_argument('--' + name, required=True)
    summary.add_argument('--batch')
    args = parser.parse_args()
    if args.command == 'plan':
        result = make_plan(json.loads(Path(args.run).read_text()), json.loads(Path(args.manifest).read_text()))
        source = Path(args.manifest).resolve().parents[1]
        result['source_files_sha256'] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ('benchmarks/a3_profile_microbench.py', 'benchmarks/a3_profile_amplified.py',
                         'benchmarks/compile_evidence.py', 'experiments/a3-profile-hypotheses.json')}
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    elif args.command == 'import-originals':
        result = import_originals(json.loads(Path(args.records).read_text()), args.output)
        (Path(args.output) / 'originals.json').write_text(json.dumps(result, indent=2) + '\n')
    elif args.command == 'capture':
        command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
        result = isolated_capture(args.helper, command, args.output)
        (Path(args.output) / 'isolation.json').write_text(json.dumps(result, indent=2) + '\n')
        return result['exit_code']
    elif args.command == 'summarize':
        result = summarize(json.loads(Path(args.plan).read_text()),
                           json.loads(Path(args.originals).read_text()),
                           json.loads(Path(args.batch).read_text()) if args.batch else None)
        Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    else:
        plan = json.loads(Path(args.plan).read_text())
        for name, expected in plan['source_files_sha256'].items():
            if hashlib.sha256((Path(args.source) / name).read_bytes()).hexdigest() != expected:
                raise ValueError('frozen source changed: ' + name)
        config = json.loads(Path(args.config).read_text())
        if args.command == 'compile':
            result = compile_one(plan['specializations'][args.key], config, args.source, args.output)
            (Path(args.output) / 'audit.json').write_text(json.dumps(result, indent=2) + '\n')
        else:
            results = {}
            for key, spec in plan['specializations'].items():
                try:
                    result = compile_one(spec, config, args.source, Path(args.output) / key)
                    result['status'] = 'compiled'
                except Exception as exc:
                    result = {'status': 'compile_failed', 'error': str(exc), 'spec': spec}
                results[key] = result
                Path(args.output).mkdir(parents=True, exist_ok=True)
                (Path(args.output) / 'batch.json').write_text(json.dumps({
                    'cells': plan['cells'], 'specializations': results}, indent=2) + '\n')
                print(json.dumps({'specialization': key, 'kernel': spec['kernel'],
                                  'status': result['status']}), flush=True)
            return int(any(item['status'] != 'compiled' for item in results.values()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
