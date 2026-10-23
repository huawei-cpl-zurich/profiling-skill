#!/usr/bin/env python3
"""One correctness validation and ten profiled forwards; reusable private baseline cache."""
import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

from model_benchmark import load, identity, case_identity, HELPERS
sys.path.insert(0, str(HELPERS))
from model_execution import GraphCompiler, prepare_model
from profile_ten import capture_ten, resolve_triton_names

PROTOCOL = 'single-validation-ten-profile-v2'


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def baseline_evidence(cache, folder, result):
    """Bind referenced reports, not unreferenced vendor raw-data trees."""
    paths = [folder / name for name in ('latency.json', 'forward-count.json', 'result.json')
             if (folder / name).exists()]
    profile = result['profile']
    paths += [Path(profile[key]) for key in ('vendor_trace', 'capture_path', 'accounting_path')
              if profile.get(key)]
    evidence = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(cache.resolve()) or not resolved.is_file():
            raise ValueError('baseline evidence must be an existing file within private cache')
        evidence[str(resolved.relative_to(cache.resolve()))] = sha(resolved)
    return evidence


def verify_baseline_evidence(cache, item):
    evidence = item.get('baseline_evidence')
    if not isinstance(evidence, dict):
        raise ValueError('baseline evidence manifest missing')
    for relative, digest in evidence.items():
        path = (cache / relative).resolve()
        if not path.is_relative_to(cache.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError('baseline evidence missing or changed: ' + relative)


def move_tree(value, torch, device):
    """Private data cache preserves nested validator state, never arbitrary objects."""
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise ValueError('private case cache supports strided tensors only')
        # Moving a sliced tensor directly can compact its stride/offset. Move
        # its storage instead, then reconstruct the original logical view.
        elements = value.untyped_storage().nbytes() // value.element_size()
        storage = value.detach().as_strided((elements,), (1,), storage_offset=0)
        return storage.to(device, copy=True).as_strided(
            value.shape, value.stride(), value.storage_offset()).requires_grad_(value.requires_grad)
    if isinstance(value, (tuple, list)):
        return type(value)(move_tree(item, torch, device) for item in value)
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: move_tree(item, torch, device) for key, item in value.items()}
    if value is None or type(value) in (bool, int, float, str) or isinstance(value, (torch.dtype, torch.device)):
        return value
    raise ValueError('cached cases require nested tensors and primitive metadata')


def measured(call, output, capture, expected=None):
    count = 0
    def forward():
        nonlocal count
        count += 1
        return call()
    result = capture(forward, output, expected_kernel=expected)
    events = result['event_latency']
    if count != 10 or events.get('status') != 'complete' or events.get('count') != 10:
        raise RuntimeError('ten successful profiled forwards/event samples required')
    return result


def run(options, *, torch=None, compiler_factory=None, prepare=prepare_model, capture=capture_ten):
    """Hooks and cache are trusted/private; candidate source may vary on reuse."""
    output, cache = Path(options.output), Path(options.cache)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(state='failed', protocol=PROTOCOL, timing_kind='profiled-device-events', cases=[])
    stack = ExitStack()
    try:
        if options.max_specializations < 1:
            raise ValueError('max-specializations must be positive')
        if torch is None:
            import torch
            import torch_npu  # Register the native runtime; device lease is external.
        # Fullgraph specialization is expected here, not an accidental recompile
        # loop. Scope the budget so subsequent users retain their own settings.
        config = torch._dynamo.config
        limits = {}
        for names in (('recompile_limit', 'cache_size_limit'),
                      ('accumulated_recompile_limit', 'accumulated_cache_size_limit')):
            key = next((name for name in names if hasattr(config, name)), None)
            if key:
                limits[key] = max(getattr(config, key), options.max_specializations + 1)
        stack.enter_context(config.patch(limits))
        hooks = {name: load(getattr(options, name)) for name in ('baseline', 'candidate', 'cases_provider', 'validator')}
        sources = {name: identity(hook) for name, hook in hooks.items()}
        report['sources'] = sources
        report['comparability'] = dict(comparable=False,
            reason='verify physical device, runtime and clock/load conditions across retained cache and candidate')
        shared = {name: value for name, value in sources.items() if name != 'candidate'}
        candidate_file = Path(inspect.getsourcefile(hooks['candidate'])).resolve()
        if any(Path(inspect.getsourcefile(hooks[name])).resolve() == candidate_file for name in shared):
            raise ValueError('candidate must use a separate source file from baseline, cases-provider and validator')
        compiler_factory = compiler_factory or (lambda: GraphCompiler(backend_name=options.backend))
        if options.cache_mode == 'create':
            cache.mkdir(parents=True, exist_ok=False)
            cases = iter(hooks['cases_provider']())
            manifest = dict(protocol=PROTOCOL, sources=shared, state='building', cases=[])
            save(cache / 'manifest.json', manifest)
        else:
            manifest = json.loads((cache / 'manifest.json').read_text())
            if manifest['protocol'] != PROTOCOL or manifest['sources'] != shared or manifest['state'] != 'complete':
                raise ValueError('baseline cache protocol/source identity mismatch or incomplete cache')
            # Verify all retained baseline evidence before any candidate execution.
            for item in manifest['cases']:
                verify_baseline_evidence(cache, item)
            def cached_cases():
                for item in manifest['cases']:
                    path = cache / item['file']
                    if not path.resolve().is_relative_to(cache.resolve()) or sha(path) != item['sha256']:
                        raise ValueError('private case cache changed')
                    case = torch.load(path, map_location='cpu', weights_only=True)
                    yield move_tree(case, torch, options.device)
            cases = cached_cases()
        report['counts'] = dict(expected_provider_invocations=int(options.cache_mode == 'create'),
            warmup_forwards=0, discovery_forwards=0, unprofiled_timing_forwards=0)
        seen = set()
        for index, case in enumerate(cases):
            if index >= options.max_specializations:
                raise ValueError('case count exceeds declared specialization budget')
            if not isinstance(case['id'], str) or not case['id'] or case['id'] in seen:
                raise ValueError('provide nonempty cases with unique identifiers')
            seen.add(case['id'])
            if 'expected' not in case:
                raise ValueError('case provider must supply expected output from one reference generation')
            folder = output / f'case-{index:04d}'; folder.mkdir()
            specialization = case_identity(case, case.get('expected_kernel', []))
            row = dict(id=case['id'], state='incomplete', specialization_sha256=specialization['sha256'])
            report['cases'].append(row)
            if options.cache_mode == 'reuse':
                item = manifest['cases'][index]
                path = cache / f'baseline-{index:04d}.json'
                if sha(path) != item['baseline_sha256'] or specialization['sha256'] != item['specialization_sha256']:
                    raise ValueError('baseline profile or case specialization changed')
                baseline_result = json.loads(path.read_text())
            if options.cache_mode == 'create':
                payload = cache / f'case-{index:04d}.pt'
                torch.save(move_tree(case, torch, 'cpu'), payload)
                manifest['cases'].append(dict(id=case['id'], file=payload.name, sha256=sha(payload),
                    specialization_sha256=specialization['sha256']))
                save(cache / 'manifest.json', manifest)
                # Separate expected/argument storage before the sole validation;
                # a candidate cannot mutate an aliased expected result.
                case = move_tree(torch.load(payload, map_location='cpu', weights_only=True),
                                 torch, options.device)
            validations = 0
            def validate(result):
                nonlocal validations
                validations += 1
                hooks['validator'](case, 'candidate', result)
            prepared = prepare(
                lambda: hooks['candidate'](*case.get('init_args', ()), **case.get('init_kwargs', {})),
                case['args'], case.get('kwargs', {}), validate,
                compiler=compiler_factory(), synchronize=torch.npu.synchronize)
            if validations != 1 or prepared.compiler.executions != 1:
                raise RuntimeError('require exactly one compiled correctness forward')
            before = prepared.compiler.snapshot()
            expected = case.get('expected_kernel') or resolve_triton_names(prepared.compiler.graphs)
            with torch.inference_mode(), prepared.compiler.monitor_compilation():
                candidate = measured(prepared, folder / 'candidate', capture, expected)
            prepared.evidence['custom_kernel_identity_verified'] = (
                candidate['kernel_identity']['status'] == 'complete')
            if prepared.compiler.snapshot() != before or prepared.compiler.executions != 11:
                raise RuntimeError('compilation or missing compiled execution during measurement')
            baseline_folder = cache / f'baseline-{index:04d}'
            if options.cache_mode == 'create':
                baseline_case = move_tree(torch.load(payload, map_location='cpu', weights_only=True),
                                          torch, options.device)
                baseline = hooks['baseline'](*baseline_case.get('init_args', ()), **baseline_case.get('init_kwargs', {}))
                if hasattr(baseline, 'eval'):
                    baseline.eval()
                with torch.inference_mode():
                    baseline_result = measured(lambda: baseline(*baseline_case['args'], **baseline_case.get('kwargs', {})),
                                               baseline_folder, capture)
                save(cache / f'baseline-{index:04d}.json', baseline_result)
                manifest['cases'][index]['baseline_sha256'] = sha(cache / f'baseline-{index:04d}.json')
                manifest['cases'][index]['baseline_evidence'] = baseline_evidence(
                    cache, baseline_folder, baseline_result)
                save(cache / 'manifest.json', manifest)
            row.update(state='complete', candidate=candidate, baseline=baseline_result,
                baseline_reused=options.cache_mode == 'reuse', execution=prepared.evidence,
                counts=dict(candidate_correctness_forwards=validations, candidate_profiled_forwards=10,
                            baseline_profiled_forwards=10 if options.cache_mode == 'create' else 0))
            # Profiles/identity may be unavailable despite ten valid event samples.
            # No verified speedup/ranking is inferred from event completeness.
            save(output / 'report.json', report)
            del prepared
        if not seen:
            raise ValueError('provide at least one case')
        if options.cache_mode == 'create':
            manifest['state'] = 'complete'; save(cache / 'manifest.json', manifest)
        report['profile_state'] = 'complete' if all(
            row[arm]['profile']['status'] == 'complete'
            for row in report['cases'] for arm in ('candidate', 'baseline')) else 'incomplete'
        report['candidate_identity_state'] = 'verified' if all(
            row['candidate']['kernel_identity']['status'] == 'complete'
            for row in report['cases']) else 'qualified'
        report['state'] = 'complete'
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
    finally:
        stack.close()
        save(output / 'report.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('baseline', 'candidate', 'cases-provider', 'validator', 'output', 'cache'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--cache-mode', choices=('create', 'reuse'), required=True)
    parser.add_argument('--backend', choices=('torchair-ge', 'npu-inductor'), default='npu-inductor')
    parser.add_argument('--max-specializations', type=int, default=256, help='scoped Dynamo specialization budget')
    parser.add_argument('--device', default='npu:0', help='logical device; supervisor owns physical selection')
    options = parser.parse_args(argv)
    try:
        report = run(options)
    except OSError as error:
        print(json.dumps(dict(state='failed', error=str(error))), file=sys.stderr)
        return 2
    print(json.dumps(dict(state=report['state'], report=str(Path(options.output) / 'report.json'))))
    return 0 if report['state'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
