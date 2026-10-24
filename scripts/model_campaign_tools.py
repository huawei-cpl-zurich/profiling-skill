#!/usr/bin/env python3
"""Private Model-task tool boundary for isolated optimization roles."""
import argparse
import json
import math
import re
from pathlib import Path
import shutil
import sys

from model_campaign_service import CampaignService, CampaignTools


def export(private, public):
    """Release constructed diagnostics/numeric rows, never oracle text or inputs."""
    value = json.loads((private / 'results/summary.json').read_text())
    result = {key: value[key] for key in ('state', 'failure_stage', 'operation',
              'source_sha256', 'physical_device', 'logical_device', 'diagnostic', 'progress', 'worker_pid',
              'runtime_initializations', 'torch_module_id', 'correctness_provenance') if key in value}
    checked = value.get('check', {})
    result['correctness'] = (checked.get('passed') is True
        and checked.get('variants') == value.get('required_variants', [0, 1]))
    # candidate_probe already authenticates candidate-owned compiler diagnostics.
    result['diagnostics'] = checked.get('diagnostics', [])
    measurement = value.get('result', {})
    result.update({key: measurement[key] for key in
                  ('latency', 'busy_time', 'cases', 'discovered_names', 'selector_mapping', 'warmup', 'samples',
                   'collector', 'granularity') if key in measurement})
    result['captures'] = []
    for capture in measurement.get('captures', []):
        selector = capture['selector']
        clean = dict(selector=selector)
        for field in ('basic_rows', 'pipe_rows'):
            clean[field] = []
            for row in capture.get(field, []):
                numeric = {}
                for key, item in row.items():
                    try:
                        number = float(item)
                        if math.isfinite(number):
                            numeric[key] = number
                    except (TypeError, ValueError):
                        if item == selector or item in ('N/A', 'NA', '-', ''):
                            numeric[key] = item
                        elif key == 'sub_block_id' and isinstance(item, str) \
                                and re.fullmatch(r'(cube|vector)[0-9]+', item):
                            numeric[key] = item
                clean[field].append(numeric)
        result['captures'].append(clean)
    public.mkdir(parents=True)
    (public / 'results').mkdir()
    (public / 'results/summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))


def prepare(output, task='matmul', reference_root=None, profile_index=None, check_indices=None):
    """Freeze a self-contained private worker and minimal public API bundle."""
    if task not in ('matmul', 'gdn', 'bsa'):
        raise ValueError('unsupported task')
    if task != 'matmul' and reference_root is None:
        raise ValueError('comparison tasks require --reference-root')
    source = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    public, private = output / 'public', output / 'private'
    public.mkdir()
    (private / 'scripts').mkdir(parents=True)
    helpers = private / 'skills/ascend-model-benchmarking/scripts'
    helpers.mkdir(parents=True)
    for name in ('model_campaign_worker.py', 'profile_device.py', 'a3_memory_batch.py',
                 'private_campaign_cases.py', 'private_campaign_models.py', 'model_campaign_native.py',
                 'model_device.py', 'phase_deadline.py', 'model_campaign_task.py', 'benchmark_correctness.py'):
        shutil.copyfile(source / 'scripts' / name, private / 'scripts' / name)
    for name in ('model_execution.py', 'candidate_probe.py', 'profile_capture.py', 'profile_accounting.py'):
        shutil.copyfile(source / 'skills/ascend-model-benchmarking/scripts' / name, helpers / name)
    shutil.copytree(source / 'third_party/cannbot_verifier',
                    private / 'third_party/cannbot_verifier',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'), copy_function=shutil.copyfile)
    for name in ('matmul', 'gdn', 'bsa'):
        folder = private / 'benchmarks' / name
        folder.mkdir(parents=True)
        for filename in ('baseline.py', 'cases.jsonl'):
            shutil.copyfile(source / 'benchmarks' / name / filename, folder / filename)
    shutil.copyfile(source / 'benchmarks/matmul/candidate.py', public / 'candidate.py')
    (public / 'task.json').write_text(json.dumps(dict(language='triton', task='matmul',
        single_kernel=False, public_files=['candidate.py', 'API.md', 'task.json'],
        api='Model.forward(lhs, rhs, bias) -> float32 tensor[M,N]')))
    (public / 'API.md').write_text('Implement torch.nn.Module Model.forward(lhs, rhs, bias).\n'
        'lhs: contiguous float16[M,K], rhs: contiguous float16[K,N], bias: float16[N].\n'
        'Return float32[M,N] equal to lhs.float() @ rhs.float() + bias.float().\n'
        'Support positive dimensions, including tails; do not mutate inputs.\n'
        'All arithmetic must use substantial Triton kernels, not torch computation.\n'
        'Multiple Triton launches allowed. Torch allocation and metadata views are allowed.\n')
    if task != 'matmul':
        import importlib
        adapter = importlib.import_module('comparison_' + task)
        metadata = adapter.prepare(private / 'reference', reference_root)
        released = metadata.get('public', metadata)
        starter = released['starter']
        if task == 'bsa':
            starter += '\nclass Model(torch.nn.Module):\n    forward = staticmethod(forward)\n'
        (public / 'candidate.py').write_text(starter)
        api = released.get('api')
        if api is None:
            api = ('Model.forward(' + ', '.join(released['arguments']) + ') -> '
                   + ', '.join(released['outputs']) + '\n'
                   'q,k: [1,T,Hk,K]; v: [1,T,Hv,V]; g,beta: [1,T,Hv].\n'
                   'initial_state: float32[N,Hv,K,V]; cu_seqlens: int32[N+1].\n'
                   'Hv is a multiple of Hk; each group uses the corresponding Q/K head.\n'
                   'Optional Q/K normalization divides by sqrt(sum(x*x)+1e-6).\n'
                   'For each token: decay state by exp(g); delta = beta*(v-k@state); '
                   'state += outer(k,delta); output = scale*q@state.\n'
                   'cu_seqlens gives independent packed sequence boundaries. '
                   'Return output and final state; do not mutate inputs.\n')
        (public / 'API.md').write_text(api + '\nAll computation must use substantial Triton kernels.\n')
        descriptor = dict(task=task, language='triton', single_kernel=False,
                          public_files=['candidate.py', 'API.md', 'task.json'])
        (public / 'task.json').write_text(json.dumps(descriptor))
        shutil.copyfile(source / 'scripts' / ('comparison_' + task + '.py'),
                        private / 'scripts' / ('comparison_' + task + '.py'))
        folder = private / 'benchmarks'
        shutil.copyfile(source / 'benchmarks' / ('comparison_' + task + '_starter.py'),
                        folder / ('comparison_' + task + '_starter.py'))
        count = len(adapter.specifications(private / 'reference'))
    else:
        count = 8
    profile_index = (7 if task == 'matmul' else 0) if profile_index is None else profile_index
    selected = list(range(8)) if task == 'matmul' else [0, count - 1]
    selected = selected if check_indices is None else list(check_indices)
    if not selected or any(i < 0 or i >= count for i in [profile_index, *selected]):
        raise ValueError('case index outside supplied task')
    (private / 'private-task.json').write_text(json.dumps(dict(task=task,
        profile_index=profile_index, check_indices=selected)))
    for path in public.iterdir():
        shutil.copyfile(path, private / path.name)
    (private / 'execute.py').write_text(
        'from pathlib import Path\nimport sys,json\nroot=Path(__file__).resolve().parent\n'
        'sys.path[:0]=[str(root/"scripts"),str(root/"skills/ascend-model-benchmarking/scripts")]\n'
        'import model_campaign_native as adapter\nfrom model_campaign_worker import execute\n'
        'result=execute(root,adapter)\nprint(json.dumps(result),flush=True)\n'
        'raise SystemExit(0 if result["state"]=="completed" else 1)\n')
    return dict(public=str(public), private=str(private))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ['prepare']:
        parser = argparse.ArgumentParser()
        parser.add_argument('prepare')
        parser.add_argument('--task', choices=('matmul', 'gdn', 'bsa'), default='matmul')
        parser.add_argument('--reference-root', type=Path)
        parser.add_argument('--profile-index', type=int)
        parser.add_argument('--check-indices', type=int, nargs='+')
        parser.add_argument('--output', type=Path, required=True)
        options = parser.parse_args(argv)
        print(json.dumps(prepare(options.output, options.task, options.reference_root, options.profile_index, options.check_indices)))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--target', default='bz-a3-1')
    parser.add_argument('--runtime', default='py311-torch')
    parser.add_argument('--devices', type=lambda value: [int(item) for item in value.split(',')],
                        help='comma-separated allowed physical IDs (for example 2,5)')
    parser.add_argument('verb', choices=('help', 'check', 'timing', 'pipes', 'observe', 'stop'))
    parser.add_argument('call_id', nargs='?')
    options = parser.parse_args(argv)
    if options.verb == 'help':
        print('experiment-tool check|timing|pipes; experiment-tool observe CALL_ID\n'
              'check always validates all configured cases/variants; changed sources also require full checks.\n'
              'timing/pipes may reuse a passed same-phase source proof, identified by correctness_provenance.\n'
              'Measurements always prepare and validate their profile case afresh.\n'
              'Compilation/numerical failures include safe diagnostics; retry edits until role deadline, no attempt quota.\n'
              'timing measures warm synchronized full-forward latency.\n'
              'pipes: Torch-NPU application PMU, per-task PipeUtilization; 20 warmups, 5 measured forwards.\n'
              'Not msprof op replay: no per-block distributions or inferred bandwidth/ceilings.\n'
              'Evidence is evidence/CALL_ID/results/summary.json. Preserve original paths in analysis reports.\n'
              'Resume nonterminal infrastructure jobs with observe; do not duplicate dispatch.\n'
              'All staging/device waits are included in execution.json deadline; no implicit budget extension.')
        return 0
    if options.verb == 'observe' and not options.call_id:
        parser.error('observe requires CALL_ID')
    private_workspace = options.workspace.parent / 'private-model-workspace'
    private_workspace.mkdir(exist_ok=True)
    tools = CampaignTools(options.bundle, private_workspace, options.target, options.runtime)
    if options.devices is not None:
        from model_device import device_allowlist
        tools.devices = device_allowlist(options.devices)
    # Match controller cleanup ownership and pending-call discovery exactly.
    tools.state = options.workspace.parent / 'tool-state'
    tools.state.mkdir(exist_ok=True)
    if options.verb == 'stop':
        from model_campaign_warm import shutdown_phase
        result = shutdown_phase(tools)
        print(json.dumps(result))
        return 0 if result['state'] == 'terminal' else 75
    service = CampaignService(tools, options.bundle, options.workspace, export)
    result = service.observe(options.call_id) if options.verb == 'observe' else service.start(
        'benchmark' if options.verb == 'timing' else options.verb)
    print(json.dumps(result, indent=2))
    return 1 if result.get('state') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
