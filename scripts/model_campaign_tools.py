#!/usr/bin/env python3
"""Private Model-task tool boundary for isolated optimization roles."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from model_campaign_service import CampaignService, CampaignTools


def export(private, public):
    """Release constructed diagnostics/numeric rows, never oracle text or inputs."""
    value = json.loads((private / 'results/summary.json').read_text())
    result = {key: value[key] for key in ('state', 'failure_stage', 'operation',
              'source_sha256', 'physical_device', 'logical_device', 'diagnostic', 'progress') if key in value}
    checked = value.get('check', {})
    result['correctness'] = checked.get('passed') is True and checked.get('variants') == [0, 1]
    # candidate_probe already authenticates candidate-owned compiler diagnostics.
    result['diagnostics'] = checked.get('diagnostics', [])
    measurement = value.get('result', {})
    result.update({key: measurement[key] for key in
                  ('latency', 'busy_time', 'cases', 'discovered_names', 'selector_mapping', 'warmup', 'samples') if key in measurement})
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
                        numeric[key] = float(item)
                    except (TypeError, ValueError):
                        if item == selector or item in ('N/A', 'NA', '-', ''):
                            numeric[key] = item
                clean[field].append(numeric)
        result['captures'].append(clean)
    public.mkdir(parents=True)
    (public / 'results').mkdir()
    (public / 'results/summary.json').write_text(json.dumps(result, indent=2))


def prepare(output, task='matmul'):
    """Freeze a self-contained private worker and minimal public API bundle."""
    if task != 'matmul':
        raise ValueError('matmul gate must pass before enabling harder tasks')
    source = Path(__file__).resolve().parents[1]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    public, private = output / 'public', output / 'private'
    public.mkdir()
    (private / 'scripts').mkdir(parents=True)
    helpers = private / 'skills/ascend-model-benchmarking/scripts'
    helpers.mkdir(parents=True)
    for name in ('model_campaign_worker.py', 'profile_device.py', 'a3_memory_batch.py',
                 'private_campaign_cases.py', 'private_campaign_models.py', 'model_campaign_native.py', 'model_device.py'):
        shutil.copyfile(source / 'scripts' / name, private / 'scripts' / name)
    for name in ('model_execution.py', 'candidate_probe.py', 'profile_capture.py', 'profile_accounting.py'):
        shutil.copyfile(source / 'skills/ascend-model-benchmarking/scripts' / name, helpers / name)
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
        parser.add_argument('--task', default='matmul')
        parser.add_argument('--output', type=Path, required=True)
        options = parser.parse_args(argv)
        print(json.dumps(prepare(options.output, options.task)))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--target', default='bz-a3-1')
    parser.add_argument('--runtime', default='py311-torch')
    parser.add_argument('verb', choices=('help', 'check', 'timing', 'pipes', 'observe'))
    parser.add_argument('call_id', nargs='?')
    options = parser.parse_args(argv)
    if options.verb == 'help':
        print('experiment-tool check|timing|pipes; experiment-tool observe CALL_ID\n'
              'Every operation independently checks compiled Model correctness on private input variants.\n'
              'Compilation/numerical failures include safe diagnostics; retry edits until role deadline, no attempt quota.\n'
              'timing measures warm synchronized full-forward latency; pipes discovers kernel names then collects metrics.\n'
              'Effective captures: 3 warm latency samples; scoped trace discovery, 1 BasicInfo replay per trace hint, '
              '1 PipeUtilization per exported canonical selector.\n'
              'Evidence is evidence/CALL_ID/results/summary.json. Preserve original paths in analysis reports.\n'
              'Resume nonterminal infrastructure jobs with observe; do not duplicate dispatch.\n'
              'All staging/device waits are included in execution.json deadline; no implicit budget extension.')
        return 0
    if options.verb == 'observe' and not options.call_id:
        parser.error('observe requires CALL_ID')
    private_workspace = options.workspace.parent / 'private-model-workspace'
    private_workspace.mkdir(exist_ok=True)
    tools = CampaignTools(options.bundle, private_workspace, options.target, options.runtime)
    # Match controller cleanup ownership and pending-call discovery exactly.
    tools.state = options.workspace.parent / 'tool-state'
    tools.state.mkdir(exist_ok=True)
    service = CampaignService(tools, options.bundle, options.workspace, export)
    result = service.observe(options.call_id) if options.verb == 'observe' else service.start(
        'benchmark' if options.verb == 'timing' else options.verb)
    print(json.dumps(result, indent=2))
    return 1 if result.get('state') == 'failed' else 0


if __name__ == '__main__':
    raise SystemExit(main())
