"""Private reproducible cases and original-reference validators, not agent inputs."""
import json
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F

from private_campaign_models import original

ROOT = Path(__file__).resolve().parents[1] / 'benchmarks'


def specifications(task):
    return [json.loads(line) for line in (ROOT / task / 'cases.jsonl').read_text().splitlines()
            if line.strip()]


def bsa_workspace_estimate(index):
    items = {item['name']: item for item in specifications('bsa')[index]['inputs']}
    tq, heads, dim = items['q']['shape']
    tk = items['k']['shape'][0]
    pairs = heads * tq * tk
    # Planning bound: six fp32-sized pair matrices, plus expanded fp32 Q/K/V/O.
    # This is a conservative admission budget, not a measured compiler peak.
    return {'pair_elements': pairs, 'one_fp32_matrix_bytes': pairs * 4,
            'planning_bytes': pairs * 24 + (2 * tq + 2 * tk) * heads * dim * 4}


def make_case(task, index, *, device='npu', variant=0, workspace_budget_bytes=None):
    """New deterministic values, original case shapes/API. Variant tests ordinary data.

    Generation is CPU/private, with a private generator. The untouched GDN oracle
    resets the CPU RNG; fork_rng restores that incidental historical side effect.
    Variant 1 is an ordinary runtime variation, not a different mathematical task.
    """
    spec = specifications(task)[index]
    if task == 'bsa' and workspace_budget_bytes is not None:
        estimate = bsa_workspace_estimate(index)
        if estimate['planning_bytes'] > workspace_budget_bytes:
            raise MemoryError(f'BSA case {index}: graph-v1 conservative workspace budget '
                              f"{estimate['planning_bytes']} exceeds available budget "
                              f'{workspace_budget_bytes}; original shape is not reduced')
    rng = torch.Generator().manual_seed(17011 + index * 97 + variant * 1009)

    def rand(shape, dtype=torch.float32, scale=.2):
        return (torch.randn(shape, generator=rng) * scale).to(dtype)

    if task == 'matmul':
        args = [rand((spec['m'], spec['k']), torch.float16),
                rand((spec['k'], spec['n']), torch.float16),
                rand((spec['n'],), torch.float16)]
    else:
        items = {item['name']: item for item in spec['inputs']}
        attrs = {name: item['value'] for name, item in items.items() if item['type'] == 'attr'}
        tensor = {name: rand(item['shape'], getattr(torch, item['dtype']))
                  for name, item in items.items() if item['type'] == 'tensor'
                  and item['dtype'] not in ('int32', 'bool')}
        if task == 'gdn':
            for name in ('q', 'k'):
                tensor[name] = F.normalize(tensor[name].float(), dim=-1, eps=1e-6).to(tensor[name].dtype)
                # Deliberately do not pre-normalize every validation input.
                if variant:
                    tensor[name] = tensor[name] * (3. if name == 'q' else .3)
            if not attrs['use_gate_in_kernel']:
                tensor['g'] = -F.softplus(tensor['g']) * .08
            if not attrs['use_beta_sigmoid']:
                tensor['beta'] = torch.sigmoid(tensor['beta'])
            names = ('q', 'k', 'v', 'g', 'beta', 'A_log', 'dt_bias', 'initial_state')
            flags = ('scale', 'chunk_size', 'use_gate_in_kernel', 'use_bias',
                     'use_initial_state', 'output_final_state', 'use_qk_l2norm',
                     'use_beta_sigmoid', 'allow_neg_eigval', 'state_v_first')
            args = [tensor[name] for name in names] + [attrs[name] for name in flags]
        elif task == 'bsa':
            heads = tensor['q'].shape[1]
            sparse_heads = items['base_blockmask']['shape'][1]
            stream_heads = heads // 3
            types = torch.tensor([0] * (heads - sparse_heads - stream_heads)
                                 + [1] * sparse_heads + [-1] * stream_heads, dtype=torch.int32)
            stream = torch.tensor([attrs['sink'], attrs['local']] * heads, dtype=torch.int32).reshape(heads, 2)
            if variant:
                types = types[torch.randperm(heads, generator=rng)]
                stream[:, 0] = torch.arange(heads) % 3
                stream[:, 1] = 1 + torch.arange(heads) % 4
            mask = torch.rand(items['base_blockmask']['shape'], generator=rng) < attrs['sparsity']
            def boundaries(name):
                return torch.tensor([0] + list(torch.tensor(attrs[name]).cumsum(0)), dtype=torch.int32)
            args = [tensor['q'], tensor['k'], tensor['v'], boundaries('q_lens'),
                    boundaries('k_lens'), types, stream.flatten(), mask,
                    attrs['softmax_scale'], attrs['is_causal'], attrs['exact_streaming']]
        else:
            raise ValueError(f'unknown task: {task}')
    args = tuple(value.to(device) if isinstance(value, torch.Tensor) else value for value in args)
    with torch.inference_mode(), torch.random.fork_rng(devices=[]):
        expected = original(task).Model()(*args)
    source = Path(original(task).__file__)
    provenance = {'task': task, 'original_case_index': index,
                  'seed': 17011 + index * 97 + variant * 1009, 'variant': variant,
                  'specification': spec, 'original_source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'adaptation_version': {'matmul': 'original', 'bsa': 'graph-v1',
                                         'gdn': 'graph-v2-native-solver-layout-bridge'}[task]}
    return {'id': f'{task}-{index}-variant-{variant}', 'args': args,
            'task': task, 'expected': expected, 'reference_version': 'original-untouched',
            'adaptation_version': provenance['adaptation_version'], 'metadata': provenance}


def validate(case, arm, result):
    """Use the pinned upstream precision policy against untouched references."""
    from benchmark_correctness import verify_outputs
    verify_outputs(case['expected'], result, case['args'],
                   non_compute=case.get('non_compute', False))


def matmul_cases():
    for index in (7, 8, 9):
        yield make_case('matmul', index)


def gdn_cases():
    for index in (40, 49, 47, 46, 45):
        yield make_case('gdn', index)


def bsa_cases():
    for index in (47, 46, 49, 44, 43):
        free_bytes, _ = torch.npu.mem_get_info()
        yield make_case('bsa', index, workspace_budget_bytes=free_bytes)
