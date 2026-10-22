"""Private packed-prefill GDN correctness adapter; never stage this oracle to actors."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

ARGUMENTS = ('q', 'k', 'v', 'g', 'beta', 'initial_state', 'cu_seqlens',
             'scale', 'use_qk_l2norm_in_kernel')
REFERENCE = '200_GDN_vllm.py'
CASES = '200_GDN_vllm.json'


def prepare(private_dir, reference_root):
    """Copy exact supplied inputs privately; return separately labelled metadata."""
    destination = Path(private_dir)
    source = Path(reference_root) / 'GDN_vllm' / 'GDN_vllm'
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in (REFERENCE, CASES):
        content = (source / name).read_bytes()
        path = destination / name
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f'refusing to replace pinned reference {name}')
        path.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    comparators = {}
    for label, relative in (('generated', 'GDN_vllm/GDN_vllm/200_GDN_vllm_generated.py'),
                            ('optimized', 'optimized_code.py'), ('code1.28X', 'code1.28X.py')):
        comparator = Path(reference_root) / relative
        if comparator.is_file():
            content = comparator.read_bytes()
            target = destination / ('comparator_' + label + '.py')
            if target.exists() and target.read_bytes() != content:
                raise ValueError(f'refusing to replace pinned comparator {label}')
            target.write_bytes(content)
            comparators[label] = {'file': target.name, 'sha256': hashlib.sha256(content).hexdigest(),
                                  'entrypoint': 'ModelNew'}
    return {'public': {'entrypoint': 'Model.forward', 'arguments': list(ARGUMENTS),
                       'contract': ('Packed B=1; q/k [1,T,Hk,K], v [1,T,Hv,V], g/beta [1,T,Hv]. '
                                    'For each cu_seqlens segment start from its FP32 state [Hv,K,V]. '
                                    'Map value head h to key head h//(Hv/Hk). Optionally normalize '
                                    'q and k by rsqrt(sum(x*x)+1e-6). Per token: S*=exp(g); '
                                    'delta=beta*(v-k@S); S+=outer(k,delta); output=(q*scale)@S. '
                                    'Return output in v.dtype and final FP32 states; do not mutate inputs.'),
                       'outputs': ['output: v.dtype [1,T,Hv,V]',
                                   'final_state: float32 [N,Hv,K,V]'],
                       'starter_status': 'requires-device-qualification',
                       'starter': (Path(__file__).resolve().parents[1] / 'benchmarks/comparison_gdn_starter.py').read_text()},
            'private': {'comparators': comparators, 'source_sha256': hashes, 'case_count': len(specifications(destination))}}


def specifications(root):
    return [json.loads(line) for line in (Path(root) / CASES).read_text().splitlines()
            if line.strip()]


def load_reference(root):
    path = Path(root) / REFERENCE
    name = '_private_gdn_' + hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model()


def make_case(root, index, variant=0, device='npu'):
    """Generate deterministic fresh values; variant zero preserves supplied boundaries."""
    import torch
    spec = {item['name']: item for item in specifications(root)[index]['inputs']}
    generator = torch.Generator(device='cpu').manual_seed(3500 + index + 100003 * variant)
    dtypes = {'bfloat16': torch.bfloat16, 'float16': torch.float16,
              'float32': torch.float32}
    def normal(name):
        item = spec[name]
        return torch.randn(item['shape'], dtype=dtypes[item['dtype']], generator=generator)
    q, k, v = (normal(name) for name in ('q', 'k', 'v'))
    g = -(0.001 + 0.04 * torch.rand(spec['g']['shape'], generator=generator))
    beta = torch.sigmoid(torch.randn(spec['beta']['shape'], generator=generator)).to(dtypes[spec['beta']['dtype']])
    initial = normal('initial_state') * 0.01 if (index + variant) % 2 else torch.zeros(spec['initial_state']['shape'], dtype=torch.float32)
    boundaries = list(spec['cu_seqlens']['value'])
    if variant and len(boundaries) > 2:
        # Exercise runtime packed boundaries, retaining positive lengths and total T.
        total, count = boundaries[-1], len(boundaries) - 1
        cuts = torch.randperm(total - 1, generator=generator)[:count - 1].add(1).sort().values.tolist()
        boundaries = [0, *cuts, total]
    args = [x.to(device) for x in (q, k, v, g, beta, initial,
                                  torch.tensor(boundaries, dtype=torch.int32))]
    args.extend((spec['scale']['value'], spec['use_qk_l2norm_in_kernel']['value']))
    snapshots = [x.clone() for x in args[:7]]
    with torch.no_grad():
        expected = load_reference(root)(*args)
    return {'id': f'gdn-{index}-v{variant}', 'args': args, 'expected': expected,
            'metadata': {'case_index': index, 'variant': variant, 'boundaries': boundaries,
                         'initial_state_nonzero': bool((index + variant) % 2)},
            '_snapshots': snapshots}


def validate(case, arm, result):
    """Check both numerical outputs and caller-owned inputs; return diagnostics."""
    import torch
    errors = []
    for name, actual, snapshot in zip(ARGUMENTS, case['args'][:7], case['_snapshots']):
        if not torch.equal(actual, snapshot):
            errors.append(f'input mutated: {name}')
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        errors.append('expected (output, final_state)')
    else:
        for name, actual, expected in zip(('output', 'final_state'), result, case['expected']):
            if not isinstance(actual, torch.Tensor):
                errors.append(f'{name}: not a tensor')
                continue
            if actual.dtype != expected.dtype or actual.device != expected.device or actual.shape != expected.shape:
                errors.append(f'{name}: shape/dtype/device mismatch')
                continue
            try:
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
            except AssertionError as exc:
                errors.append(f'{name}: {exc}')
    if errors:
        raise AssertionError('; '.join(errors))
    return {'passed': True, 'arm': arm, 'case_id': case['id']}
