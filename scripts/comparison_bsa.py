"""Private adapter for supplied RainFusionAttention comparison cases.

This module and prepared roots belong to the trusted harness, never actor bundles.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

REFERENCE = '202_RainFusionAttention.py'
CASES = '202_RainFusionAttention.json'
API = '''forward(query, key, value, select_idx, select_num_idx, block_shape,
q_seqlen_list, kv_seqlen_list, scale_value, q_input_layout="TND",
kv_input_layout="TND", inner_precise=0) -> float32 Tensor[Tq,H,D]
TND concatenates independent sequences. Query head h uses KV head h//(H/KH).
For each query block and head, attend only KV block IDs in the first
select_num_idx slots. Negative/out-of-range block IDs and KV tail padding are
ignored. Normalize exp(scale*QK) across all selected valid tokens; no visible
tokens yields zero. Q block and KV block sizes are block_shape. Sequence lengths
are per-sequence, not cumulative. inner_precise=1 rounds scale to float16 first.
Inputs must remain unchanged. Only TND layouts are supported.
'''
STARTER = (Path(__file__).resolve().parents[1] / 'benchmarks/comparison_bsa_starter.py').read_text()


def prepare(private_dir, reference_root):
    """Copy exact supplied oracle and cases; return only public worker material."""
    source, dest = Path(reference_root), Path(private_dir)
    dest.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name in (REFERENCE, CASES):
        payload = (source / name).read_bytes()
        target = dest / name
        if target.exists() and target.read_bytes() != payload:
            raise ValueError(f'refusing to replace private evidence: {name}')
        if source.resolve() != dest.resolve():
            shutil.copyfile(source / name, target)
        provenance[name] = hashlib.sha256(payload).hexdigest()
    return {'api': API, 'starter': STARTER, 'starter_status': 'naive-triton',
            'case_count': len(specifications(dest)), 'provenance': provenance}


def specifications(root):
    with (Path(root) / CASES).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_reference(root):
    path = Path(root) / REFERENCE
    spec = importlib.util.spec_from_file_location('_private_rain_reference', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Model().eval()


def make_case(root, index, variant=0, device='npu'):
    """Generate one independent case without materializing the other 49 cases.

    Variant 0 preserves the supplied selection counts. Variants 1/2 vary runtime
    selections, and 3/4 exercise empty/all-invalid visibility respectively.
    All variants use distinct random Q/K/V values.
    """
    import torch
    if variant not in range(5):
        raise ValueError('variant must be in 0..4')
    spec = specifications(root)[index]
    tensors = {x['name']: x for x in spec['inputs'] if x['type'] == 'tensor'}
    attrs = {x['name']: x.get('value') for x in spec['inputs'] if x['type'] != 'tensor'}
    gen = torch.Generator().manual_seed(3407 + index + 100003 * variant)
    qkv = [torch.randn(tensors[n]['shape'], generator=gen, dtype=torch.float32).to(
        getattr(torch, tensors[n]['dtype'])) for n in ('query', 'key', 'value')]
    counts = torch.tensor(tensors['select_num_idx']['value'], dtype=torch.int64)
    ids = torch.full(tensors['select_idx']['shape'], -1, dtype=torch.int64)
    sx, sy = attrs['blockshape']
    offset = 0
    for qlen, kvlen in zip(attrs['actual_seq_qlen'], attrs['actual_seq_kvlen']):
        nq, nk = (qlen + sx - 1) // sx, (kvlen + sy - 1) // sy
        for block in range(offset, offset + nq):
            for head in range(ids.shape[1]):
                limit = min(nk, ids.shape[2])
                if variant in (1, 2):
                    counts[block, head] = torch.randint(limit + 1, (), generator=gen)
                elif variant == 3:
                    counts[block, head] = 0
                n = min(int(counts[block, head]), limit)
                if n and variant != 4:
                    ids[block, head, :n] = torch.randperm(nk, generator=gen)[:n].sort().values
                elif variant == 4:
                    ids[block, head, :n] = nk + 7
        offset += nq
    args = tuple(x.to(device) for x in (*qkv, ids, counts)) + (
        attrs['blockshape'], attrs['actual_seq_qlen'], attrs['actual_seq_kvlen'],
        attrs['scale'], attrs['q_input_layout'], attrs['kv_input_layout'], attrs['inner_precise'])
    snapshots = tuple(x.clone() if torch.is_tensor(x) else __import__('copy').deepcopy(x)
                      for x in args)
    with torch.no_grad():
        expected = load_reference(root)(*args)
    return {'id': f'bsa-{index}-v{variant}', 'args': args, 'expected': expected,
            'metadata': {'case_index': index, 'variant': variant}, '_snapshots': snapshots}


def validate(case, arm, result):
    """Reject wrong shape/dtype, nonfinite output, numerical error and mutation."""
    import torch
    for actual, original in zip(case['args'], case['_snapshots']):
        if torch.is_tensor(actual):
            if not torch.equal(actual, original):
                raise AssertionError(f'{arm}: input mutation')
        elif actual != original:
            raise AssertionError(f'{arm}: input mutation')
    expected = case['expected']
    if not torch.is_tensor(result) or result.dtype != torch.float32:
        raise AssertionError(f'{arm}: expected float32 tensor')
    if result.shape != expected.shape or result.device != expected.device:
        raise AssertionError(f'{arm}: output shape/device mismatch')
    torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3,
                               equal_nan=False, check_dtype=True)
    return {'passed': True, 'case_id': case['id']}
