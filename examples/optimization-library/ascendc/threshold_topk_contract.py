"""Frozen F03 finite-FP16 TopK semantics; tied indices may vary."""
from pathlib import Path
import torch


def dimensions(spec, *, guard=False):
    rows, vocab, k, seed = (spec.get(key) for key in ('rows', 'V', 'K', 'seed'))
    if any(type(value) is not int for value in (rows, vocab, k, seed)):
        raise ValueError('integer rows, V, K and seed required')
    if rows <= 0 or vocab <= 0 or seed < 0 or not 1 <= k <= min(vocab, 1024):
        raise ValueError('positive rows/V, nonnegative seed and 1<=K<=min(V,1024) required')
    return rows, vocab, k, seed


def make_case(spec):
    rows, vocab, k, seed = dimensions(spec)
    rng = torch.Generator().manual_seed(seed)
    distribution = spec['distribution']
    if distribution == 'equal':
        scores = torch.ones(rows, vocab, dtype=torch.float16)
    else:
        jitter = torch.rand(rows, vocab, generator=rng) * 0.5 - 0.25
        tile = torch.arange(vocab) // 8192
        if distribution == 'first-tile-dominant-fp16':
            base = torch.where(tile == 0, 8.0, -8.0)
        elif distribution in ('ascending-tiles', 'ascending-quantized-levels'):
            base = 2.0 * tile
            if distribution == 'ascending-quantized-levels':
                jitter = torch.round((jitter + 0.25) * 62) / 62 - 0.25
        else:
            raise ValueError('unknown frozen score distribution')
        scores = (base + jitter).half()
    expected = scores.sort(descending=True).values[:, :k].clone()
    return dict(scores=scores, expected=expected)


def running_reference(scores, k, tile=8192):
    """Executable mechanism model, used only by host functional tests."""
    dimensions(dict(rows=scores.shape[0], V=scores.shape[1], K=k, seed=0))
    values, indices = [], []
    for row in scores:
        v, i = row[:tile].sort(descending=True)
        v, i = v[:k], i[:k]
        for start in range(tile, row.numel(), tile):
            segment = row[start:start + tile]
            mask = segment > v[-1]
            positions = torch.arange(start, start + segment.numel())[mask]
            candidates = torch.cat((v, segment[mask]))
            origins = torch.cat((i, positions))
            v, order = candidates.sort(descending=True)
            v, i = v[:k], origins[order[:k]]
        values.append(v)
        indices.append(i)
    return torch.stack(values), torch.stack(indices)


def diagnostics(actual, fixture):
    values, indices = (item.detach().cpu() for item in actual)
    scores, expected = fixture['scores'], fixture['expected']
    if values.shape != expected.shape or indices.shape != expected.shape:
        return dict(passed=False, reason='output shape')
    if values.dtype != torch.float16 or indices.dtype not in (torch.int32, torch.int64):
        return dict(passed=False, reason='output dtype')
    if not bool(((indices >= 0) & (indices < scores.shape[1])).all()):
        return dict(passed=False, reason='index range')
    ordered = indices.sort().values
    unique = bool((ordered[:, 1:] != ordered[:, :-1]).all())
    gathered = torch.equal(scores.gather(1, indices.long()), values)
    descending = bool((values[:, :-1] >= values[:, 1:]).all())
    exact = torch.equal(values, expected)
    # Exact descending code-value multiset implies every unselected score <= min.
    return dict(passed=unique and gathered and descending and exact,
                unique_indices=unique, gather_matches=gathered, descending=descending,
                exact_topk_values=exact)


CASE_SOURCES = [Path(__file__)]
