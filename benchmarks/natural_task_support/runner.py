"""Trusted seeded inputs, numerical reference and standalone device checker."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import torch


def reference(spec, inputs):
    family = spec['family']
    if family == 'gather':
        value = inputs['x'][inputs['index'].long()]
    elif family == 'masked':
        value = inputs['x']
        score = value * (2 * torch.sigmoid(value * inputs['gain']) - 1)
        score = score.masked_fill(~inputs['selected'], -float('inf'))
        maximum = score.amax(-1, keepdim=True)
        maximum = torch.where(torch.isfinite(maximum), maximum, 0)
        exp = torch.exp(score - maximum)
        return exp / exp.sum(-1, keepdim=True).clamp_min(1e-20)
    elif family == 'matmul':
        value = inputs['lhs'].float() @ inputs['rhs'].float() + inputs['bias']
    else:
        raise ValueError('unknown task family')
    for a, b in inputs['coeff']:
        value = value + a * (2 * torch.sigmoid(value * b) - 1)
    if family == 'gather':
        weight = inputs['weight']
        return torch.stack(((value * weight).mean(-1), (value.square() * weight).mean(-1)), -1)
    return inputs['residual'] + value * torch.sigmoid(value)


def inputs_and_reference(spec, seed, check=0):
    generator = torch.Generator().manual_seed(seed)
    def randn(*shape):
        return torch.randn(shape, generator=generator)
    family = spec['family']
    if family == 'gather':
        rows, width, table = spec['rows'], spec['width'], spec['table']
        index = torch.randint(table if check % 3 != 2 else min(table, 64),
                              (rows, width), generator=generator, dtype=torch.int32)
        if check % 3 == 1:
            index = (torch.arange(rows * width).reshape(rows, width) % table).int()
        inputs = dict(x=randn(table), index=index, weight=randn(rows, width).abs() + .1)
    elif family == 'masked':
        rows, width = spec['rows'], spec['width']
        selected = torch.rand((rows, width), generator=generator) < spec['density']
        if check % 3 == 0:
            selected = ((torch.arange(width)[None, :] + torch.arange(rows)[:, None]) %
                        8) < round(8 * spec['density'])
        selected[0] = False
        selected[1] = False
        selected[1, -1] = True
        inputs = dict(x=randn(rows, width), selected=selected, gain=randn(width))
    elif family == 'matmul':
        m, n, k = spec['m'], spec['n'], spec['k']
        inputs = dict(lhs=(randn(m, k) * .2).half(), rhs=(randn(k, n) * .2).half(),
                      bias=randn(n), residual=randn(m, n))
    else:
        raise ValueError('unknown task family')
    if family != 'masked':
        inputs['coeff'] = torch.stack((randn(spec['rounds']) * .08,
                                       randn(spec['rounds']) * .4 + .8), -1)
    return inputs, reference(spec, inputs)


def validate_output(spec, actual, expected):
    torch.testing.assert_close(actual, expected, **spec['tolerances'])


def launch_with_contract(module, inputs, output, spec):
    return module.launch(inputs, output), None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--implementation', choices=['baseline', 'candidate'], default='candidate')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--checks', type=int, default=1)
    parser.add_argument('--launches', type=int, default=1)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cpu-only', action='store_true')
    args = parser.parse_args()
    if min(args.checks, args.launches) < 1:
        parser.error('checks and launches must be positive')
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    spec = json.loads((root / 'task.json').read_text())
    module = None
    if not args.cpu_only:
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
        loader = importlib.util.spec_from_file_location('implementation', root / f'{args.implementation}.py')
        module = importlib.util.module_from_spec(loader)
        sys.modules['implementation'] = module
        loader.loader.exec_module(module)
    for check in range(args.checks):
        inputs, expected = inputs_and_reference(spec, args.seed + check, check)
        if module is not None:
            device_inputs = {name: value.npu() for name, value in inputs.items()}
            output = torch.empty_like(expected, device='npu')
            for _ in range(args.launches):
                compiled, _ = launch_with_contract(module, device_inputs, output, spec)
            torch.npu.synchronize()
            validate_output(spec, output.cpu(), expected)
            if compiled is not None and hasattr(compiled, 'asm'):
                artifacts = root / 'compiler-evidence' / args.implementation
                artifacts.mkdir(parents=True, exist_ok=True)
                for stage, data in compiled.asm.items():
                    if isinstance(data, str) and stage.replace('_', '').isalnum():
                        (artifacts / (stage + '.txt')).write_text(data)
        print(json.dumps(dict(correctness=True, seed=args.seed + check, check=check,
                              selector=spec['selector'], elements=expected.numel())), flush=True)


if __name__ == '__main__':
    main()
