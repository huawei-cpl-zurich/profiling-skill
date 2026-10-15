"""Numerical contract and execution entrypoint for a standalone task."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

import torch


def validate_repetition_ir(text, roles):
    """Qualify task04's compiled repeated volatile work, not source spelling.

    This deliberately supports canonical scf.for [0,repetitions) with step 1.
    Other loop forms need a separately qualified validator, not silent acceptance.
    """
    if set(roles) != {'x', 'selected', 'repetitions', 'output'}:
        raise ValueError('semantic contract: executed kernel tensor bindings unavailable')
    origins = {value: {name} for name, value in roles.items()}
    constants, bounds, loops, stores = {}, set(), [], []
    active, depth = None, 0
    token = r'%[\w.$-]+(?:#[0-9]+)?'
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = raw.split(' loc(', 1)[0].strip()
        loop = re.search(r'scf\.for\s+' + token + r'\s*=\s*(' + token + r')\s+to\s+(' + token + r')\s+step\s+(' + token + ')', line)
        if loop:
            lower, upper, step = loop.groups()
            qualified = (constants.get(lower) == 0 and constants.get(step) == 1 and
                         upper in bounds)
            if qualified:
                if active is not None:
                    raise ValueError('semantic contract: nested repetition loops unsupported')
                active = dict(line=line_number, depth=depth, loads=set(), stores=0)
                loops.append(active)
        assignment = re.match('(' + token + r')\s*=\s*(.*)', line)
        if assignment:
            result, operation = assignment.groups()
            operands = re.findall(token, operation)
            inherited = set().union(*(origins.get(value, set()) for value in operands))
            constant = re.match(r'arith\.constant\s+(-?\d+)\s*:', operation)
            if constant:
                constants[result] = int(constant[1])
            if re.match(r'arith\.(?:extsi|extui|index_cast)\s+', operation) and operands and operands[0] in bounds:
                bounds.add(result)
            load = re.match(r'tt\.load\s+(' + token + ')', operation)
            if load:
                pointer_roles = origins.get(load[1], set()) & set(roles)
                inherited.update('loaded:' + role for role in pointer_roles)
                if pointer_roles == {'repetitions'}:
                    bounds.add(result)
                if re.search(r'isVolatile\s*=\s*true', operation):
                    if active is not None:
                        inherited.update('volatile:' + role for role in pointer_roles)
                        active['loads'].update(pointer_roles)
            origins[result] = inherited
        store = re.match(r'tt\.store\s+(' + token + r'),\s*(' + token + ')', line)
        if store and 'output' in origins.get(store[1], set()):
            if active is None:
                raise ValueError('semantic contract: output store outside runtime repetition loop')
            if not {'volatile:x', 'volatile:selected'} <= origins.get(store[2], set()):
                raise ValueError('semantic contract: output does not depend on both volatile inputs')
            active['stores'] += 1
            stores.append(line_number)
        depth += line.count('{') - line.count('}')
        if active is not None and depth <= active['depth']:
            active = None
    if len(loops) != 1 or not stores or not {'x', 'selected'} <= loops[0]['loads']:
        raise ValueError('semantic contract: repeated volatile input loads and output store not established')
    return dict(qualified=True, kind='runtime-repeated-volatile-work', stage='ttir',
                loop_line=loops[0]['line'], output_store_lines=stores,
                ttir_sha256=hashlib.sha256(text.encode()).hexdigest())


def launch_with_contract(module, inputs, output, spec):
    if spec['id'] != 'task04':
        return module.launch(inputs, output), None
    from triton.runtime.jit import JITFunction
    original, records = JITFunction.run, []
    tensor_roles = {value.data_ptr(): name for name, value in dict(inputs, output=output).items()}
    def checked_run(kernel, *args, **kwargs):
        bound = dict(zip(kernel.arg_names, args))
        bound.update({name: kwargs[name] for name in kernel.arg_names if name in kwargs})
        compiled = original(kernel, *args, **kwargs)
        # constexpr arguments are absent from the compiled function signature.
        dynamic = [bound[name] for i, name in enumerate(kernel.arg_names) if i not in kernel.constexprs]
        signature = re.search(r'tt\.func\s+(?:public\s+)?@[^\(]+\((.*?)\)\s*(?:attributes|\{)',
                              compiled.asm.get('ttir', ''), re.S)
        if signature is None:
            raise ValueError('semantic contract: compiled TTIR function signature unavailable')
        names = re.findall(r'(%[\w.$-]+)\s*:', signature[1])
        if len(names) != len(dynamic):
            raise ValueError('semantic contract: unsupported specialized runtime signature')
        roles = {tensor_roles[value.data_ptr()]: name for name, value in zip(names, dynamic)
                 if isinstance(value, torch.Tensor) and value.data_ptr() in tensor_roles}
        evidence = validate_repetition_ir(compiled.asm.get('ttir', ''), roles)
        records.append((compiled, evidence))
        return compiled
    JITFunction.run = checked_run
    try:
        compiled = module.launch(inputs, output)
    finally:
        JITFunction.run = original
    if len(records) != 1 or records[0][0] is not compiled:
        raise ValueError('semantic contract: return the one executed compiled kernel')
    return compiled, records[0][1]


def inputs_and_reference(spec, seed, check=0):
    generator = torch.Generator().manual_seed(seed)
    task, n = spec['id'], spec['n']
    x = torch.randn(n, generator=generator, dtype=torch.float32)
    inputs = {'x': x}
    if task == 'task01':
        work = (64, 0, 256, 17)[check % 4]
        inputs['work'] = torch.tensor([work], dtype=torch.int32)
        expected = x * (1 + work * 1e-6) * 1.5 + 2
    elif task == 'task02':
        if check % 3 == 0:
            indices = torch.arange(n, dtype=torch.int32)
        elif check % 3 == 1:
            indices = torch.randperm(n, generator=generator).int()
        else:
            indices = torch.randint(0, 64, (n,), generator=generator, dtype=torch.int32)
        inputs['index'] = indices
        expected = x[indices.long()]
    elif task == 'task03':
        inputs = {k: torch.randn((n, n), generator=generator) for k in ('lhs', 'rhs')}
        expected = inputs['lhs'] @ inputs['rhs']
    elif task == 'task04':
        selected = ((torch.arange(n) * 17 % 4) == 0) if check == 0 else torch.rand(n, generator=generator) < 0.35
        inputs['selected'] = selected
        inputs['repetitions'] = torch.tensor([(128, 1, 32)[check % 3]], dtype=torch.int32)
        expected = torch.where(selected, x * 1.5 + 2, 0)
    elif task == 'task05':
        expected = x * 1.5 + 2
    elif task == 'task06':
        state, values = seed & 0xffffffff, []
        for _ in range(2 * n):
            state = (1664525 * state + 1013904223) & 0xffffffff
            values.append((state % 2049 - 1024) / 64)
        inputs = {'x': torch.tensor(values, dtype=torch.float32)}
        expected = torch.cat((inputs['x'][:n] + inputs['x'][n:] + 32, torch.full((n,), 32.)))
    else:
        raise ValueError('unknown task')
    return inputs, expected


def native_run(root, args):
    source = root / f'{args.implementation}.asc'
    identity = hashlib.sha256(source.read_bytes() + (root / 'native_host.asc').read_bytes()).hexdigest()[:16]
    folder = root / 'build' / identity
    folder.mkdir(parents=True, exist_ok=True)
    executable = folder / 'kernel'
    if not executable.exists():
        command = ['bisheng', str(root / 'native_host.asc'), '-I', str(root), '-O2', '-g',
                   '--npu-arch=dav-2201', '-lascendcl', '-o', str(executable),
                   f'-DIMPLEMENTATION="{args.implementation}.asc"']
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, cwd=folder)
        (folder / 'compile.json').write_text(json.dumps({'command': command, 'returncode': result.returncode}))
        (folder / 'compile.stdout').write_text(result.stdout)
        (folder / 'compile.stderr').write_text(result.stderr)
        if result.returncode:
            sys.stderr.write(result.stderr)
            raise RuntimeError('native compilation failed; retained diagnostics in ' + str(folder))
    for check in range(args.checks):
        subprocess.run([str(executable), str(args.seed + check), str(args.launches), str(args.device)], check=True, timeout=300)


def validate_output(spec, actual, expected):
    torch.testing.assert_close(actual, expected, **spec['tolerances'])


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
    root = Path(__file__).resolve().parent
    spec = json.loads((root / 'task.json').read_text())
    if spec['language'] == 'ascendc':
        if args.cpu_only:
            inputs, expected = inputs_and_reference(spec, args.seed)
            print(json.dumps({'correctness': True, 'native_oracle': 'fixed host', 'elements': expected.numel()}))
        else:
            native_run(root, args)
        return
    module = None
    if not args.cpu_only:
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
        module_spec = importlib.util.spec_from_file_location('implementation', root / f'{args.implementation}.py')
        module = importlib.util.module_from_spec(module_spec)
        sys.modules['implementation'] = module
        module_spec.loader.exec_module(module)
    for check in range(args.checks):
        inputs, expected = inputs_and_reference(spec, args.seed + check, check)
        if module is not None:
            device_inputs = {k: v.npu() for k, v in inputs.items()}
            output = torch.empty_like(expected, device='npu')
            for _ in range(args.launches):
                compiled, semantic_contract = launch_with_contract(module, device_inputs, output, spec)
            torch.npu.synchronize()
            actual = output.cpu()
            validate_output(spec, actual, expected)
            if compiled is not None and hasattr(compiled, 'asm'):
                artifacts = root / 'compiler-evidence' / args.implementation
                artifacts.mkdir(parents=True, exist_ok=True)
                for stage, data in compiled.asm.items():
                    if isinstance(data, str) and stage.replace('_', '').isalnum():
                        (artifacts / (stage + '.txt')).write_text(data)
        print(json.dumps({'correctness': True, 'seed': args.seed + check,
                          'check': check, 'selector': 'task_kernel', 'elements': expected.numel(),
                          'semantic_contract': semantic_contract if module is not None else None}), flush=True)


if __name__ == '__main__':
    main()
