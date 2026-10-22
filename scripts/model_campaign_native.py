"""Private native matmul Model adapter. Never mount this into optimizer folders."""
import csv
import json
from pathlib import Path
import shlex
import statistics
import subprocess
import sys
import time

from a3_memory_batch import execute, validate_capture


def audited_compiler():
    from model_execution import GraphCompiler
    class TritonCompiler(GraphCompiler):
        def backend_wrapper(self, graph, example_inputs):
            triton_nodes = 0
            for node in graph.graph.nodes:
                if node.op in ('placeholder', 'output', 'get_attr'):
                    continue
                name = str(node.target)
                if 'triton_kernel_wrapper' in name:
                    triton_nodes += 1
                elif node.op == 'call_method' and name in ('view', 'reshape', 'size', 'stride', 'numel'):
                    continue
                elif node.op == 'call_function' and any(token in name for token in
                        ('empty', 'getitem', 'getattr', 'sym_size', 'sym_numel')):
                    continue
                else:
                    raise RuntimeError('Model graph contains non-Triton computation: ' + name)
            if not triton_nodes:
                raise RuntimeError('Model graph has no Triton computation')
            return super().backend_wrapper(graph, example_inputs)
    return TritonCompiler(backend_name='npu-inductor')


def _prepare(root, index=7, variant=0):
    import torch
    import torch_npu
    import candidate
    from model_execution import prepare_model
    from private_campaign_cases import make_case, validate
    case = make_case('matmul', index, variant=variant)
    prepared = prepare_model(candidate.Model, case['args'], {},
        lambda output: validate(case, 'candidate', output),
        compiler=audited_compiler(), synchronize=torch.npu.synchronize)
    return prepared, case


def check(root, env, remaining):
    import torch
    import torch_npu
    from candidate_probe import probe_candidate
    from private_campaign_cases import make_case, validate
    try:
        import candidate
    except Exception as error:
        owned = Path(root / 'candidate.py').resolve()
        line = getattr(error, 'lineno', None) if getattr(error, 'filename', None) == str(owned) else None
        trace = error.__traceback__
        while trace:
            if Path(trace.tb_frame.f_code.co_filename).resolve() == owned:
                line = trace.tb_lineno
            trace = trace.tb_next
        detail = dict(kind='candidate-import', error_type=type(error).__name__)
        if line:
            detail.update(file='candidate.py', line=line)
            if isinstance(error, (SyntaxError, NameError)):
                detail['message'] = (error.msg if isinstance(error, SyntaxError) else str(error))[:4096]
        return dict(passed=False, variants=[], diagnostics=[dict(state='failed',
            category='candidate-compile-or-runtime-error', diagnostic=detail)])
    # Tiny, independent M/N/K tails, and a real performance shape, two data variants.
    results = []
    for variant in (0, 1):
        for index in range(8):
            case = make_case('matmul', index, variant=variant)
            # Each proof gets an actual new backend capture. Neither Dynamo's
            # eight-specialization limit nor a warm Python cache substitutes for it.
            torch._dynamo.reset()
            result = probe_candidate(candidate.Model, lambda: [(case['args'], {})],
                lambda output, args, kwargs: validate(case, 'candidate', output),
                candidate_file=root / 'candidate.py',
                evidence_dir=root / 'diagnostics' / str(variant) / str(index),
                backend='npu-inductor', synchronize=torch.npu.synchronize,
                reference=lambda args, kwargs: case['expected'], compiler=audited_compiler())
            results.append(result)
            if result['state'] != 'passed':
                return dict(passed=False, variants=list(range(variant)), diagnostics=results)
    return dict(passed=True, variants=[0, 1], diagnostics=results)


def benchmark(root, env, remaining):
    import torch
    prepared, case = _prepare(root)
    for _ in range(20):
        prepared()
    torch.npu.synchronize()
    values = []
    for _ in range(3):
        started = time.perf_counter_ns()
        for _ in range(20):
            prepared()
        torch.npu.synchronize()
        values.append((time.perf_counter_ns() - started) / 1000 / 20)
    from profile_capture import Acquisition
    from profile_accounting import account_profile
    capture = Acquisition().profile(prepared, samples=3, output=root / 'measurement/timing-trace')
    accounting = account_profile(capture['iterations'], capture['tasks'], capture_complete=True)
    return dict(state='complete', cases=[case['id']], warmup=20, samples=3,
        busy_time=dict(metric=accounting['metric'],
                       samples_us=[row['device_busy_us'] for row in accounting['iterations']],
                       active_union_samples_us=[row['device_active_union_us'] for row in accounting['iterations']],
                       source='explicitly correlated warmed full-forward profiler traces'),
        latency=dict(metric='warm synchronized full-forward wall latency',
                     samples_us=values, median_us=statistics.median(values),
                     compile_time_included=False))


def _rows(folder, pattern):
    rows = []
    for path in sorted(folder.rglob(pattern)):
        with path.open(encoding='utf-8-sig') as stream:
            rows.extend(csv.DictReader(stream))
    return rows


def _name(row):
    names = {value for key, value in row.items() if value and 'name' in key.lower()
             and ('kernel' in key.lower() or 'op' in key.lower())}
    if len(names) != 1:
        raise RuntimeError('ambiguous exported kernel name')
    return names.pop()


def pipes(root, env, remaining):
    output = root / 'measurement'
    output.mkdir(exist_ok=True)
    wrapper = root / 'application.sh'
    wrapper.write_text('#!/bin/sh\nexec ' + shlex.join([
        sys.executable, str(root / 'scripts/model_campaign_native.py'), str(root)]) + '\n')
    wrapper.chmod(0o700)
    def capture(label, metric, selector=None):
        folder = output / label
        command = ['msprof', 'op', '--application=' + str(wrapper),
            '--output=' + str(folder / 'report'), '--aic-metrics=' + metric,
            '--warm-up=21', '--launch-count=' + ('5000' if metric == 'BasicInfo' else '1'), '--kill=off']
        if selector:
            command.append('--kernel-name=' + selector)
        process = execute(command, folder, env, timeout=remaining())
        validate_capture(process.returncode, process.stdout + process.stderr)
        return folder / 'report'
    # Discovery must be scoped to the warmed candidate, not reference/setup tasks
    # that a whole-process msprof BasicInfo capture would also observe.
    from profile_capture import Acquisition
    prepared, _ = _prepare(root)
    discovered = Acquisition().profile(prepared, samples=1, output=output / 'discovery')
    names = sorted({row['name'] for row in discovered['tasks'] if row['kind'] == 'device'})
    if not names:
        raise RuntimeError('compiled candidate exported no kernel names')
    mapping = {}
    for index, hint in enumerate(names):
        basic = capture('basic-' + str(index), 'BasicInfo', hint)
        mapping[hint] = sorted({_name(row) for row in _rows(basic, '*OpBasicInfo*.csv')})
        if not mapping[hint]:
            raise RuntimeError('candidate trace hint exported no canonical BasicInfo selector')
    selectors = sorted({name for values in mapping.values() for name in values})
    captures = []
    for index, selector in enumerate(selectors):
        folder = capture('pipes-' + str(index), 'PipeUtilization', selector)
        basic, rows = _rows(folder, '*OpBasicInfo*.csv'), _rows(folder, '*PipeUtilization*.csv')
        if not rows or not basic or any(_name(row) != selector for row in basic):
            raise RuntimeError('exact-selector pipe export incomplete')
        captures.append(dict(selector=selector, basic_rows=basic, pipe_rows=rows))
    result = dict(state='complete', cases=['matmul-7-variant-0'],
                  discovered_names=selectors, selector_mapping=mapping, captures=captures, warmup=21)
    (output / 'pipe-evidence.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    root = Path(sys.argv[1]).resolve()
    sys.path[:0] = [str(root), str(root / 'scripts'),
                   str(root / 'skills/ascend-model-benchmarking/scripts')]
    import torch
    prepared, _ = _prepare(root)
    # Cold preparation is the first launch; profiler skips it plus20 warm launches.
    with torch.inference_mode():
        for _ in range(21):
            prepared()
        torch.npu.synchronize()
