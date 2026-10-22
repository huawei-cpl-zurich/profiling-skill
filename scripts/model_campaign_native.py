"""Private native matmul Model adapter. Never mount this into optimizer folders."""
import csv
import json
import math
from pathlib import Path
import statistics
import time


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
    except TimeoutError:
        raise
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
        discovered_names=sorted({row['name'] for row in capture['tasks'] if row['kind'] == 'device'}),
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


def validate_pmu_captures(captures, samples=5):
    """Require distinct attributable tasks with available finite pipe evidence."""
    if not captures:
        raise RuntimeError('application PMU export incomplete')
    for capture in captures:
        identities = set()
        for row in capture['pipe_rows']:
            identity = tuple(str(row.get(key, '')).strip()
                             for key in ('Device_id', 'Stream ID', 'Task ID'))
            if any(value in ('', 'N/A', 'NA', '-') for value in identity):
                continue
            for key, value in row.items():
                if not key.endswith('_ratio'):
                    continue
                try:
                    usable = math.isfinite(float(value))
                except (TypeError, ValueError):
                    usable = False
                if usable:
                    identities.add(identity)
                    break
        if len(identities) < samples:
            raise RuntimeError('application PMU export incomplete')


def pipes(root, env, remaining):
    """Application-level per-task PMU; deliberately not op-replay block PMU."""
    import torch
    import torch_npu
    from torch_npu.profiler.experimental_config import AiCMetrics
    from profile_capture import Acquisition
    output = root / 'measurement'
    output.mkdir(exist_ok=True)
    prepared, _ = _prepare(root)
    discovered = Acquisition().profile(prepared, samples=1, output=output / 'discovery')
    names = sorted({row['name'] for row in discovered['tasks'] if row['kind'] == 'device'})
    if not names:
        raise RuntimeError('compiled candidate exported no kernel names')
    profiler = torch_npu.profiler
    with torch.inference_mode(), prepared.compiler.monitor_compilation():
        for _ in range(20):
            prepared()
        torch.npu.synchronize()
        before = prepared.compiler.snapshot()
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                experimental_config=profiler._ExperimentalConfig(
                    profiler_level=profiler.ProfilerLevel.Level1,
                    aic_metrics=AiCMetrics.PipeUtilization, data_simplification=False),
                schedule=profiler.schedule(wait=0, warmup=0, active=5, repeat=1),
                on_trace_ready=profiler.tensorboard_trace_handler(str(output / 'pmu'))) as capture:
            for _ in range(5):
                remaining()
                prepared()
                torch.npu.synchronize()
                if prepared.compiler.snapshot() != before:
                    raise RuntimeError('compilation during PMU measurement')
                capture.step()
    rows = _rows(output / 'pmu', '*kernel_details.csv')
    captures = [dict(selector=name, basic_rows=[],
        pipe_rows=[row for row in rows if row.get('Name') == name]) for name in names]
    validate_pmu_captures(captures)
    result = dict(state='complete', cases=['matmul-7-variant-0'],
        collector='Torch-NPU application PMU', granularity='per-task',
        discovered_names=names, captures=captures, warmup=20, samples=5)
    (output / 'pipe-evidence.json').write_text(json.dumps(result, indent=2))
    return result
