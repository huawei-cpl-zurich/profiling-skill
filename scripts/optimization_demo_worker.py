"""Trusted single-candidate worker: one check, then ten profiled forwards per case."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import fcntl
import os
import subprocess
import time
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/ascend-model-benchmarking/scripts'))
from optimization_pair import check_capture
from optimization_profile_evidence import operation_windows, pmu_rows
from profile_ten import capture_ten


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


class InfrastructureError(RuntimeError):
    pass


class LaunchAudit:
    """Derive names and binary identity from actual candidate-owned JIT launches."""

    def __init__(self, source, jit_class, *, allow_launch=True):
        self.source, self.jit_class = Path(source).resolve(), jit_class
        self.allow_launch = allow_launch
        self.seen = {}

    def __enter__(self):
        original = self.original = self.jit_class.run
        def run(kernel, *args, **kwargs):
            function = kernel.fn
            if Path(function.__code__.co_filename).resolve() != self.source:
                raise ValueError('device arithmetic must use submitted candidate-owned kernels')
            if not self.allow_launch and not kwargs.get('warmup', False):
                raise ValueError('candidate import cannot launch a kernel')
            compiled = original(kernel, *args, **kwargs)
            if not kwargs.get('warmup', False):
                artifacts = {key: hashlib.sha256(value).hexdigest()
                             for key, value in compiled.asm.items() if isinstance(value, bytes)}
                if not artifacts:
                    raise ValueError('compiled candidate binary identity is unavailable')
                identity = dict(name=compiled.name, hash=str(compiled.hash), artifacts=artifacts)
                self.seen[(function.__name__, str(compiled.hash))] = identity
            return compiled
        self.jit_class.run = run
        try:
            from triton.runtime.autotuner import Autotuner
            self.autotuner, self.autotune_run = Autotuner, Autotuner.run
            def forbidden(*args, **kwargs):
                raise ValueError('autotuning would violate the fixed one-check/ten-forward protocol')
            Autotuner.run = forbidden
        except ImportError:
            self.autotuner = None
        return self

    def snapshot(self):
        return [self.seen[key] for key in sorted(self.seen)]

    def names(self):
        return tuple(sorted({key[0] for key in self.seen}))

    def __exit__(self, *args):
        self.jit_class.run = self.original
        if self.autotuner is not None: self.autotuner.run = self.autotune_run


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_cases(request, output, *, torch=None, jit_class=None, capture=capture_ten,
                   synchronize=None, device='npu:0'):
    """Private fixtures are trusted hash-bound CPU payloads, never actor files."""
    torch = torch or __import__('torch')
    if jit_class is None:
        from triton.runtime.jit import JITFunction
        jit_class = JITFunction
    synchronize = synchronize or torch.npu.synchronize
    source = Path(request['candidate_path']).resolve()
    if sha(source) != request['candidate_sha256']:
        raise ValueError('submitted source digest differs')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if sha(request['provider_path']) != request['provider_sha256']:
        raise InfrastructureError('fixture provider changed')
    validator = load_module(request['provider_path'], 'trusted_fixture_provider')
    report = dict(schema='optimization-coding-evaluation-v1',
                  candidate_sha256=sha(source), measurement_kind='profiled_device_operation_span_us',
                  outcome='success', cases=[dict(id=c['id'], correct=False, identity_verified=False,
                  correctness_calls=0, profiled_calls=0, status='not-run') for c in request['cases']],
                  new_oracle_calls=0)
    try:
        with LaunchAudit(source, jit_class, allow_launch=False):
            module = load_module(source, 'submitted_' + sha(source))
    except Exception as error:
        report.update(outcome='candidate_failure', error=str(error))
        save(output / 'result.json', report)
        return report
    for case, row in zip(request['cases'], report['cases']):
        folder = output / case['id']
        folder.mkdir()
        try:
            if sha(case['fixture_path']) != case['fixture_sha256']:
                raise InfrastructureError('private fixture changed')
            fixture = torch.load(case['fixture_path'], map_location='cpu', weights_only=True)
            inputs = {key: value.to(device=device, memory_format=torch.preserve_format)
                      for key, value in fixture['inputs'].items()}
            if any(inputs[key].stride() != value.stride() for key, value in fixture['inputs'].items()):
                raise InfrastructureError('device transfer changed a required input stride')
            if str(device).startswith('npu'):
                import torch_npu
                row['input_metadata'] = {key:dict(shape=list(value.shape),strides=list(value.stride()),
                    format=torch_npu.get_npu_format(value),storage_bytes=value.untyped_storage().nbytes())
                    for key,value in inputs.items()}
                if any(item['format'] != 2 for item in row['input_metadata'].values()):
                    raise InfrastructureError('candidate inputs require verified physical ND format')
            with LaunchAudit(source, jit_class) as audit:
                row['correctness_calls'] = 1
                actual = module.run(inputs)
                synchronize()
                diagnostic = validator.validate_actual(actual, fixture, request['operation_id'], inputs)
                row['diagnostics'] = diagnostic
                if diagnostic.get('passed') is not True:
                    raise ValueError('candidate correctness or input immutability rejected')
                row['correct'] = True
                expected, compiled = audit.names(), audit.snapshot()
                if not expected:
                    raise ValueError('no candidate-owned compiled kernel was executed')
                calls = 0
                def forward():
                    nonlocal calls
                    if calls >= 10:
                        raise ValueError('capture attempted an extra forward')
                    calls += 1
                    return module.run(inputs)
                try:
                    captured = capture(forward, folder / 'capture', expected_kernel=expected,
                                       metric_group='PipeUtilization')
                except Exception as error:
                    # Export/schema failures do not establish a model's performance failure.
                    from profile_capture import CaptureError
                    if isinstance(error, CaptureError):
                        raise InfrastructureError(str(error)) from error
                    raise
                finally:
                    row['profiled_calls'] = calls
                row['capture'] = captured
                if audit.snapshot() != compiled or sha(source) != request['candidate_sha256']:
                    raise ValueError('source or compiled binary changed during capture')
                try:
                    complete = check_capture(captured, expected, calls)
                except Exception as error:
                    raise InfrastructureError('profile recount failed: ' + str(error)) from error
                if not complete:
                    raise InfrastructureError('profile export, ownership or identity is incomplete')
                if set(captured['kernel_identity']['names']) != set(expected):
                    raise ValueError('profile contains arithmetic outside the submitted kernels')
                windows = operation_windows(captured['operation_accounting'])
                row.update(identity_verified=True, status='complete', compiled=compiled,
                           candidate_samples_us=[w['device_operation_span_us'] for w in windows],
                           baseline_samples_us=case.get('baseline_samples_us', []))
                csv_path = Path(captured['profile']['vendor_trace']).parent / 'kernel_details.csv'
                if csv_path.is_file():
                    try:
                        metrics = pmu_rows(csv_path, captured['operation_accounting']['tasks'], request['physical_die'])
                        row['metrics'] = dict(status=metrics['status'], kernels={name:data['numeric_columns']
                                              for name,data in metrics.get('kernels',{}).items()})
                    except Exception as error:
                        row['metrics'] = dict(status='unavailable',reason=str(error))
                else:
                    row['metrics'] = dict(status='unavailable')
        except Exception as error:
            row.update(status='failed', error_type=type(error).__name__, error=str(error),
                       traceback=traceback.format_exc())
            report['outcome'] = 'infrastructure_failure' if isinstance(error, InfrastructureError) else 'candidate_failure'
        save(folder / 'result.json', row)
        if row['status'] != 'complete':
            break
    save(output / 'result.json', report)
    return report


def remote_main(request):
    """Lease the selected physical device before importing its backend."""
    started = time.monotonic()
    handles = []
    try:
        locks = Path(request['lease_directory']); locks.mkdir(parents=True, exist_ok=True)
        for name in ('die-' + str(request['physical_die']),):
            handle = (locks / (name + '.lock')).open('a')
            fcntl.flock(handle, fcntl.LOCK_EX); handles.append(handle)
        query = request['device_query']; health = []
        for kind in ('health', 'usages', 'memory', 'proc-mem'):
            result = subprocess.run(['npu-smi','info','-t',kind,'-i',str(query['card']),
                                     '-c',str(query['chip'])], capture_output=True,text=True,timeout=5)
            health.append(dict(kind=kind,exit=result.returncode,stdout=result.stdout,stderr=result.stderr))
            if result.returncode: raise InfrastructureError('device query unavailable')
        if ': OK' not in health[0]['stdout'] or 'No process in device' not in health[3]['stdout']:
            raise InfrastructureError('device is unhealthy or occupied')
        rates = [line for line in health[1]['stdout'].splitlines() if 'Aicore Usage Rate' in line or 'Aivector Usage Rate' in line]
        if not rates or any(float(line.split(':')[-1]) != 0 for line in rates):
            raise InfrastructureError('timing requires a verified idle device')
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(request['physical_die'])
        import torch, torch_npu, triton
        torch.set_num_threads(2); torch.npu.set_device(0)
        runtime = dict(torch=torch.__version__,torch_npu=torch_npu.__version__,
                       triton=triton.__version__,product=torch.npu.get_device_properties(0).name)
        if any(runtime.get(k) != v for k,v in request.get('runtime_identity',{}).items()):
            raise InfrastructureError('runtime identity changed since baseline qualification')
        original = capture_ten
        exports = Path(request.get('export_directory',str(locks/'exports')))
        exports.mkdir(parents=True,exist_ok=True)
        def serialized_capture(*args, **kwargs):
            slots = [(exports / (str(i) + '.lock')).open('a') for i in range(2)]
            try:
                while True:
                    for slot in slots:
                        try: fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError: continue
                        try: return original(*args, **kwargs)
                        finally: fcntl.flock(slot, fcntl.LOCK_UN)
                    time.sleep(.1)
            finally:
                for slot in slots: slot.close()
        report = evaluate_cases(request, request['output_directory'], capture=serialized_capture)
        report.update(health=health, runtime_identity=runtime, elapsed_seconds=time.monotonic()-started)
    except Exception as error:
        report = dict(schema='optimization-coding-evaluation-v1',candidate_sha256=request['candidate_sha256'],
                      measurement_kind='profiled_device_operation_span_us',outcome='infrastructure_failure',
                      error=str(error),cases=[dict(id=c['id'],correct=False,identity_verified=False,status='not-run') for c in request['cases']])
    finally:
        for handle in handles: handle.close()
    # Full captures remain remotely; tool responses contain only compact evidence.
    for row in report['cases']: row.pop('capture', None)
    print('DEMO_RESULT ' + json.dumps(report, allow_nan=False), flush=True)
    return report
