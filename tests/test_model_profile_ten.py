from contextlib import nullcontext
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'skills/ascend-model-benchmarking/scripts'))
import profile_ten

class CaptureTests(unittest.TestCase):
    def capture(self, export_error=False, missing=False, expected=True, metrics=None):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'capture'
            calls, events, syncs, markers, lifecycle = [], [], [], [], []
            scheduled = object()
            class Event:
                def __init__(self, **kwargs): events.append(kwargs)
                def record(self): pass
                def elapsed_time(self, end): return .5
            class Profile:
                def __enter__(self):
                    lifecycle.append('start')
                    return self
                def step(self):
                    lifecycle.append('step/raw')
                    saved = json.loads((output / 'latency.json').read_text())
                    self_test.assertEqual(saved['samples_ms'], [.5] * 10)
                    self_test.assertEqual(len(calls), 10)
                    self_test.assertEqual(len(events), 20)
                    self_test.assertEqual(syncs, [True])
                    self_test.assertEqual(markers, [f'model_iteration_{i}' for i in range(10)])
                    (output / 'vendor').mkdir()
                def __exit__(self, *args):
                    lifecycle.append('stop')
                    self_test.assertEqual(lifecycle, ['start', 'step/raw', 'stop'])
            self_test = self
            def begin(**kwargs):
                self.assertIs(kwargs['schedule'], scheduled)
                self.assertEqual(kwargs['experimental_config'].get('aic_metrics'), metrics)
                return Profile()
            def schedule(**kwargs):
                self.assertEqual(kwargs, dict(wait=0, warmup=0, active=1, repeat=1))
                return scheduled
            def handler(path, *, analyse_flag):
                self.assertFalse(analyse_flag)
                return None
            def analyse(path, *, max_process_number):
                self.assertEqual(lifecycle, ['start', 'step/raw', 'stop'])
                self.assertEqual(max_process_number, 1)
                self.assertEqual(json.loads((output / 'latency.json').read_text())['count'], 10)
                lifecycle.append('analyse/export')
                if export_error: raise RuntimeError('export failed after samples saved')
                events = []
                for i in range(10):
                    base = i * 1000
                    name = 'wrong' if missing and i == 9 else 'kernel'
                    events.extend([
                        dict(name=f'model_iteration_{i}', pid=1, tid=1, ts=base, dur=900, ph='X', cat='cpu_op'),
                        dict(name=name, pid=1, tid=1, ts=base+20, dur=5, ph='X', cat='cpu_op'),
                        dict(name=name, pid=2, tid=2, ts=base+100, dur=10, ph='X',
                             args={'Task Type': 'KERNEL_AIVEC', 'Physic Stream Id': 2, 'Task Id': i,
                                   'Batch Id': 0, 'Subtask Id': 0}),
                        dict(name='torch_to_npu', pid=1, tid=1, ts=base+20, ph='s', cat='async_npu', id=i),
                        dict(name='torch_to_npu', pid=2, tid=2, ts=base+100, ph='f', cat='async_npu', id=i)])
                (Path(path) / 'trace_view.json').write_text(json.dumps(dict(traceEvents=events)))
            torch = N(inference_mode=nullcontext, npu=N(Event=Event, synchronize=lambda: syncs.append(True)),
                autograd=N(profiler=N(record_function=lambda name: (markers.append(name) or nullcontext()))))
            npu = N(profiler=N(profile=begin, schedule=schedule, ProfilerActivity=N(CPU=1, NPU=2),
                _ExperimentalConfig=lambda **kwargs: kwargs, ProfilerLevel=N(Level1=1),
                AiCMetrics=N(PipeUtilization='pipe'),
                tensorboard_trace_handler=handler))
            with patch.dict(sys.modules, {'torch_npu.profiler.profiler': N(analyse=analyse)}):
                result = profile_ten.capture_ten(lambda: calls.append(True), output,
                    expected_kernel=['kernel'] if expected else None, torch_module=torch, npu_module=npu,
                    metric_group='PipeUtilization' if metrics else None)
            self.assertEqual(lifecycle, ['start', 'step/raw', 'stop', 'analyse/export'])
            self.assertEqual(result['profile']['analysis_calls'], 1)
            self.assertEqual(result['event_latency']['count'], 10)
            self.assertEqual(result['event_latency']['sample_sd_ms'], 0)
            if not export_error:
                normalized = json.loads((output / 'normalized.json').read_text())
                accounted = json.loads((output / 'accounting.json').read_text())
                self.assertNotIn('iteration_id', normalized['tasks'][0])
                self.assertEqual(accounted['tasks'][0]['iteration_id'], 'model_iteration_0')
                self.assertEqual(result['profile']['export_status'], 'complete')
                self.assertEqual(result['profile']['accounting_status'], 'complete')
            return result
    def test_exact_ten_save_before_export_and_profile_errors_separate(self):
        for error in (False, True):
            result = self.capture(export_error=error)
            self.assertEqual(result['event_latency']['status'], 'complete')
            self.assertEqual(result['profile']['status'], 'failed' if error else 'complete')
    def test_identity_requires_every_iteration_not_union_of_names(self):
        self.assertEqual(self.capture(missing=True)['kernel_identity']['status'], 'failed')
        self.assertEqual(self.capture(expected=False)['kernel_identity']['status'], 'observed-only')
    def test_requested_metrics_reach_vendor_configuration(self):
        result = self.capture(metrics='pipe')
        self.assertEqual(result['metrics'], dict(status='requested', requested='PipeUtilization', available_status='unverified'))
    def test_empty_expected_names_cannot_claim_verified_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            calls = []
            with self.assertRaises(ValueError):
                profile_ten.capture_ten(lambda: calls.append(True), Path(temporary) / 'new',
                    expected_kernel=[], torch_module=N(), npu_module=N())
            self.assertEqual(calls, [])
    def test_resolver_uses_live_kernel_side_table_without_execution(self):
        def actual_kernel(): pass
        module = 'torch._higher_order_ops.triton_kernel_wrap'
        with patch.dict(sys.modules, {module: N(kernel_side_table=N(get_kernel=lambda index: N(fn=actual_kernel)))}):
            self.assertEqual(profile_ten.resolve_triton_names(['call(kernel_idx = 7)']), ['actual_kernel'])
            self.assertIsNone(profile_ten.resolve_triton_names(['ordinary graph']))
        with patch.dict(sys.modules, {module: N(kernel_side_table=N(get_kernel=lambda index: N()))}):
            self.assertIsNone(profile_ten.resolve_triton_names(['call(kernel_idx=7)']))

if __name__ == '__main__': unittest.main()
