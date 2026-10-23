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
    def capture(self, export_error=False, missing=False, expected=True):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'capture'
            calls, events, syncs, markers = [], [], [], []
            class Event:
                def __init__(self, **kwargs): events.append(kwargs)
                def record(self): pass
                def elapsed_time(self, end): return .5
            class Profile:
                def __enter__(self): return self
                def __exit__(self, *args):
                    saved = json.loads((output / 'latency.json').read_text())
                    self_test.assertEqual(saved['samples_ms'], [.5] * 10)
                    self_test.assertEqual(len(calls), 10)
                    self_test.assertEqual(len(events), 20)
                    self_test.assertEqual(syncs, [True])
                    self_test.assertEqual(markers, [f'model_iteration_{i}' for i in range(10)])
                    if export_error: raise RuntimeError('export failed after samples saved')
                    path = output / 'vendor/trace_view.json'; path.parent.mkdir(); path.write_text('[]')
            self_test = self
            def begin(**kwargs):
                self.assertNotIn('schedule', kwargs)
                return Profile()
            torch = N(inference_mode=nullcontext, npu=N(Event=Event, synchronize=lambda: syncs.append(True)),
                autograd=N(profiler=N(record_function=lambda name: (markers.append(name) or nullcontext()))))
            npu = N(profiler=N(profile=begin, ProfilerActivity=N(CPU=1, NPU=2),
                _ExperimentalConfig=lambda **kwargs: kwargs, ProfilerLevel=N(Level1=1),
                tensorboard_trace_handler=lambda path: None))
            def normalized(*args, **kwargs):
                return dict(iterations=[dict(iteration_id=str(i)) for i in range(10)],
                    tasks=[dict(name='wrong' if missing and i == 9 else 'kernel', kind='device',
                                iteration_id=str(i)) for i in range(10)])
            with patch.dict(sys.modules, dict(profile_capture=N(normalize_trace=normalized, CaptureError=ValueError),
                    profile_accounting=N(account_profile=lambda *a, **k: dict(iterations=10)))):
                result = profile_ten.capture_ten(lambda: calls.append(True), output,
                    expected_kernel=['kernel'] if expected else None, torch_module=torch, npu_module=npu)
            self.assertEqual(result['event_latency']['count'], 10)
            self.assertEqual(result['event_latency']['sample_sd_ms'], 0)
            return result
    def test_exact_ten_save_before_export_and_profile_errors_separate(self):
        for error in (False, True):
            result = self.capture(export_error=error)
            self.assertEqual(result['event_latency']['status'], 'complete')
            self.assertEqual(result['profile']['status'], 'failed' if error else 'complete')
    def test_identity_requires_every_iteration_not_union_of_names(self):
        self.assertEqual(self.capture(missing=True)['kernel_identity']['status'], 'failed')
        self.assertEqual(self.capture(expected=False)['kernel_identity']['status'], 'observed-only')
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
