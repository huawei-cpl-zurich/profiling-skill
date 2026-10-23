from argparse import Namespace
from contextlib import nullcontext
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import model_benchmark_profiled as controller

COUNTS = {}
def baseline():
    def forward(value):
        COUNTS['baseline'] += 1
        return value + 1
    return forward

def candidate():
    def forward(value):
        COUNTS['candidate'] += 1
        return value + 1
    return forward

def changed_baseline():
    return baseline()

def cases():
    COUNTS['provider'] += 1
    value = torch.tensor(2.)
    return [dict(id='scalar', args=(value,), expected=baseline()(value), private_snapshot=value.clone())]

def validator(case, arm, value):
    COUNTS['validate'] += 1
    torch.testing.assert_close(value, case['expected'])

class Prepared:
    def __init__(self, factory, args, kwargs, validate, **unused):
        self.call = factory(); self.args = args; self.kwargs = kwargs
        self.compiler = N(executions=0, graphs=[], snapshot=lambda: (1, 0), monitor_compilation=nullcontext)
        self.evidence = dict(fullgraph=True)
        validate(self())
    def __call__(self):
        self.compiler.executions += 1
        return self.call(*self.args, **self.kwargs)

def capture(call, output, **unused):
    COUNTS['capture'] += 1
    for _ in range(10):
        call()
    return dict(event_latency=dict(status='complete', count=10, samples_ms=[1.] * 10,
                                   mean_ms=1., sample_sd_ms=0.),
                profile=dict(status='failed', error='injected export failure'),
                kernel_identity=dict(status='unavailable'))

class ProtocolTests(unittest.TestCase):
    def setUp(self):
        COUNTS.update(provider=0, baseline=0, candidate=0, validate=0, capture=0)
        self.temporary = tempfile.TemporaryDirectory(); self.root = Path(self.temporary.name)
        self.hooks = dict(baseline=baseline, candidate=self.write_candidate(), cases_provider=cases, validator=validator)
    def write_candidate(self, edited=False, mutating=False):
        path = self.root / 'candidate_source.py'
        expression = '1 + value' if edited else 'value + 1'
        if mutating:
            expression = 'value.add_(1)'
        path.write_text('def candidate():\n    def forward(value):\n        COUNTS[\"candidate\"] += 1\n        return ' + expression + '\n    return forward\n')
        name = 'separate_candidate_' + str(id(self))
        module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, path))
        sys.modules[name] = module
        exec(compile(path.read_text(), str(path), 'exec'), module.__dict__)
        module.COUNTS = COUNTS
        return module.candidate
    def tearDown(self):
        sys.modules.pop('separate_candidate_' + str(id(self)), None)
        self.temporary.cleanup()
    def run_protocol(self, mode, name, **overrides):
        options = Namespace(**{key: key for key in self.hooks}, output=str(self.root / name),
            cache=str(self.root / 'cache'), cache_mode=mode, device='cpu', backend='npu-inductor',
            max_specializations=256)
        with patch.object(controller, 'load', side_effect=self.hooks.__getitem__), \
                patch.object(torch, 'npu', N(synchronize=lambda: None, is_available=lambda: False), create=True):
            return controller.run(options, torch=torch, prepare=overrides.pop('prepare', Prepared),
                compiler_factory=lambda: None, capture=overrides.pop('capture', capture), **overrides)
    def test_create_reuse_exact_counts_and_export_failure_separation(self):
        first = self.run_protocol('create', 'first')
        self.assertEqual(first['state'], 'complete')
        self.assertEqual(first['profile_state'], 'incomplete')
        self.assertEqual(first['candidate_identity_state'], 'qualified')
        self.assertEqual(COUNTS, dict(provider=1, baseline=11, candidate=11, validate=1, capture=2))
        self.hooks['candidate'] = self.write_candidate(edited=True)
        reused = self.run_protocol('reuse', 'second')
        self.assertEqual(reused['state'], 'complete')
        self.assertNotEqual(first['sources']['candidate']['source_sha256'],
                            reused['sources']['candidate']['source_sha256'])
        self.assertEqual(COUNTS, dict(provider=1, baseline=11, candidate=22, validate=2, capture=3))
        self.assertTrue(reused['cases'][0]['baseline_reused'])
        self.assertEqual(reused['cases'][0]['candidate']['event_latency']['count'], 10)
        self.assertFalse(reused['comparability']['comparable'])
        self.assertNotIn('speedup', reused['cases'][0])
    def test_shared_candidate_source_is_rejected_before_provider_or_forwards(self):
        self.hooks['candidate'] = candidate
        result = self.run_protocol('create', 'same-source')
        self.assertEqual(result['state'], 'failed')
        self.assertIn('separate source file', result['error'])
        self.assertEqual(COUNTS, dict(provider=0, baseline=0, candidate=0, validate=0, capture=0))
    def test_nondense_and_broadcast_cache_layout_survives_transfer(self):
        value = torch.arange(12.).reshape(3, 4)[:, 1::2].requires_grad_(True)
        meta = controller.move_tree(value, torch, 'meta')
        self.assertTrue(meta.requires_grad)
        self.assertEqual((meta.shape, meta.stride(), meta.storage_offset()),
                         (value.shape, value.stride(), value.storage_offset()))
        broadcast = torch.ones(1).expand(4)
        meta = controller.move_tree(broadcast, torch, 'meta')
        self.assertEqual(meta.stride(), (0,))
        def sliced_cases():
            COUNTS['provider'] += 1
            return [dict(id='slice', args=(value,), expected=baseline()(value))]
        self.hooks['cases_provider'] = sliced_cases
        original_to = torch.Tensor.to
        def copied(tensor, device, **kwargs):
            return original_to(tensor, device, copy=True)
        with patch.object(torch.Tensor, 'to', copied):
            first = self.run_protocol('create', 'sliced')
            reused = self.run_protocol('reuse', 'sliced-reused')
        self.assertEqual(first['state'], 'complete', first.get('error'))
        self.assertEqual(reused['state'], 'complete', reused.get('error'))
        payload = torch.load(self.root / 'cache/case-0000.pt', weights_only=True)
        restored = payload['args'][0]
        torch.testing.assert_close(restored, value)
        self.assertTrue(restored.requires_grad)
        self.assertEqual((restored.stride(), restored.storage_offset()),
                         (value.stride(), value.storage_offset()))
        self.assertEqual((COUNTS['provider'], COUNTS['baseline']), (1, 11))
    def test_aliased_expected_is_frozen_before_candidate_validation(self):
        def identity_baseline():
            def forward(value):
                COUNTS['baseline'] += 1
                return value
            return forward
        def aliased_cases():
            COUNTS['provider'] += 1
            value = torch.tensor(2.)
            return [dict(id='alias', args=(value,), expected=identity_baseline()(value))]
        self.hooks.update(baseline=identity_baseline, cases_provider=aliased_cases,
                          candidate=self.write_candidate(mutating=True))
        result = self.run_protocol('create', 'aliased')
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(COUNTS, dict(provider=1, baseline=1, candidate=1, validate=1, capture=0))
    def test_baseline_uses_original_inputs_after_candidate_mutations(self):
        seen = []
        def recording_baseline():
            def forward(value):
                COUNTS['baseline'] += 1
                seen.append(float(value))
                return value + 1
            return forward
        def recorded_cases():
            COUNTS['provider'] += 1
            value = torch.tensor(2.)
            return [dict(id='mutating', args=(value,), expected=recording_baseline()(value))]
        self.hooks.update(baseline=recording_baseline, cases_provider=recorded_cases,
                          candidate=self.write_candidate(mutating=True))
        result = self.run_protocol('create', 'restored')
        self.assertEqual(result['state'], 'complete', result.get('error'))
        self.assertEqual(seen, [2.] * 11)
        self.assertEqual(COUNTS, dict(provider=1, baseline=11, candidate=11, validate=1, capture=2))
    def test_cli_selects_explicit_cache_mode_and_fixed_protocol(self):
        args = ['--baseline', 'x:baseline', '--candidate', 'x:candidate',
            '--cases-provider', 'x:cases', '--validator', 'x:validator',
            '--output', 'new', '--cache', 'private', '--cache-mode', 'reuse']
        with patch.object(controller, 'run', return_value={'state': 'complete'}) as run:
            self.assertEqual(controller.main(args), 0)
            self.assertEqual(run.call_args.args[0].cache_mode, 'reuse')
            self.assertEqual(run.call_args.args[0].backend, 'npu-inductor')
        with patch.object(controller, 'run') as run, self.assertRaises(SystemExit):
            controller.main(args[:-1] + ['unknown'])
        run.assert_not_called()
    def test_bad_correctness_stops_before_measured_forwards(self):
        def reject(*args):
            raise AssertionError('wrong output')
        self.hooks['validator'] = reject
        result = self.run_protocol('create', 'bad')
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(COUNTS['candidate'], 1)
        self.assertEqual(COUNTS['capture'], 0)
        self.assertTrue((self.root / 'cache/case-0000.pt').is_file())
    def test_source_and_payload_changes_reject_cache_without_baseline_rerun(self):
        self.run_protocol('create', 'first')
        self.hooks['baseline'] = changed_baseline
        self.assertEqual(self.run_protocol('reuse', 'changed')['state'], 'failed')
        self.hooks['baseline'] = baseline
        payload = self.root / 'cache/case-0000.pt'
        original = payload.read_bytes()
        payload.write_bytes(b'changed')
        self.assertEqual(self.run_protocol('reuse', 'corrupt')['state'], 'failed')
        payload.write_bytes(original)
        (self.root / 'cache/baseline-0000.json').write_text('{}')
        self.assertEqual(self.run_protocol('reuse', 'profile-corrupt')['state'], 'failed')
        self.assertEqual(COUNTS['provider'], 1)
        self.assertEqual(COUNTS['candidate'], 11)
        self.assertEqual(COUNTS['baseline'], 11)
    def test_incomplete_samples_fail(self):
        with patch.object(controller, 'measured', side_effect=RuntimeError('nine samples')):
            self.assertEqual(self.run_protocol('create', 'partial')['state'], 'failed')
        self.assertEqual(COUNTS['baseline'], 1)
    def test_measured_recompile_fails(self):
        def recompile(call, output, **kwargs):
            result = capture(call, output, **kwargs)
            for cell in call.__closure__:
                if isinstance(cell.cell_contents, Prepared):
                    cell.cell_contents.compiler.snapshot = lambda: (2, 0)
            return result
        result = self.run_protocol('create', 'recompiled', capture=recompile)
        self.assertEqual(result['state'], 'failed')
        self.assertIn('compilation', result['error'])
        self.assertEqual(COUNTS['baseline'], 1)
    def test_more_than_eight_real_fullgraph_specializations_and_config_restored(self):
        # Exercise the same Python forward code at ten different static shapes.
        def shapes():
            COUNTS['provider'] += 1
            for size in range(1, 11):
                value = torch.ones(size)
                yield dict(id=str(size), args=(value,), expected=baseline()(value))
        self.hooks['cases_provider'] = shapes
        class Compiler:
            def __init__(self):
                self.executions = 0; self.compilations = 0; self.graphs = []
                self.backend_name = 'cpu-counting-test'; self.last_failure = None
                self.torch = torch; self.triton_compilations = 0
            def backend_wrapper(self, graph, inputs):
                self.compilations += 1; self.graphs.append(graph.code)
                def execute(*args):
                    self.executions += 1
                    return graph(*args)
                return execute
            def snapshot(self): return self.compilations, 0
            def monitor_compilation(self): return nullcontext()
        from model_execution import prepare_model
        def real_prepare(factory, args, kwargs, validate, **unused):
            # Avoid Python counter side effects inside graph capture.
            return prepare_model(lambda: (lambda value: value + 1), args, kwargs, validate,
                                 compiler=Compiler(), synchronize=lambda: None)
        config = torch._dynamo.config
        key = 'recompile_limit' if hasattr(config, 'recompile_limit') else 'cache_size_limit'
        before = getattr(config, key)
        with config.patch({key: 8}):
            result = self.run_protocol('create', 'ten-shapes', prepare=real_prepare)
            self.assertEqual(getattr(config, key), 8)
        self.assertEqual(result['state'], 'complete', result.get('error'))
        self.assertEqual(len(result['cases']), 10)
        self.assertEqual(getattr(config, key), before)
        torch._dynamo.reset()

if __name__ == '__main__':
    unittest.main()
