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

def changed_candidate():
    return candidate()

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
        self.hooks = dict(baseline=baseline, candidate=candidate, cases_provider=cases, validator=validator)
    def tearDown(self):
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
        self.hooks['candidate'] = changed_candidate
        reused = self.run_protocol('reuse', 'second')
        self.assertEqual(reused['state'], 'complete')
        self.assertEqual(COUNTS, dict(provider=1, baseline=11, candidate=22, validate=2, capture=3))
        self.assertTrue(reused['cases'][0]['baseline_reused'])
        self.assertEqual(reused['cases'][0]['candidate']['event_latency']['count'], 10)
        self.assertFalse(reused['comparability']['comparable'])
        self.assertNotIn('speedup', reused['cases'][0])
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
