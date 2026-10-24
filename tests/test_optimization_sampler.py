import importlib.util
from pathlib import Path
import sys
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'examples/optimization-library/triton'
sys.path.insert(0, str(DIRECTORY))
from sampler_contract import make_case, reference, qualify_ranking, validate_inputs
from sampler import Adapter
sys.path.insert(0, str(ROOT / 'scripts'))
from optimization_pair import fixture


def spec(**changes):
    return dict(seed=19, parameters=dict(lengths=[2, 0, 1], vocab=37,
                                       draft_probs_present=False, **changes))


class SamplerContractTests(unittest.TestCase):
    def test_integer_reference_matches_scalar_scores(self):
        case = spec()
        value = make_case(case)
        validate_inputs(value['inputs'], case)
        x = value['inputs']
        expected = []
        for row, request in enumerate([0, 0, 2]):
            scores = [0. if col == int(x['ids'][row]) else
                      float(x['target'][row, col]) / float(x['q'][request, col]) for col in range(37)]
            expected.append(max(range(37), key=scores.__getitem__))
        self.assertEqual(value['expected'].tolist(), expected)

    def test_allzero_includes_draft_id_zero(self):
        value = make_case(spec())
        value['inputs']['target'].zero_()
        value['inputs']['ids'].zero_()
        self.assertEqual(reference(value['inputs']).tolist(), [0, 0, 0])

    def test_draft_probability_mode(self):
        case = dict(seed=23, parameters=dict(lengths=[1, 2], vocab=13, draft_probs_present=True))
        value = make_case(case)
        x = value['inputs']
        x['draft'].copy_(x['target'])
        self.assertEqual(reference(x).tolist(), [0, 0, 0])
        validate_inputs(x, case)

    def test_public_prepare_rejects_fp32_rounding_tie(self):
        case = dict(seed=29, lengths=[1], vocab=3, draft_probs_present=False)
        inputs = dict(target=torch.tensor([[.1, .3, .6]]), q=torch.tensor([[3., 9., 1.]]),
                      ids=torch.tensor([2], dtype=torch.int32), ends=torch.tensor([1], dtype=torch.int32))
        expected = reference(inputs)
        self.assertEqual(expected.tolist(), [1])  # Preserve the unique FP64 winner.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'not-created'
            with patch.dict(sys.modules, {'torch_npu': None}):
                with self.assertRaisesRegex(ValueError, 'FP32 scores disagree with FP64 oracle'):
                    Adapter().prepare('before', case, dict(inputs=inputs, expected=expected), output)
            self.assertFalse(output.exists())
        for values in ([.25, .25, .5], [0., 0., 0.]):
            inputs['target'] = torch.tensor([values])
            inputs['q'].fill_(1)
            qualify_ranking(inputs, reference(inputs))  # Exact ties and zero scores remain valid.
        inputs['draft'] = inputs['target'].clone()
        qualify_ranking(inputs, reference(inputs))  # Unchanged draft-probability control.
        inputs = dict(target=torch.zeros(1, 8193), q=torch.ones(1, 8193),
                      ids=torch.tensor([1], dtype=torch.int32), ends=torch.tensor([1], dtype=torch.int32))
        inputs['target'][0, [4095, 8192]] = .5
        self.assertEqual(reference(inputs).tolist(), [4095])
        qualify_ranking(inputs, reference(inputs))

    def test_bad_inputs_rejected_before_backend_import(self):
        case = spec()
        for key, replacement in [('q', torch.zeros(3, 37)), ('ids', torch.full((3,), 37, dtype=torch.int32)),
                                 ('ends', torch.tensor([2, 1, 3], dtype=torch.int32))]:
            value = make_case(case)
            value['inputs'][key] = replacement
            with self.assertRaises(ValueError):
                Adapter().prepare('before', case, value, Path('.'))
        value = make_case(case)
        value['inputs']['target'] = value['inputs']['target'].double()
        with self.assertRaises(ValueError):
            Adapter().prepare('after', case, value, Path('.'))

    def test_alias_and_hidden_storage_rejected(self):
        case = spec()
        value = make_case(case)
        value['inputs']['q'] = value['inputs']['target']
        with self.assertRaises(ValueError):
            validate_inputs(value['inputs'], case)
        value = make_case(case)
        value['inputs']['target'] = torch.zeros(4, 37)[1:]
        with self.assertRaises(ValueError):
            validate_inputs(value['inputs'], case)

    def test_cpu_cache_reused_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, first = fixture(Adapter(), spec(), Path(directory), torch)
            b, second = fixture(Adapter(), spec(), Path(directory), torch)
            self.assertEqual(first['oracle_calls'], 1)
            self.assertEqual(second['oracle_calls'], 0)
            self.assertTrue(torch.equal(a['expected'], b['expected']))
            Path(first['cache'], 'fixture.pt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                fixture(Adapter(), spec(), Path(directory), torch)

    def test_prepare_reuse_compilation_and_output_ownership(self):
        artifact = SimpleNamespace(hash='a' * 64, asm={'npubin': b'binary'})
        launches = []
        class Kernel:
            fn = SimpleNamespace(__name__='fake_sampler')
            def warmup(self, *args, **kwargs):
                return artifact
            def __getitem__(self, grid):
                def launch(*args, **flags):
                    launches.append((grid, flags, args[0].numel()))
                    args[0].zero_()
                    return artifact
                return launch
        original_full = torch.full
        def cpu_full(*args, **kwargs):
            kwargs.pop('device', None)
            return original_full(*args, **kwargs)
        modules = dict(torch_npu=SimpleNamespace(),
                       triton=SimpleNamespace(next_power_of_2=lambda n: 1 << (n-1).bit_length(), __version__='fake'),
                       sampler_kernel=SimpleNamespace(sample_recovered_tokens_kernel=Kernel()))
        adapter = Adapter()
        with patch.dict(sys.modules, modules), patch.object(torch.Tensor, 'npu', lambda self: self.clone(), create=True), patch.object(torch, 'full', cpu_full):
            one = adapter.prepare('before', spec(), make_case(spec()), Path('.'))
            one.compile_only()
            self.assertEqual(launches, [])
            one.run()
            other_spec = dict(seed=99, lengths=[1], vocab=9, draft_probs_present=False)
            two = adapter.prepare('before', other_spec, make_case(other_spec), Path('.'))
            two.run()
            self.assertEqual([x[2] for x in launches], [3, 1])
            self.assertIs(adapter.prepared['before'], two)
            self.assertNotEqual(one.output_storage.data_ptr(), two.output_storage.data_ptr())
            self.assertEqual(one.output_storage[[0,-1]].tolist(), [-701, -701])
            self.assertEqual(two.compilation_state()[0]['binaries']['npubin'], __import__('hashlib').sha256(b'binary').hexdigest())

    def test_public_cli_external_cwd_flat_nested_and_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            case_file = directory / 'case.json'
            command = [sys.executable, str(DIRECTORY / 'sampler_cli.py'), '--case', str(case_file),
                       '--output', str(directory / 'out'), '--cache', str(directory / 'cache'),
                       '--provenance', str(directory / 'provenance.json'), '--validate-only']
            for case in (spec(), dict(seed=19, **spec()['parameters'])):
                case_file.write_text(json.dumps(case))
                result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['status'], 'supported')
            invalid = spec()
            invalid['parameters']['vocab'] = 0
            case_file.write_text(json.dumps(invalid))
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((directory / 'cache').exists())
            self.assertFalse((directory / 'out').exists())

    def test_standalone_import_and_invalid_arm(self):
        module_spec = importlib.util.spec_from_file_location('standalone_sampler', DIRECTORY / 'sampler.py')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        with self.assertRaises(ValueError):
            module.factory().prepare('bad', spec(), make_case(spec()), Path('.'))


if __name__ == '__main__':
    unittest.main()
