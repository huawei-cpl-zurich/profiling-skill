import importlib.util
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "model_execution", Path(__file__).parents[1]
    / "skills/ascend-model-benchmarking/scripts/model_execution.py")
execution = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(execution)


class FakeTorch:
    __version__ = "test"

    inference_mode = staticmethod(nullcontext)

    def compile(self, model, *, backend, fullgraph, dynamic):
        assert fullgraph and not dynamic
        compiled = {}

        def invoke(*args, **kwargs):
            shape = tuple(type(arg) for arg in args)
            if shape not in compiled:
                graph = SimpleNamespace(code="def forward(x): return x + 1")
                graph.forward = model
                compiled[shape] = backend(graph, args)
            return compiled[shape](*args, **kwargs)
        return invoke


class Clock:
    def __init__(self):
        self.value = 0

    def __call__(self):
        self.value += 1000
        return self.value


class ExecutionTests(unittest.TestCase):
    def prepare(self, value=3, model=None):
        compiler = execution.GECompiler(
            torch_module=FakeTorch(), backend=lambda graph, args: graph.forward)
        return execution.prepare_model(
            lambda: model or (lambda x: x + 1), (value,), {},
            lambda result: self.assertEqual(result, value + 1), compiler=compiler)

    def test_compiles_and_validates_before_measurement(self):
        prepared = self.prepare()
        self.assertEqual(prepared.evidence["backend"], "torchair-ge")
        self.assertEqual(len(prepared.evidence["graphs"]), 1)
        self.assertEqual(prepared.evidence["compile_count"], 1)
        self.assertGreater(prepared.evidence["compiled_executions"], 0)

    def test_all_samples_alternate_order_and_report_sync_overhead(self):
        calls = []
        baseline = self.prepare(model=lambda x: calls.append("baseline") or x + 1)
        candidate = self.prepare(model=lambda x: calls.append("candidate") or x + 1)
        calls.clear()
        report = execution.benchmark_pair(
            baseline, candidate, synchronize=lambda: None, clock=Clock())
        self.assertEqual(report["batch_order"], [
            ["baseline", "candidate"], ["candidate", "baseline"],
            ["baseline", "candidate"]])
        for arm in ("baseline", "candidate"):
            result = report[arm]
            self.assertEqual([len(batch) for batch in result["batches_ms"]], [50] * 3)
            self.assertAlmostEqual(result["median_ms"], .001)
            self.assertEqual(result["between_batch_relative_spread"], 0)
        self.assertEqual(len(report["empty_sync_ms"]), 50)
        self.assertEqual(len(calls), 2 * (20 + 150))
        self.assertEqual(report["baseline"]["median_ms"], report["empty_sync_median_ms"])

    def test_correctness_failure_never_reaches_benchmark(self):
        with self.assertRaises(AssertionError):
            self.prepare(model=lambda x: 100)

    def test_lazy_backend_not_executed_is_rejected(self):
        class EagerFallback(FakeTorch):
            def compile(self, model, **kwargs):
                return model
        with self.assertRaisesRegex(execution.CapabilityError, "compiled graph"):
            execution.prepare_model(
                lambda: lambda x: x, (1,), {}, lambda value: None,
                compiler=execution.GECompiler(torch_module=EagerFallback(),
                    backend=lambda graph, args: graph.forward))

    def test_recompilation_during_measurement_invalidates_batch(self):
        prepared = self.prepare()
        original = prepared.compiled
        count = 0

        def recompiling(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 21:
                prepared.compiler.compilations += 1
            return original(*args, **kwargs)
        prepared.compiled = recompiling
        with self.assertRaisesRegex(execution.MeasurementError, "compilation") as caught:
            execution.benchmark_pair(prepared, self.prepare(), synchronize=lambda: None,
                                     clock=Clock())
        self.assertEqual(len(caught.exception.details["invalid_batch_ms"]), 1)

    def test_backend_error_not_silently_retried_eager(self):
        def broken(graph, args):
            raise RuntimeError("graph break")
        with self.assertRaisesRegex(execution.ModelExecutionError, "graph break") as caught:
            execution.prepare_model(
                lambda: lambda x: x, (1,), {}, lambda value: None,
                compiler=execution.GECompiler(torch_module=FakeTorch(), backend=broken))
        self.assertEqual(caught.exception.stage, "backend-compilation")

    def test_factory_eval_and_inference_are_used_in_all_phases(self):
        phases = []
        active = False
        class Mode:
            def __enter__(self):
                nonlocal active
                active = True
            def __exit__(self, *args):
                nonlocal active
                active = False
        class Model:
            def eval(self):
                phases.append("eval")
                return self
            def __call__(self, x):
                self.assert_active()
                return x + 1
            def assert_active(self):
                if not active:
                    raise AssertionError("autograd active")
        torch = FakeTorch()
        torch.inference_mode = Mode
        model = execution.prepare_model(Model, (1,), {}, lambda output: None,
            compiler=execution.GECompiler(torch_module=torch, backend=lambda graph, args: graph.forward))
        execution.benchmark_pair(model, model, synchronize=lambda: None, clock=Clock())
        self.assertEqual(phases, ["eval"])

    def test_invalid_counts_rejected(self):
        prepared = self.prepare()
        with self.assertRaises(ValueError):
            execution.benchmark_pair(prepared, prepared, synchronize=lambda: None, samples=0)

    def test_percentiles_and_outliers_are_not_filtered(self):
        summary = execution.summarize([[1, 2, 3, 4, 100], [2, 2, 2, 2, 2]])
        self.assertEqual(summary["batches_ms"][0][-1], 100)
        self.assertEqual(summary["median_ms"], 2)
        self.assertGreater(summary["p90_ms"], 4)

    def test_triton_compilation_in_active_phase_is_rejected(self):
        prepared = self.prepare()
        original = prepared.compiled
        count = 0
        def compile_late(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 21:
                prepared.compiler.triton_compilations += 1
            return original(*args, **kwargs)
        prepared.compiled = compile_late
        with self.assertRaises(execution.MeasurementError):
            execution.benchmark_pair(prepared, self.prepare(), synchronize=lambda: None)

    def test_real_compile_hook_restores_entrypoint_after_failure(self):
        compiler = self.prepare().compiler
        original = lambda: "binary"
        module = SimpleNamespace(compile=original)
        modules = {name: module for name in (
            "triton.compiler.compiler", "triton.compiler", "triton.runtime.jit")}
        with patch.dict(execution.sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "stop"):
                with compiler.monitor_compilation():
                    self.assertEqual(module.compile(), "binary")
                    self.assertGreater(compiler.triton_compilations, 0)
                    raise RuntimeError("stop")
            self.assertIs(module.compile, original)

    def test_missing_ge_is_explicit_capability_failure(self):
        with patch.object(execution.importlib, "import_module", side_effect=ModuleNotFoundError("torchair")):
            with self.assertRaisesRegex(execution.CapabilityError, "TorchAir GE backend unavailable"):
                execution.GECompiler(torch_module=FakeTorch())

    def test_torch_npu_initializes_before_bundled_torchair(self):
        imports = []
        def load(name):
            imports.append(name)
            return SimpleNamespace(get_npu_backend=lambda: "ge")
        with patch.object(execution.importlib, "import_module", side_effect=load):
            compiler = execution.GECompiler(torch_module=FakeTorch())
        self.assertEqual(imports, ["torch_npu", "torchair"])
        self.assertEqual(compiler.backend, "ge")

    def test_existing_compiler_evidence_cannot_cover_a_later_eager_model(self):
        compiler = self.prepare().compiler
        compiler.torch.compile = lambda model, **kwargs: model
        with self.assertRaisesRegex(execution.CapabilityError, "compiled graph"):
            execution.prepare_model(lambda: lambda x: x, (1,), {}, lambda output: None,
                                    compiler=compiler)

    def test_backend_unwrap_metadata_cannot_bypass_execution_counter(self):
        def backend(graph, args):
            def executable(*args):
                return graph.forward(*args)
            executable._torchdynamo_orig_callable = graph.forward
            executable._boxed_call = True
            return executable
        compiler = execution.GECompiler(torch_module=FakeTorch(), backend=backend)
        wrapped = compiler.backend_wrapper(SimpleNamespace(code="graph", forward=lambda x: x), (1,))
        self.assertFalse(hasattr(wrapped, "_torchdynamo_orig_callable"))
        self.assertTrue(wrapped._boxed_call)
        self.assertEqual(wrapped(1), 1)
        self.assertEqual(compiler.executions, 1)


if __name__ == "__main__":
    unittest.main()
