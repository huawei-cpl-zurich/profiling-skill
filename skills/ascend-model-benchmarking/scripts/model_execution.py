"""Compiled Model preparation and unprofiled, synchronized warm latency.

Importable without accelerator dependencies. Models and device inputs are supplied
by a private adapter; input creation, correctness and compilation are never timed.
Use a separate profiler replay for device busy-time, not these wall-clock samples.
"""

from contextlib import contextmanager
import importlib
import math
import statistics
import sys
import time


class CapabilityError(RuntimeError):
    """The required compiled NPU execution mode could not be established."""


class ModelExecutionError(RuntimeError):
    """Model construction, graph compilation or execution failed, not transport."""

    def __init__(self, message, stage):
        super().__init__(message)
        self.stage = stage


class MeasurementError(RuntimeError):
    """An active measurement was invalidated; retain diagnostics, not a timing."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


class GraphCompiler:
    """Explicit NPU graph backend. Injection is for CPU functional tests only.

    The counting backend wraps the real compiled graph callable, not Model.forward.
    Fullgraph makes graph breaks errors; neither compilation nor execution retries
    through an eager callable. Graph text is evidence, not custom-kernel identity
    proof: the acquisition adapter must also verify executed kernel identities.
    """

    def __init__(self, *, backend_name="torchair-ge", torch_module=None, backend=None):
        if backend_name not in ("torchair-ge", "npu-inductor"):
            raise ValueError("supported backends: torchair-ge, npu-inductor; no automatic fallback")
        self.backend_name = backend_name
        try:
            self.torch = torch_module or importlib.import_module("torch")
            if backend is None:
                # Torch-NPU registers bundled TorchAir on some supported images.
                importlib.import_module("torch_npu")
                if backend_name == "torchair-ge":
                    self.backend = importlib.import_module("torchair").get_npu_backend()
                else:
                    importlib.import_module("torch_npu._inductor")
                    registry = importlib.import_module("torch._dynamo.backends.registry")
                    self.backend = registry.lookup_backend("inductor")
            else:
                self.backend = backend
        except (ImportError, AttributeError) as exc:
            label = "TorchAir GE" if backend_name == "torchair-ge" else "NPU Inductor"
            raise CapabilityError(f"{label} backend unavailable: {exc}") from exc
        self.compilations = 0
        self.triton_compilations = 0
        self.executions = 0
        self.graphs = []
        self.last_failure = None

    def backend_wrapper(self, graph, example_inputs):
        self.compilations += 1
        self.graphs.append(graph.code)
        try:
            executable = self.backend(graph, example_inputs)
        except Exception:
            self.last_failure = "backend-compilation"
            raise
        if not callable(executable):
            raise CapabilityError("NPU backend did not return a callable graph")

        def execute(*args, **kwargs):
            self.executions += 1
            return executable(*args, **kwargs)
        # Do not copy unwrap/disable metadata: Dynamo may otherwise bypass this
        # execution counter. Preserve only the calling-convention marker.
        if hasattr(executable, "_boxed_call"):
            execute._boxed_call = executable._boxed_call
        return execute

    def snapshot(self):
        return self.compilations, self.triton_compilations

    @contextmanager
    def monitor_compilation(self, *, on_compile=None):
        """Catch Triton compile entrypoints, including cached Python aliases.

        Install before compilation so modules lazily imported by Triton runtime are
        included. Intended for an isolated single-threaded benchmark process.
        """
        for name in ("triton.compiler.compiler", "triton.runtime.jit"):
            try:
                importlib.import_module(name)
            except ModuleNotFoundError as exc:
                if exc.name and exc.name.split(".")[0] == "triton":
                    continue
                raise CapabilityError(f"broken Triton installation: {exc}") from exc
        entries = []
        for name in ("triton", "triton.compiler.compiler", "triton.compiler", "triton.runtime.jit"):
            module = sys.modules.get(name)
            original = getattr(module, "compile", None)
            if not callable(original):
                continue

            def wrapper(*args, _original=original, **kwargs):
                if on_compile is None:
                    self.triton_compilations += 1
                else:
                    on_compile()
                return _original(*args, **kwargs)
            entries.append((module, original))
            module.compile = wrapper
        try:
            yield
        finally:
            for module, original in reversed(entries):
                module.compile = original


# Compatibility for the initial GE-default adapter; new callers select GraphCompiler.
GECompiler = GraphCompiler


class EagerExecution(GraphCompiler):
    """Explicit eager reference; monitor compilation without loading a backend."""
    def __init__(self, *, torch_module=None):
        self.torch = torch_module or importlib.import_module("torch")
        self.backend_name = None
        self.compilations = self.triton_compilations = self.executions = 0
        self.graphs = []


def prepare_eager_model(factory, args, kwargs, validate, *, execution=None,
                        synchronize=lambda: None):
    execution = execution or EagerExecution()
    start = time.perf_counter_ns()
    stage = "model-construction"
    try:
        with execution.torch.inference_mode(), execution.monitor_compilation():
            model = factory()
            if hasattr(model, "eval"):
                model.eval()
            stage = "eager-forward"
            output = model(*args, **kwargs)
            synchronize()
    except TimeoutError:
        raise
    except Exception as exc:
        raise ModelExecutionError(f"explicit eager reference failed: {exc}", stage) from exc
    validate(output)
    return PreparedModel(model, tuple(args), dict(kwargs), execution, {
        "execution_mode": "eager", "backend": None, "fullgraph": False,
        "autograd_mode": "inference", "model_mode": "eval-if-supported",
        "torch_version": execution.torch.__version__, "graphs": [],
        "compile_count": 0, "compiled_executions": 0,
        "triton_compile_count": execution.triton_compilations,
        "cold_prepare_ms": (time.perf_counter_ns() - start) / 1e6,
        "cold_prepare_includes": "factory + first execution + synchronization",
        "custom_kernel_identity_verified": False,
    })


class PreparedModel:
    def __init__(self, compiled, args, kwargs, compiler, evidence):
        self.compiled = compiled
        self.args = args
        self.kwargs = kwargs
        self.compiler = compiler
        self.evidence = evidence

    def __call__(self):
        return self.compiled(*self.args, **self.kwargs)


def prepare_model(factory, args, kwargs, validate, *, compiler=None,
                  synchronize=lambda: None):
    """Construct and compile a Model, then validate the actual compiled output.

    ``validate(output)`` raises on mismatch against the untouched reference;
    adapters own domain-specific cases/tolerances. Supply NPU synchronize to keep
    cold preparation diagnostics separate from later active measurements.
    """
    compiler = compiler or GraphCompiler()
    before = compiler.compilations, compiler.executions
    start = time.perf_counter_ns()
    stage = "model-construction"
    try:
        with compiler.torch.inference_mode(), compiler.monitor_compilation():
            model = factory()
            if hasattr(model, "eval"):
                model.eval()
            stage = "graph-compilation"
            compiled = compiler.torch.compile(
                model, backend=compiler.backend_wrapper, fullgraph=True, dynamic=False)
            stage = "compiled-forward"
            output = compiled(*args, **kwargs)
            synchronize()
    except TimeoutError:
        raise
    except Exception as exc:
        raise ModelExecutionError(f"required fullgraph {compiler.backend_name} execution failed: {exc}",
                                  compiler.last_failure or stage) from exc
    cold_ms = (time.perf_counter_ns() - start) / 1e6
    if compiler.compilations <= before[0] or compiler.executions <= before[1]:
        raise CapabilityError("no compiled graph capture/execution evidence "
            f"(captures={compiler.compilations - before[0]}, "
            f"executions={compiler.executions - before[1]}); eager is forbidden")
    validate(output)
    evidence = {
        "execution_mode": "compiled", "backend": compiler.backend_name, "fullgraph": True, "dynamic": False,
        "autograd_mode": "inference", "model_mode": "eval-if-supported",
        "torch_version": compiler.torch.__version__, "graphs": list(compiler.graphs),
        "compile_count": compiler.compilations, "compiled_executions": compiler.executions,
        "triton_compile_count": compiler.triton_compilations,
        "cold_prepare_ms": cold_ms,
        "cold_prepare_includes": "factory + compile + first execution + synchronization",
        "custom_kernel_identity_verified": False,
    }
    return PreparedModel(compiled, tuple(args), dict(kwargs), compiler, evidence)


def percentile(values, fraction):
    ordered = sorted(values)
    offset = (len(ordered) - 1) * fraction
    left = math.floor(offset)
    right = math.ceil(offset)
    return ordered[left] + (ordered[right] - ordered[left]) * (offset - left)


def summarize(batches):
    """Preserve all samples; no threshold, task-row or slow-sample filtering."""
    raw = [list(batch) for batch in batches]
    values = [value for batch in raw for value in batch]
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise MeasurementError("empty, negative or nonfinite latency samples")
    medians = [statistics.median(batch) for batch in raw]
    median = statistics.median(values)
    return {
        "batches_ms": raw, "median_ms": median,
        "p10_ms": percentile(values, .1), "p90_ms": percentile(values, .9),
        "batch_medians_ms": medians,
        "between_batch_relative_spread": (max(medians) - min(medians)) / median
        if median else 0.0,
    }


def benchmark_pair(baseline, candidate, *, synchronize, batches=3, samples=50,
                   warmup=20, clock=time.perf_counter_ns, validate_inputs=None):
    """Measure full forward completion with profiler disabled and warm inputs.

    Invoke once per already-prepared specialization. Device/clock/load metadata,
    source identities and kernel identity verification belong in the acquisition
    adapter. A raised MeasurementError means no valid timing result is produced.
    Optional validate_inputs(phase) runs after each arm's warmup and measured batch,
    outside clock windows and before another arm can consume shared inputs.
    """
    if batches < 1 or samples < 1 or warmup < 20:
        raise ValueError("require positive batches/samples and at least 20 warmups")
    models = {"baseline": baseline, "candidate": candidate}
    results = {name: [] for name in models}
    orders = []
    empty = []
    active_triton_compilations = 0

    def record_active_compile():
        nonlocal active_triton_compilations
        active_triton_compilations += 1

    def timed(call):
        synchronize()
        started = clock()
        call()
        synchronize()
        elapsed = (clock() - started) / 1e6
        return elapsed

    with baseline.compiler.torch.inference_mode():
        for name, model in models.items():
            # Warmup compilation belongs to the model being warmed, not whichever
            # implementation supplied the measurement-level monitor.
            with model.compiler.monitor_compilation():
                for _ in range(warmup):
                    model()
                synchronize()
            model.evidence["post_warmup_compile_count"] = model.compiler.compilations
            model.evidence["post_warmup_triton_compile_count"] = model.compiler.triton_compilations
            if validate_inputs is not None:
                validate_inputs(f"{name}:warmup")
        expected = {name: model.compiler.snapshot() for name, model in models.items()}
        for _ in range(samples):
            empty.append(timed(lambda: None))
        for batch in range(batches):
            order = ["baseline", "candidate"] if batch % 2 == 0 else ["candidate", "baseline"]
            orders.append(order)
            for name in order:
                model = models[name]
                executions = model.compiler.executions
                values = []
                with baseline.compiler.monitor_compilation(on_compile=record_active_compile):
                    for _ in range(samples):
                        values.append(timed(model))
                        if active_triton_compilations or any(
                                item.compiler.snapshot() != expected[key]
                                for key, item in models.items()):
                            raise MeasurementError("compilation detected during active measurement", {
                                "completed_batches_ms": results, "invalid_arm": name,
                                "invalid_batch_ms": values, "empty_sync_ms": empty,
                                "expected_compile_counts": expected,
                                "measurement_triton_compile_count": active_triton_compilations,
                                "actual_compile_counts": {
                                    key: item.compiler.snapshot() for key, item in models.items()},
                            })
                if (model.evidence.get("execution_mode", "compiled") == "compiled"
                        and model.compiler.executions - executions < samples):
                    raise MeasurementError("compiled graph was not executed for every sample")
                results[name].append(values)
                synchronize()
                if validate_inputs is not None:
                    validate_inputs(f"{name}:batch-{batch}")
    report = {name: summarize(values) for name, values in results.items()}
    report.update({
        "metric": "unprofiled synchronized Model forward wall latency",
        "unit": "ms", "warmup_calls": warmup, "batch_order": orders,
        "empty_sync_ms": empty, "empty_sync_median_ms": statistics.median(empty),
        "sync_overhead_subtracted": False, "compilation_in_active_window": False,
        "execution_evidence": {name: model.evidence for name, model in models.items()},
    })
    if report["candidate"]["median_ms"] <= 0:
        raise MeasurementError("zero-resolution candidate timings; use a higher resolution clock")
    report["latency_speedup"] = report["baseline"]["median_ms"] / report["candidate"]["median_ms"]
    return report
