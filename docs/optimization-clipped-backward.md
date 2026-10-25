# Clipped SwiGLU backward: failed resource qualification

This draft retains a source-backed experiment that failed before scientific
correctness or profiling. The first baseline linked successfully. The first
candidate, at source tile BLOCK_M=4/BLOCK_K=1024, required **214,016 bytes of UB**
while the compiler reported **196,608 bytes available**. Its automatic code-motion
fallback also failed. This is a resource failure of the recorded fixed
configuration, not evidence that the technique is generally impossible or that
infrastructure caused the failure. No retile, flag change or retry was performed.

| Frozen cases | Compile outcome | Scientific correctness / profiling |
|---|---|---|
| W008-T04-01 baseline | Linked binary retained | Unattempted |
| W008-T04-01 candidate | UB overflow | Unattempted |
| Remaining five cases, both arms | Unattempted after fatal stop | Unattempted |

There were zero scientific CPU oracle payloads, source backward forwards,
correctness calls, profiles or captures. NPU initialization and input-only probe
setup occurred before the compiler failure. No speedup or geomean is available.
The complete inventory audit is expected to fail: no numerical or complete
resource qualification is claimed. The [portable result](../experiments/optimization-wave008/T04-result.json)
binds the source, baseline binary/resources, failed candidate IR/compiler error,
runtime, exact counts and retained handles. Raw diagnostics are retained in a
compressed artifact. All compact file hashes were verified.

The original v1 observer was cancelled for an unknown reason after only queued
observations; the original source/output directories were absent in a read-only
audit. No v1 execution was observed. The unique v2 attempt retained the same JIT
kernels, provider, matrix, math, source block picker and common flags, with a
prospective host-only prototype rejection fix. A startup watchdog covered NPU
initialization/transfers/compilation; failure stopped before expected-data
creation. There was no scientific replay. Read-only cleanup found no live
original PID/source/compiler-cache processes, free device/profiler locks, healthy
idle device status, no device processes and zero ECC errors; no reset was used.
The original process-group ID was not serialized and is explicitly unavailable.

Default numeric workflows are disabled: factory, `Adapter.make_case`, prepare,
validate, capture and CLI reject before expected-data generation, compilation
or input transfers. Direct CPU provider functions remain available for
NONFROZEN contract tests. Explicit compile inspection allocates uninitialized
prototypes only and never authorizes a numeric forward:

```python
from clipped_backward import Adapter
probe = Adapter(compile_inspection=True).compile_probe(
    'after', {'M': 2, 'K': 27}, None)
artifacts = probe.compile_only()  # Requires a selected compatible NPU runtime.
# probe.run() always raises StudyUnsupported, even after successful inspection.
```

Inspection uses atomic cached qualification and explicit FP32 storage independent
of the process default dtype. Five functional CPU tests cover default rejection
before provider/cache/backend work, external-cwd CLI rejection, prototype no-launch,
compile-failure retry, changed defaults, and the independent CPU numerical
contract. The separate CPU-only A3 publication gate performs no real compilation,
NPU execution, frozen-fixture generation or profiling. Publication host hashes
are separate from the failed-attempt host wrapper and unchanged JIT/provider.

The retained candidate derives from MindSpeed revision
`4efe2b675c2d51517c3fdd14c721020b8d41e241`, under the full retained Apache 2.0
license and Huawei copyright. It tests whether every active gate in the complete
source tile exceeds 7, treating inactive lanes as true. Saturated tiles write
positive-zero gate gradients and the inclusive-clamp up gradient using the same
FP32 constant-sigmoid path and multiplication order; other tiles retain the source
graph. Masks and stores are unchanged. The bounded intended inputs are contiguous
FP32 GUP[M,2K], DY[M,K], 1<=M,K<=1024, alpha1.702, limit7, GUP in[-12,12] and
DY in[-1,1]. These arithmetic and dispatch properties have no passing device
numerical gate in this study.

The independent frozen oracle would use FP64 with fixed atol2e-5/rtol2e-5,
retaining clamp derivatives at exact boundaries and requiring positive zero
strictly outside. It was not generated for any scientific case. No dynamic
internal branch execution or candidate selector was observed, and no CPU fixture
predicate counts were computed. Source hardware comments are provenance rather
than proof of current resource fit. No allocator, ordinary latency, physical
traffic, roofline or occupancy claim is made.
