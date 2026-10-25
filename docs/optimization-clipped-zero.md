# Zero-limit clipped forward compile screen

This bounded prepared FP32 component derives from MindSpeed revision
`4efe2b675c2d51517c3fdd14c721020b8d41e241`, retaining Apache2 and Huawei
notices. The baseline is the source forward kernel and source2D block
picker. The candidate changes only constexpr LIMIT==0: it omits the UP
load/clamp and multiplication by1, retaining gate minimum, sigmoid operand
precision, multiplication order, cast, and signed zeros. LIMIT7 executes
the exact source fallback graph. No flat-index or dynamic saturation
transformation is combined here.

GUP[M,2K] must be contiguous FP32, 1<=M,K<=1024, with finite1/64-grid values
in[-12,12], including the entire UP half. The CPU provider validates values
before oracle creation/cache publication; direct preparation checks caller
inputs before backend/cache qualification. Invalid odd shapes, dtypes,
strides, NaN and infinity are rejected. Outputs use private preallocated
storage with prefix/suffix canaries and are reused by ordinary forwards.
The allocation boundary excludes allocator/full-model latency claims.

Input-only probes cannot execute: their `run()` raises before compilation
or launch. `compile_only()`, metadata and binary identity remain inspectable.
Compiled state is committed only after qualification succeeds; a failed
qualification leaves no cached success. Ordinary prepared items retain
normal execution with cached first-launch compilation.

The frozen collector compiles all six cases/arms before generating any
scientific expected payload or numerical forward. It records actual ELF
identity, executable text, all allocated constants/data/BSS, semantic
relocation targets/addends, ELF and compiled signature/metadata ABI, loader
notes, compiler IR and resources. Identical text alone does not prove a
no-op. Unknown ABI/relocations or differing constant/loader state yield an
inconclusive screen. Different entry names or whole ELF files alone never
manufacture a contrast; optimizer flags remain common.

If all four main cases are compiler no-ops or inconclusive, collection stops
with zero scientific oracles, numerical forwards and captures. Guard
compilation probes remain recorded, while mathematical guards are explicitly
unattempted. Such a result validates a compiler screen and CPU interfaces;
it does not certify kernel numerical correctness or performance.

Only a qualified code contrast proceeds to six independent cached CPU FP64
source-formula expectations, fixed atol3e-5/rtol2e-5, exact paired FP32 output
bits and signed-zero checks. All twelve numerical calls precede exactly ten
profile calls per arm on four main cases, guards0 captures. No warmups,
autotuning, extra timing, profile IO inspection, or data-driven retuning.

Public exploratory CLI from any working directory:

```sh
python /absolute/repo/examples/optimization-library/triton/clipped_zero.py \
  --case case.json --output results --cache oracle-cache \
  --provenance provenance.json --correctness-only
```

Flat `{"M":2,"K":19,"limit":0,"seed":1211}` and nested parameters with
role metadata are supported. The CLI rejects invalid specifications before
cache lookup and labels its qualification exploratory; it does not implement
the frozen all-case screening order. Scope is forward only. Source hardware
comments are provenance, not installed-resource or throughput proof. Profile
diagnostics, if collected, establish no ordinary latency, physical traffic,
roofline, occupancy, model-utility or backward-behavior claim.

On the selected native BZ-A3-2 runtime (observed9382/Ascend910V1,
24 Cube and48 vector cores, imported Triton3.2.0/CANN9.1), all four main
cases have distinct selected executable text with equal allocated constant
state, decoded semantic relocations, and calling/resource ABI. For the
larger tile, text is3128→1800 bytes; the small-M tile is2732→1600 bytes.
These byte counts are not instruction counts. The LIMIT7 guard is an
effective compiler no-op and is recorded as the actual source fallback.

All six numerical pairs pass the fixed tolerance, exact paired output bits
and signed-zero checks. Six CPU expectations were generated once;12
correctness calls precede80 profiled API calls/8 captures. All333 compact
artifact hashes and measured source bindings verify. The measured means are:

| M × K | Operation span before/after (µs) | Profiled API event before/after (ms) |
|---|---|---|
| 128 × 1024 | 4.0441 /3.2240 | 0.1665 /0.2023 |
| 1 ×17 | 2.0581 /1.6981 | 0.2001 /0.2033 |
| 1024 ×1024 | 11.8042 /7.5501 | 0.1948 /0.1945 |
| 33 ×1000 | 3.9102 /2.9801 | 0.2082 /0.2432 |

Operation-span means decrease in all four cases; complete-API event means
are mixed. All ten original samples and variation are retained, including
first calls. These are instrumented diagnostics, not ordinary API latency.
Raw supported PMU rows, full binary identities, resource exports and the
unmodified comparison proof are retained. Public proof JSON replaces runtime
path prefixes with labels; its raw counterpart is hash-bound in the compact
manifest. The complete six-case coverage audit passes, with LIMIT7 explicitly
on the source fallback rather than claiming zero-limit specialization there.
