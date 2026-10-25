# Packed AWQ expansion and group metadata reuse

This prepared dequantization component compares educational unpacking per output
lane with the pinned source's packed-word loads, three interleaves and broadcast
group metadata. Both use a 16×16 packed tile (16×128 output elements), the same
grid and compiler flags. The educational arm is not a vendor baseline; the
source configuration is not an autotuner result. Output allocation is excluded
and private output storage is reused in both arms.

The source is `src/triton_ascend_kernels/gemm/awq_triton.py` at
`7acb9c40a6ca827e885d902b7eaee5d45457b230`. Its dequantization kernel is copied
with attribution and [license notices](../examples/optimization-library/triton/awq_expansion_NOTICE).
The application makes no new unique mechanism claim and does not include GEMM.

Inputs are contiguous owned CPU int32 packed weights `[K,N/8]`, int32 packed
zeros `[K/group,N/8]`, and FP16 scales `[K/group,N]`. `N` must be a multiple of
eight; groups are 32, 64, 128 or the full K, with no tile crossing valid groups.
Nibble order is `[0,4,1,5,2,6,3,7]`. This source zero format has no `+1` offset
and is not a generic checkpoint interoperability API.

Scales are exact `q/256`, integer q from 4 through 64. An independent CPU
uint32 decoder forms integer differences and products before the final cast.
Products have magnitude at most 960, so outputs must match every FP16 bit.
Signed packed words, all nibble values, packed-column and row tails are covered
by the [frozen matrix](../experiments/optimization-wave009/R01.json).
The public adapter validates input shape, dtype, layout and scale domain before
moving inputs to the device. Its input-only probes refuse execution. A normal
prepared launch must successfully compile before launching; failed compilation
is retried on the next call.

The standalone CLI is exploratory; it does not replace the controlled collector's
all-case codegen and resource screens, correctness gates or exclusive profiling:

```bash
python examples/optimization-library/triton/awq_expansion_cli.py \
  --case case.json --cache oracle-cache --output result \
  --provenance provenance.json
```

The collector compiles both arms for every case before generating the six
scientific payloads. It retains compiler no-ops or inconclusive code identity
without manufacturing a contrast. If code differs, every arm must first pass
one correctness gate. Only then are the four main pairs profiled with ten full
API invocations per arm, without warmup; guards receive no captures. Supported
raw PMUs and all samples are retained. Profiled operation spans are diagnostic,
not ordinary application latency, instruction-count proof or physical traffic.

Collection on native A3 Ascend910_9382 completed: twelve input-only arm probes
produced distinct `.text` code for all six pairs before oracle generation.
All twelve numeric gates matched the CPU oracle exactly, and all six paired
output bit checks passed. Five host functional tests passed both locally and
on the A3 runtime. The coverage/layout audit passed against the frozen inventory.

Mean profiled device operation spans (microseconds, ten samples per side):

| Case | Output lanes before | Packed expansion after |
|---|---:|---:|
| 01 | 129.685 | 64.629 |
| 02 | 130.481 | 66.039 |
| 03 | 384.938 | 190.560 |
| 04 | 257.933 | 129.167 |

The [retained evidence](data/optimization-w009-awq-expansion.json) includes every
sample, raw supported PMU fields, compiler/resource/code hashes, six fixture
bindings and the 261-file verified compact manifest. It records the imported
Triton version separately from installed distribution versions. These paired
spans apply to this prepared, instrumented component and common configuration.
No additional measurement or optimizer adjustment followed collection.
