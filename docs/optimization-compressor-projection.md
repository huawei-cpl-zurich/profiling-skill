# Controlled Compressor dual projections

`examples/optimization-library/triton/compressor_projection.py` compares the
projection component from Triton-Ascend-Kernels revision
`7acb9c40a6ca827e885d902b7eaee5d45457b230`. Upstream MIT and third-party notices
are retained beside the source. It computes two independent BF16 projections
into separate, prepared FP32 outputs. It excludes pooling, cache maintenance,
output allocation, and the complete Compressor operator.

Both arms use the predeclared common BM32/BN64/BK128/GROUP8 configuration.
This is a bounded controlled comparison, not the vendor's public small-M
BK1024 selection or large-M autotuner throughput. The before arm launches the
source single projection twice; the after arm launches its two-task-space
kernel once. Each projection loads X separately. The after arm preserves the
source's fixed W0/W1 pointer branches. Its persistent grid covers both tile
spaces; the before grids cover each separately. Cube count comes from the
current logical device's `num_aicore` property, not its vector-core count.

The supported contract is contiguous BF16 X[M,5120], W0/W1[512,5120], M in
[0,256], distinct allocations, and values q/128 for integer q in [-16,16].
The CPU fixture provider computes FP64 integer-valued dot products. The
conservative error bound gamma(5120, FP64) * 1310720 is below 0.5, recovering
the exact integer dot; every possible integer prefix is below 2^24. Dividing
by 16384 gives exactly representable FP32 results. Both output bit patterns
must match independently; tolerance is zero. Output canaries, shape, dtype,
input immutability, and storage nonaliasing are checked after correctness.
Empty M produces two empty outputs without kernel launches.

The ordinary CLI can run from any working directory:

```sh
python /absolute/repo/examples/optimization-library/triton/compressor_projection.py \
  --case case.json --output results --cache oracle-cache \
  --provenance provenance.json --correctness-only
```

A case can use `{"M":3,"seed":112}` or
`{"parameters":{"M":3},"seed":112,"role":"guard"}`. Output and cache folders
are explicit. Empty M always runs correctness only. Unsupported dimensions
and tile settings fail before cache lookup in this CLI. It labels itself
exploratory: the scientific collector additionally compiles/screens all frozen
cases before math and gates all six pairs before profiles. A first public
launch also qualifies and caches compilation before math.

The frozen wave008/T01 matrix fixes four profile cases and two correctness
only guards. Profiling uses ten complete prepared API calls per arm without
warmups, autotuning, extra timing, or inspecting outputs inside capture.
Profiler event and operation-span durations are instrumented diagnostics;
they do not establish ordinary latency, physical traffic, roofline limits,
or occupancy. Reused prepared outputs and JIT scratch are outside allocator
claims. No unique-mechanism or complete-operator speedup is asserted.

The A3 receipt in `experiments/optimization-wave008/T01-results.json` passes
all six paired exact-bit checks. The independent coverage audit passes all
six inventory entries. The four recorded means are:

| M | Profiled API event before/after (ms) | Operation span before/after (µs) |
|---|---|---|
| 8 | 0.4884 / 0.3755 | 169.618 / 22.780 |
| 128 | 0.5444 / 0.3941 | 191.216 / 50.666 |
| 64 | 0.5338 / 0.4001 | 171.912 / 27.214 |
| 256 | 0.5569 / 0.3925 | 209.724 / 89.688 |

These are all ten original samples, including first-call costs and observed
variation; no samples were discarded. Raw supported PMU rows are in
`T01-metrics.json`. Compilation identities and ELF text/relocation screens
are recorded separately. The selected runtime imports Triton 3.2.0 with
triton-ascend 3.2.2 and CANN 9.1.0; installed distribution metadata reports
Triton 3.5.0 and is retained separately. The device reports 20 Cube cores.
