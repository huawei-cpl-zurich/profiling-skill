# Tail-only StreamK on A2/A3

This controlled FP16 GEMM pair changes how the incomplete final output-tile wave
uses the Cube cores. It is distinct from splitting K for every output tile.
The two arms use the same Catlass `StreamkMatmul` / `BlockMmad`, 128×256×256 L1
and 128×256×64 L0 tiles, two operand stages, ShuffleK enabled, unit flags, output
type, and mixed Cube/Vector launch. The control supplies an ordinary scheduler:
one core computes an output tile's full K dimension. The candidate supplies the
original tail scheduler: complete waves remain unsplit, while the final wave's
K blocks are distributed across cores. The source schedules that tail before
the last ordinary wave to overlap its reduction; ordering is part of this change.
This control includes the common mixed-core synchronization shell. It is a
mechanism comparison, rather than a vendor-library baseline comparison.

Source: Catlass revision `40aced927fdaaa0f3eaa9c6f5c5d859e06392f55`,
[`examples/37_streamk_matmul/streamk_matmul.cpp`](https://github.com/Ascend/catlass/blob/40aced927fdaaa0f3eaa9c6f5c5d859e06392f55/examples/37_streamk_matmul/streamk_matmul.cpp),
`include/catlass/gemm/kernel/streamk_matmul.hpp`, and
`StreamkGemmIdentityBlockSwizzle` in `block_swizzle.hpp`.
The managed source citation is
`ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#include/catlass/gemm/kernel/streamk_matmul.hpp`.
The derivative retains [CANN_LICENSE](CANN_LICENSE).

The initial contract accepts contiguous row-major FP16 operands with dimensions
at most 32768, M/N divisible by16 and K divisible by256. The frozen study requires
20 healthy idle Cube cores, with 40 accompanying Vector cores. Hardware
synchronization comes from `aclrtGetHardwareSyncAddr`, passed unchanged to
`SetSyncBaseAddr`. An independent CPU FP32 GEMM checks bounded inputs in
[-0.125,0.125], with fixed `atol=rtol=0.02`; input preservation is checked only
once. M/N tile tails must overwrite every valid element of the initially NaN
output. Arbitrary odd ND dimensions are outside this example's contract.

Each forward includes all Cube work, FP32 partial writes, cross-core
synchronization and the Vector merge/cast. Both arms preallocate 5,242,880 bytes
of workspace capacity. Every partial consumed by the reducer is overwritten by
its owning Cube task during that same forward, so there is no required initial
workspace value or omitted zeroing operation. Allocation, input transfer and
compilation are outside the measured operation. Profiler operation spans include
the entire launched mixed kernel. Instrumented caller/event timing and active
kernel time are separate diagnostics; neither is an uninstrumented application
latency measurement.

With 41 or61 output tiles on20 cores and long K, tail splitting may reduce the
last-wave imbalance. With78 tiles or a short K loop, partial traffic, additional
Vector work and synchronization may cost more than the recovered parallelism.
In code, inspect the output-tile remainder, K-block count and partial-buffer
ownership. In the profiler, inspect complete operation span, mixed core geometry,
Cube/Vector task durations and reported pipe composition. Pipe busy ratios do
not establish physical memory traffic, whole-device utilization or distance to
roofline. Retain both positive and negative cases without changing the matrix.

Build before running (the filtered dependency closure must contain the cited
StreamK header and a verified pinned `manifest.json`):

```bash
python examples/optimization-library/catlass/tail_streamk_compile.py \
  --catlass "$CATLASS_FILTERED" --output "$CATLASS_TAIL_DIRECTORY"
export CATLASS_CUBE_CORES=20
export PYTHONPATH="examples/optimization-library/catlass:$PYTHONPATH"
python scripts/optimization_pair.py --adapter tail_streamk_adapter:factory \
  --case "$CASE_JSON" --provenance "$PROVENANCE_JSON" \
  --output "$NEW_OUTPUT" --cache "$ORACLE_CACHE"
```

Split `tail_streamk-cases.json` into individual case objects for the CLI. Each
profile case checks correctness once per arm, then records exactly ten complete
forwards per arm with no warmup or profile-time inspection. Guard objects have
`role: "guard"`, which automatically selects correctness-only execution; the
explicit `--correctness-only` flag provides the same behavior. Guard execution
produces no captures or performance claims. Cache each CPU oracle once and keep
failed or partially captured source versions rather than replaying them.

## Retained A3 results

All four fixed pairs and both guards passed on Ascend910B4-1 physical die5.
The table shows mean ± sample SD of all ten profiled complete-operation spans
in microseconds, with no samples dropped. Each launch is one `MIX_AIC` task,
with reported Block Num20 and Mix Block Num40. Active-union equals span and
inter-task gap is zero for these single-task operations.

| Case | M×N×K | Before µs | After µs | Before/after |
|---|---|---:|---:|---:|
| F06-01 | 128×10496×4096 | 77.882 ± 19.545 | 59.138 ± 15.743 | 1.317× |
| F06-02 | 384×6656×512 | 17.588 ± 2.414 | 21.736 ± 3.632 | 0.809× |
| F06-03 | 128×15616×6144 | 211.084 ± 8.908 | 183.384 ± 4.600 | 1.151× |
| F06-04 | 384×6656×1024 | 27.686 ± 2.404 | 30.572 ± 3.695 | 0.906× |

The four-case geomean is 1.027×. First-launch variation is
visible, especially in case01, and is retained. These data support the stated
benefit/cost hypotheses for this matrix; they do not establish a universal
speedup. [Structured evidence](../../../experiments/optimization-library/w003-tail-streamk/evidence.json)
retains every span/event sample, exact task identity, PMU values, correctness
diagnostics and source/binary/report hashes. The initial version failed only
at compile time because its C++ `Tile` alias collided with `tla::Tile`; it made
zero device calls. That source and failure log are retained alongside the
corrected version rather than removed from the experiment history.
