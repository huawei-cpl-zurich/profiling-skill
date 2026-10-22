# A2 worked examples: DMA, scalar work and queue stalls are different

These results are from **Ascend910B4-1, an A2 product**. They are not A3
measurements and are not pooled with the historical Ascend910_9382 or
Ascend910_9362 series. The example's filename `a3_paired_probes.py` does not
change the observed product. The A3-only activity normalizer was not used:
the evidence preserves raw exported fields and descriptive per-engine ranges.

[Measurement metadata](a2-measured-pairs.json) binds the frozen source,
configurations, selectors, raw timing samples and CSV hashes. The timing
payloads record Torch 2.10.0, Torch-NPU 2.10.0, Triton 3.5.0 and
Triton-Ascend 3.2.2. Compare implementations within this series, not across
products or runtimes. All five new debug captures match this series' ordinary binary hashes;
[their metadata and complete late IR](../examples/a2-compiler-evidence/metadata.json)
are separate from the older [A3 matrix dumps](a3-synchronization-inspection.md#real-matched-matrix-example-flags-and-buffer-slots)
retain their own product/version bindings.

## Three controlled comparisons and their ordinary timing

Each specialization passed correctness, exact-name discovery and three
independent BasicInfo captures with rotating order, warmup 3, launch count 1
and kernel replay. All declared samples are retained. Diagnostic captures
below are separate replays; their durations are not ordinary timing samples.

| Control | Task samples, μs | Median, μs | Useful rate from median |
| --- | --- | --- | --- |
| Direct contiguous copy, N 1048576 | 23.799999,23.639999,24.5 | 23.799999 | 352.46 GB/s |
| Stride 2 copy, same count/allocation | 1040.319946,1004.679993,1023.179993 | 1023.179993 | 8.20 GB/s |
| Runtime identity-index gather | 608.739990,609.260010,608.960022 | 608.960022 | 13.78 GB/s |
| Matrix 512³, stages 1 | 36.380001,37.599998,37.18 | 37.18 | 7.22 TFLOP/s |
| Matrix 512³, stages 2 | 36.02,35.380001,35.400002 | 35.400002 | 7.58 TFLOP/s |

Memory rates use 8 MiB useful input+output bytes, excluding the indexed variant's
additional 4 MiB logical index reads. They are whole-task useful rates, not
physical DRAM bandwidth. All three memory variants allocate 8 MiB source,
4 MiB index tensor and 4 MiB output on the device—even the direct variant allocates
an unused index tensor. This 16 MiB tensor allocation is neither physical traffic
nor a measurement of the full process/workspace/allocator footprint.

Stride 2 deliberately changes the access pattern; it is not a semantics-preserving
rewrite. Identity gather has the same output as direct access, but loads its
indices at runtime. A known identity contract admits the direct reference;
arbitrary indices do not. The stage comparison holds matrix 512³, fp16 inputs,
fp32 accumulation/output and tile 32×32×64 fixed, changing only requested stages.
Its median task duration is 4.79% lower with stages 2 in these three captures;
that is an observed comparison, not a universal speedup or proof of overlap.

## Memory signatures across every exported Vector row

Each memory diagnostic contains 40 AIV rows. Ranges below summarize all 40 rows;
parentheses contain the median of the reported row values. They are not a
common-clock timeline, a device-wide utilization value or a critical-path sum.
All values in this table are microseconds from the separate PipeUtilization
capture, preserving the exporter’s scopes.

| Reported AIV metric | Direct | Stride 2 | Identity gather |
| --- | --- | --- | --- |
| Unit task window | 20.721–22.993(21.465) | 906.388–1026.264(983.386) | 590.979–604.494(597.256) |
| MTE2 active time | 19.250–20.490(19.650) | 904.917–1024.345(982.080) | 25.116–27.848(26.759) |
| Scalar time | 3.696–5.081(4.131) | 5.336–6.703(5.756) | 565.190–578.672(571.568) |
| Scalar intra-core wait | 0–0(0) | 0–0(0) | 25.916–28.192(26.815) |
| Scalar stalled by MTE2 IQ | 18.305–20.204(18.923) | 899.095–971.158(931.695) | 0–0(0) |

The strided case has a movement-dominated signature, not a scalar-dominated one:
MTE2 time spans almost its entire reported unit window while scalar time is
small. The indexed case has the opposite signature: long scalar time, much
shorter MTE2 activity and an explicit wait category. Calling both “irregular
loads” would hide the distinction. Inspect the emitted transfer dimensions and
address/index loops before choosing coalescing, reuse or scalar-path changes.
The binary-bound late IR below distinguishes the emitted paths at the captured
compiler stage; counters alone would not establish that distinction.

Direct access also has substantial MTE2 queue blockage despite being much
faster than the other cases. Zero `scalar_wait_time` therefore does not mean
zero blocking, and a queue counter is not automatically useless overhead that
can be removed while preserving the same work. Conversely, indexed access has
nonzero explicit wait with zero reported MTE2 queue blockage. Keep the
categories distinct rather than adding them into a latency budget.

Reported MTE2 active bandwidth here is **per-row input-path bandwidth**, not
the whole-task input+output rate in the first table. Its different scope and
traffic definition must remain explicit. Do not compare one row's few GB/s
directly with 352GB/s whole-task useful read/write rate or multiply a median
ratio by an independently captured task median.

## Matrix: unchanged Cube work, zero explicit waits, changed queue blockage

Both matrix pipe captures contain 20 AIC rows. Across every row, the reported
scalar explicit-wait counter is zero; Cube-active time has the same range,
3.432727–3.718788μs, in both variants. These observations do not make
synchronization or buffering irrelevant.

| Reported AIC metric | Stages1 min–max(median), μs | Stages2 min–max(median), μs |
| --- | --- | --- |
| Unit task window | 30.754–35.227(33.070) | 31.955–36.042(34.005) |
| MTE2 active time | 28.919–33.734(30.910) | 30.978–35.322(33.024) |
| MTE1 active time | 7.849–8.545(8.498) | 7.855–8.612(8.536) |
| Scalar stalled by MTE1 IQ | 28.672–33.413(31.716) | 26.988–31.006(28.890) |
| FIXPIPE active time | 2.033–3.523(2.660) | 2.335–4.727(3.205) |

The MTE1-queue category changes while scalar explicit wait stays zero. Queue
backpressure can depend on downstream dependencies as well as throughput;
it does not alone prove MTE1 capacity saturation. Other instrumented times
increase even though ordinary task timing improves. Separate replay counters
are evidence to investigate, not an additive explanation of the 1.78 μs ordinary
median difference. The binary-bound A2 IR below confirms buffer/event changes, while these
separate replay measurements do not establish the precise saved critical path.

ResourceConflictRatio separately reports 20 AIC rows per variant. Its blocked
instruction ratios are distinct from scalar wait time: for example,
`aic_cube_wait_ratio` ranges 0.968395–0.988769(stages 1) and
0.968430–0.988789(stages 2). A reported ratio near 1 is not proof that Cube
arithmetic is near capacity; the measured Cube-active time is unchanged and
small. Vector conflict fields on these pure-Cube rows are unavailable, not zero.

## Binary-matched late IR explains why the memory cases differ

All five diagnostic `npubin` hashes equal their ordinary counterparts. The
complete files retain the GraphSyncSolver stage and unchanged line numbering;
trailing compiler warnings are omitted with separate raw/display hashes.
These are observed compiler transformations, not decoded final-machine
instruction counts or a common-clock hardware trace.

- [Direct copy](../examples/a2-compiler-evidence/direct.mlir), lines 28–29 and
  49–57: a contiguous GM→UB `hivm.hir.load`, followed by UB→GM
  `hivm.hir.store`. Two 1 KiB UB slots already rotate across the blockified
  loop; “baseline” does not mean the compiler used one buffer.
- [Stride 2](../examples/a2-compiler-evidence/stride2.mlir), lines36–39 and
  59–73: a GM stride 2 load into UB stride 8, then `vdeinterleave` with eight
  channels before the contiguous store. The shown double-buffered staging is
  18 KiB versus direct's 2 KiB. This is a DMA-plus-layout-conversion path, not a
  scalar per-element GM load loop. Eight-channel UB layout does not itself
  establish eightfold physical DRAM traffic.
- [Runtime identity index](../examples/a2-compiler-evidence/indexed.mlir),
  lines 56–77: DMA first loads the index tile; a scalarized per-element loop
  then reads each UB index, loads one GM value, and stores it into UB, before
  an ordinary UB→GM DMA store. `PIPE_S` producer/consumer flags surround this
  loop. That late-IR path is consistent with the measured scalar-dominated
  signature; runtime identity values did not become the direct-copy lowering.

The indexed loop is explicit at lines 62–68:

```mlir
scf.for %arg10 = %c0 to %22 step %c1 {
  %23 = memref.load %13[%arg10] {DiscreteMemAccess} : memref<256xi32, #hivm.address_space<ub>>
  %24 = arith.index_cast %23 : i32 to index
  %reinterpret_cast_5 = memref.reinterpret_cast %arg3 to offset: [%24], sizes: [1], strides: [1] : memref<?xf32, #hivm.address_space<gm>> to memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  %25 = memref.load %reinterpret_cast_5[%c0] : memref<1xf32, strided<[1], offset: ?>, #hivm.address_space<gm>>
  memref.store %25, %12[%arg10] : memref<256xf32, #hivm.address_space<ub>>
} {ExtractedLoadOrStore}
```

[Matrix stages 1](../examples/a2-compiler-evidence/stages1.mlir) uses cbuf
bases 0/4096. [Stages 2](../examples/a2-compiler-evidence/stages2.mlir),
lines 71–83, selects modulo 2 event IDs and two slots per input at
0/8192 and 4096/12288. Input cbuf extents grow 8 → 16 KiB; the matrix macro
retains `sync_related_args`. Lines 132–142 order matrix→FIXPIPE output and
FIXPIPE→matrix accumulator release, with explicit CC→GM `fixpipe` output.
These A2 artifacts independently confirm the transformation; historical A3
artifacts are not used as proof of the A2 binary's internals.

## A measured reference curve with its repeat contract visible

Five additional sizes each passed correctness and fresh selector discovery.
Each then produced **three matching samples inside one BasicInfo capture**,
with warmup 3, launch count 3, kernel replay and three workload kernel launches
in one application invocation. The profiler performs its own additional replay
executions; these are not three application invocations.
These are not three independent process captures. The middle points below
come from the paired study's three independent one-sample captures; they are
shown with their different repeat policy, not pooled into one sample set.

| Direct-copy N | Repeat policy | Median(min–max), μs | Useful read+write GB/s |
| --- | --- | --- | --- |
| 65536 | One capture × three samples | 4.26(4.00–4.28) | 123.07 |
| 1048576 | Three captures × one sample | 23.80(23.64–24.50) | 352.46 |
| 16777216 | One capture × three samples | 348.30(347.34–349.10) | 385.35 |

| Matrix N³, stages 2 | Repeat policy | Median(min–max), μs | Useful TFLOP/s |
| --- | --- | --- | --- |
| 256³ | One capture × three samples | 7.44(7.36–7.64) | 4.51 |
| 512³ | Three captures × one sample | 35.40(35.38–36.02) | 7.58 |
| 1024³ | One capture × three samples | 204.86(204.16–205.62) | 10.48 |
| 2048³ | One capture × three samples | 1417.16(1412.70–1418.82) | 12.12 |

The copy curve shows why 385GB/s is not a matched 65536-element target:
startup, program count and working-set regime differ. The GEMM rate increases
with shape but these points do not establish a hardware ceiling or validate
that the 512³ bottleneck signature holds at 2048³. Repeated warmup/replay
reuses each allocation; cache residency and physical traffic remain unmeasured.
The largest copy uses 256 MiB of the three explicitly allocated device tensors,
not a measured whole-process footprint.

For the identity-index contract, the same-size direct baseline is a demonstrated
attainable alternative: its 23.80 μs median versus 608.96 μs identifies a large
observed path gap worth removing if the identity contract holds. Arbitrary
indices cannot assume that alternative. For matrix 512³, the stages 2 median is
4.79% lower than stages 1; this is neither 95.21%
hardware efficiency nor a capacity bound. The larger-tile experiment below
turns the source-requested traffic model into a concrete next hypothesis.

## A concrete next matrix experiment, not an executed result

A diagnostic 64×64×64 tile at matrix 512³ and stages 2 would halve this source's
requested input loads from 16 MiB to 8 MiB, plus 1 MiB output. This is source-level
request accounting, not a prediction of halved DRAM traffic or task latency.
See [the byte-count derivation](a3-performance-envelope.md#the-paired-gemm-three-byte-counts-that-must-not-be-mixed).

To test it, copy the launcher into a separate candidate and change all three
coupled descriptions together: the grid's `cdiv(n,32)` divisors to 64, the
passed `BLOCK` from 32 to 64 while keeping `BLOCK_K=64`, and the reported tile
in `describe()` to `[64,64,64]`. Preserve the original timed source and retain
the new source/compiler hashes. Check correctness before fresh selector
discovery and ordinary timing.

The grid decreases from 256 to 64 programs at N 512. Larger operand/accumulator
buffers may reduce residency or trigger compiler resource fallback; fewer
programs can change load balance. Neither improved cache reuse nor improved
parallelism is guaranteed. Compare actual buffer extents and all-row movement,
queue and compute evidence with the original, using matched timing contracts.
The bundled frozen source has not been changed, and this 64-tile experiment
has not been executed in the retained series.

## Inspect the complete rows without loading a verbose report tree

Metadata links canonical CSVs containing every metric column and row, including
empty/NA markers. It records each original-export SHA-256 separately from the
displayed CSV SHA-256: canonical newline formatting changes bytes. BasicInfo
copies omit process/device identifiers explicitly; selectors, task duration,
block dimensions and frequency remain bound to the same capture.

From the skill folder, a standard-library reader can inspect any selected field:

```python
import csv
from pathlib import Path

path = Path('examples/a2-measured-evidence/indexed-PipeUtilization.csv')
with path.open(newline='') as stream:
    for row in csv.DictReader(stream):
        print(row['block_id'], row['sub_block_id'],
              row['aiv_scalar_time(us)'], row['aiv_scalar_wait_time(us)'])
```

Use the [A2/A3 field glossary](a2-a3-msprof-op.md#field-glossary-and-calculations)
for event meanings and denominator checks. Do not infer a normalization window,
capacity denominator, source phase or cross-replay overlap from these rows.
