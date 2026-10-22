# A3 profiler characterization findings

## Status and evidence boundaries

This report separates measured performance, per-core profiler observations,
compiler-confirmed mechanisms, and unresolved hypotheses. The initial battery
contains 70 cells (35 before/after pairs): all ran, 39 met its timing stability
gate, and 31 retained valid but timing-inconclusive evidence. It retained 385
timing samples and 216 pipe captures. Twelve pairs have accepted timings on
both sides. The historical gate selected the first consecutive three-sample
window within 5%; that is not a statistical confidence interval.

The historical source is frozen at `e4d7cff`. It is not being reprofiled for
the compiler audit. New compile-only reconstructions cannot establish what a
historical measured binary did unless their equivalence is verified. Native
timing/counters and simulator scheduling are separate evidence classes.

The extension completed 54 new profiling cases: 22 memory, 20 synchronization,
and 12 small-matmul cases, each with five timing and three pipe captures. All
passed numerical correctness. Thirty-seven timing distributions exceed 5%
spread and remain explicitly low confidence. Separately, 108 sanitizer captures
qualify detection rather than performance, and four clean simulator captures
qualify the binary replay route. No historical kernel was relaunched.

## Code changes that produced useful profiler signals

Pipe values below are medians from three separate profiling captures for
matched block 0 and the named sub-block. They are not whole-device totals,
do not necessarily represent the slowest core, and are not additive latency
components. Latency is a separate `msprof op` BasicInfo capture.

| Controlled change | Matched per-core observation | BasicInfo before / after | Supported interpretation |
| --- | --- | --- | --- |
| Fixed masked loop of 256 iterations -> local extent of 64 | AIV vector time 6.155 -> 1.995 us; scalar/vector stall time 5.629 -> 1.564 us | 9.800 / 4.740 us (2.07x) | Removing source iterations correlates with less vector execution and waiting; inspect lowered loop before claiming exact instruction savings. |
| Scan 128 head flags -> consume precomputed rank | AIV scalar time 3.825 -> 2.876 us | 5.300 / 4.400 us (1.20x) | Scalar metrics identify removable indexing/control cost. Preparing the rank is outside the measured kernel. |
| Reduction key tile 64 -> 128, size 2048 | AIV MTE2 time 5.778 -> 4.088 us; MTE2 ratio 72.62% -> 72.69% | 10.180 / 7.460 us (1.36x) | Duration improves without a meaningful change in dominant pipe fraction. Tiling can help while the dominant pipe stays the same. |
| Column-by-column multiply/reduce -> matrix-tiled `tl.dot`, size 32 | Before AIV MTE2 ratio 97.77%, time 145.914 us; after AIC Cube activity is present | 152.337 / 1.940 us (78.53x) | Algorithm/access structure and execution path changed. This tiny synthetic control does not establish production GDN/BSA gains or that the original kernel saturated physical bandwidth. |
| One matrix row -> four rows per program, size 64 | AIC MTE2 time 1.454 -> 0.432 us | 6.180 / 4.840 us (1.28x) | Consistent with reducing repeated loading by grouping work; physical traffic and cache effects need separate evidence. |
| Predicate loads instead of load-then-mask, amplified 128 times | AIV MTE2 time 17.485 -> 15.275 us; vector time 9.639 -> 15.706 us | 30.219 / 38.319 us (27% slower) | Reduced observed transfer time can lose to increased vector/indexing cost. These are derived volatile runtime-loop probes, not original-kernel timings. |

The broad direction of the first five comparisons also survives using all
retained timing samples rather than the selected stable window. Small
differences are not treated as established gains: scalar decision broadcasting
at 128 amplified repetitions is effectively flat (48.339 / 48.079 us), despite
large shifts in scalar activity.

## What not to teach from these numbers

- High pipe activity is not proof of capacity saturation. Resource activity,
  active bandwidth, transferred bytes, and whole-device utilization have
  different denominators and scopes.
- A faster implementation can have a higher pipe fraction because total time
  shrank. Do not optimize a percentage independently of execution cost.
- Separate captures cannot prove simultaneous overlap or a critical path.
  Do not sum overlapping pipe times, pool unrelated cores, or derive time by
  multiplying a ratio by an independently aggregated duration.
- A high-level `tl.load`, broadcast, or `tl.dot` is not an instruction-level
  explanation. Verify retained operations, layouts, copies, and synchronization
  in the compiler artifacts and qualify unavailable lowerings explicitly.
- A numerically passing kernel is not necessarily race-free. Sanitizer findings,
  profiler measurements, and numerical checks answer different questions.

## Timeline acquisition correction

All twelve historical automatic TimelineDetail captures failed context/argument
dumping. This is not evidence that A3 has no instruction-timeline support.
The documented automatic route generates a simulated instruction timeline and
excludes Triton; explicit simulator profiling is a separate supported workflow
to qualify. See the [profiler guide](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)
and [Triton profiling guide](https://gitcode.com/Ascend/triton-ascend/blob/release%2F3.5.x/docs/zh/debug_guide/profiling.md).

The explicit Python simulator application produced bound events but exited with
a child signal after numerical success. Replaying the same compiled kernel
through the simulator's binary configuration route fixes acquisition: two clean
repeats each of an add control and a 32-round multiply/add control completed.
The route verifies the binary and configuration, uses the observed product's
simulator library, and does not guess an unknown kernel ABI. The worker still
requires the observed `--soc-version` for provenance, but its child `msprof`
config-mode command must omit that flag: the installed tool rejects the combination
despite returning zero. Exit status alone is therefore not an acquisition gate.

| Simulator control | Events | Core-local VECTOR active | MTE2 active | MTE3 active |
| --- | ---: | ---: | ---: | ---: |
| Add | 47 | 0.022703 us | approximately 0.159 us | 0.126486 us |
| 32 multiply/add rounds | 721 | 0.832973 us | approximately 0.161 us | 0.126486 us |

Both repeats agree to the shown precision. The increased VECTOR activity is
consistent with the controlled arithmetic increase; these are **simulated
core-local intervals**, not hardware latency, bandwidth, or capacity saturation.
Label-based phase inference is not a source map. Preserve core/lane/PC/units;
general source-phase mapping and actual overlap need their own validation.

## What the compiler audit changes

All 70 historical cells map to 53 source-dispatch specializations, and all 53
compiled without launching their kernels. All 36 available original amplified
TTIR/TTAdapter artifacts were hash-verified. Reconstructions match those early
stages after normalizing only absolute source paths; this does not establish
historical final-binary equivalence. See the [detailed compiler audit](a3-mlir-retrospective.md)
for artifact hashes, stage boundaries, and exact operation references.

| Source-level hypothesis | Compiler evidence | Revised profiler interpretation |
| --- | --- | --- |
| Predicating a load avoids false-lane traffic | Original TTAdapter still copies the contiguous valid FP32 slice and applies vector predicates afterward; reconstructed scheduled IR retains a unit-stride GM-to-UB load. | Lower observed MTE2 time is not proof of fewer physical bytes. Additional vector masking is consistent with the measured slowdown. |
| A scalar decision is cheaper than a vector decision | Original IR retains scalar extraction and conditional work; reconstructed scheduled IR adds scalar-to-vector and vector-to-scalar set/wait dependencies. | Higher scalar activity can represent extra work and cross-pipe dependencies, not a useful optimization. IR alone does not measure wait duration. |
| Identical TTIR means schedule options did nothing | Changing warps/stages together leaves early stages identical but changes late IR from single-address to rotating UB buffers with additional flags. | Inspect scheduled IR before declaring tuning knobs inert. This combined control cannot distinguish the contribution of warps from stages. |
| `tl.dot` necessarily means Cube execution | This reconstruction actually contains MMAD into CC and FIXPIPE from CC to GM. | The execution-engine change is supported for this compiler/control, rather than inferred from the source spelling. |
| Moving loads in source proves overlap | Load order partly survives lowering, but runtime `scratch=x` aliases arguments and is not an AST compile constant. | Neither source order nor early IR proves independent memory streams or realized overlap. Obtain a bound instruction trace. |

## Historical compiler audit and extension controls

### Strided and indexed memory

All 22 new cases passed numerical correctness and retained five BasicInfo and
three PipeUtilization captures each. Ten cases exceed the 5% spread threshold;
they remain in the report as low-confidence timing observations. Parentheses
below show `(maximum - minimum) / median`, not a confidence interval. Large-case
pipe fractions are the median of three captures for block 0 / vector0 only.
They overlap and must not be added.

| Access | N=4,096 median us (spread) | N=1,048,576 median us (spread) | Large-case MTE2 fraction | Large-case scalar fraction |
| --- | ---: | ---: | ---: | ---: |
| Stride 1 | 3.820 (18.8%) | 30.199 (13.8%) | 93.15% | 16.66% |
| Stride 2 | 17.820 (12.2%) | 892.322 (3.2%) | 99.93% | 0.50% |
| Stride 4 | 12.540 (13.9%) | 813.684 (4.3%) | 99.92% | 0.55% |
| Stride 8 | 9.860 (10.3%) | 514.170 (1.4%) | 99.88% | 0.85% |
| Stride 16 | 8.300 (18.8%) | 360.753 (1.3%) | 99.83% | 1.22% |
| Stride 32 | 6.880 (10.8%) | 268.735 (2.8%) | 99.73% | 1.69% |
| Stride 64 | 6.740 (15.7%) | 263.715 (4.3%) | 99.75% | 1.79% |
| Indexed contiguous | 9.720 (4.1%) | 478.850 (0.7%) | 4.75% | 95.42% |
| Indexed local-32 permutation | 9.800 (10.0%) | 481.690 (1.1%) | 4.74% | 95.43% |
| Indexed global permutation | 23.680 (4.1%) | 2352.813 (0.1%) | 0.99% | 99.05% |
| Indexed hot-set of 64 | 9.000 (6.7%) | 342.453 (1.3%) | 6.35% | 93.89% |

The indexed cases use the same compiled indirect-load path with different
runtime indices, including the intentionally contiguous index control. This
separates the cost of the compiler's indirect path from index locality:

- Retained TTAdapter for direct stride 1 contains a unit-stride `memref.copy`
  (line 22); stride 2 contains a stride-2 copy (line 24).
- Indexed contiguous and global-permutation TTAdapter copy the index vector,
  then use an `scf.for` containing a single-element FP32 `memref.load`
  (lines 25/28/32). At this stage the gather is scalarized, despite the
  high-level tensor `tl.load`. These are artifacts returned by the measured
  dispatches, not retrospective reconstructions; late instruction scheduling
  is not retained by this path.
- Global permutation is 4.91x slower than indexed contiguous at the large size,
  with scalar fraction rising from 95.42% to 99.05%, while MTE2 time stays around
  22 us on the matched core. A memory-access problem therefore need not present
  as high MTE2 activity. Scalar activity is not automatically arithmetic/control
  work: scalarized indirect loads and their latency are a competing explanation.
- Large strides retain approximately 99.7-99.9% MTE2 activity yet differ by
  over 3x in latency. Near-full activity alone does not establish efficient
  transfer, peak bandwidth, or that code changes cannot help. Increasing stride
  also changes address span and allocation size; the nonmonotonic ordering does
  not isolate cache, bank, or transaction behavior. Those mechanisms remain
  hypotheses, not device-fault explanations.

Evidence identifiers for the three TTAdapter examples are
`cd4fc8f9e4fd59f56b726f3d6b4e4571d61be28833d624df1a8f0b60b1b15e04`
(direct), `7e5dc7eb52d56d483e9414d228198729e6c97fbd805a7eadd8e7c81a56aeff13`
(stride 2), and
`64912c319a15284cfb21e4a83e94b87607753800521f03b3661c941a1872f28a`
(indexed contiguous). Raw timing samples, all per-core rows, command logs, and
compiler text are retained separately from this compact report. All 198 raw
BasicInfo exports contain exactly one row; none needed selector-based removal
of another computation kernel.

### Synchronization placement

All 20 native controls passed full-array numerical checks. They preserve work
within each comparison group while moving source-level event set/wait calls.
The installed HIIPU object decoder prints `<not available>` for every machine
instruction, so emitted flag placement and compiler-inserted synchronization
remain unverified. Source intent, numerical correctness, and measured metrics
are distinct evidence. The independent AIC copy in mode 2 is not numerically
checked; the checked outputs and handshake must not be presented as a proof of
that copy's contents.

All 20 variants completed five timing and three pipe captures. Sixteen timing
distributions exceed 5% spread; none of their samples were discarded.

| Intra-core source control | Overlap median us (spread) | Early-wait median us (spread) | Late-set median us (spread) |
| --- | ---: | ---: | ---: |
| MTE2-to-vector, 1 round | 4.340 (7.4%) | 4.340 (12.0%) | 4.040 (11.9%) |
| MTE2-to-vector, 32 rounds | 4.420 (9.5%) | 5.700 (7.7%) | 4.440 (3.2%) |
| Vector-to-MTE3, 1 round | 2.680 (6.7%) | 2.900 (23.4%) | 2.820 (12.1%) |
| Vector-to-MTE3, 32 rounds | 3.900 (8.7%) | 3.840 (8.9%) | 4.320 (6.9%) |

For the 32-round MTE2-to-vector control, the matched vector-active median stays
1.767 us across all placements, while scalar/vector-stall rises from 0.787 us
with overlap placement to 1.843 us with early wait. That is a useful measured
association with retained useful work, not proof that this counter measures a
particular `WaitFlag` instruction. The timing increase is observable, but its
exact speedup is not accepted under the spread gate. Conversely, moving the
vector-to-MTE3 wait does not yield a consistent large stall or latency change.
Do not teach "every wait change must move every wait-like counter."

Inter-core captures:

| Source control | Overlap median us (spread) | Early-wait median us (spread) | Block 0/vector0 wait-IB us, overlap / early |
| --- | ---: | ---: | ---: |
| Mode 0, balanced | 7.680 (8.3%) | 7.920 (12.9%) | 0.836 / 0.843 |
| Mode 0, imbalanced | 7.920 (10.9%) | 9.200 (7.8%) | 0.684 / 0.739 |
| Mode 2, balanced | 8.780 (4.8%) | 9.600 (4.8%) | 0.687 / 0.668 |
| Mode 2, imbalanced | 8.780 (2.1%) | 10.220 (5.7%) | 0.777 / 0.874 |

Every matched AIV `scalar_wait_time` median is zero despite explicit
synchronization in the source controls. Mode 2 also exports zero AIC
`scalar_wait_time`. A zero in that field therefore must not be translated to
"the kernel has no synchronization cost." Balanced mode 2 has a measured
0.820-us increase with early-wait source placement without an increase in
matched wait-IB time; exported wait metrics do not directly reconstruct the
source's waits or the critical path. Other comparisons exceed the timing-spread
threshold and are not accepted as precise speedups. These observations motivate
instruction-trace validation rather than inventing a direct counter-to-API map.

### Small matmuls

Six shapes each completed native low-level Mmad and single-program Triton dot
correctness, discovery, five timing captures, and three pipe captures. Nonuniform
deterministic FP16 inputs with FP32 accumulation match the CPU reference exactly.
Native A ZZ / B ZN / output NZ packing includes asymmetric shape checks. These
are one-block/one-dot controls, not optimized device-wide GEMM implementations.

| M x N x K | Native median us (spread) | Triton median us (spread) | Native / Triton Cube-active median us |
| --- | ---: | ---: | ---: |
| 16 x 16 x 16 | 2.260 (8.0%) | 2.400 (10.0%) | 0.025000 / 0.025000 |
| 32 x 32 x 32 | 2.380 (11.8%) | 1.900 (14.7%) | 0.028333 / 0.028333 |
| 64 x 64 x 64 | 2.880 (9.0%) | 2.060 (15.5%) | 0.059444 / 0.059444 |
| 128 x 128 x 128 | 6.140 (22.8%) | 3.060 (2.6%) | 0.308333 / 0.308333 |
| 16 x 64 x 32 | 2.340 (18.8%) | 1.820 (12.1%) | 0.028333 / 0.028333 |
| 64 x 16 x 32 | 2.320 (11.2%) | 2.600 (5.4%) | 0.028333 / 0.028333 |

Pipe values are three-capture medians for block 0 / cube0. Equal Cube-active
time at each shape coexists with different whole-kernel durations. For 128-cubed,
native versus Triton MTE2 is 2.207 / 0.968 us, FIXPIPE 1.978 / 0.451 us, and
scalar 1.285 / 0.206 us. These overlapping intervals must not be summed or
subtracted to manufacture a latency decomposition. Only the Triton 128-cubed
timing distribution passes the spread gate, so this table is not an accepted
precise native-versus-Triton speedup ranking.

The useful direction for further work is transfer, output layout and issue
overhead, not different measured Cube-arithmetic time. Native runtime dimensions
and TPipe setup versus Triton constexpr dimensions are candidate explanations,
not machine-code-verified causes. These small controls do not saturate Cube.
Tiny MTE3 intervals with implausibly high active-bandwidth values must not be
used as a bottleneck diagnosis when the actual output path is FIXPIPE.

### Race-detector qualification

Native racecheck completed three positive-mutant and three repaired/disjoint
captures for each family: UB RAW, UB WAR, UB WAW, inter-core GM overlapping
writes, and inter-core GM producer/consumer. All five families produced 3/3
positive detections and 0/3 repaired detections. Reports retain source lines,
PCs, pipes and memory spaces. The RAW-labelled mutant produces multiple related
hazards: MTE2 writes at source line 35 / PC 0x3f6c and vector reads/writes at
line 40 / PC 0x4714, plus a downstream transfer hazard. Do not pretend that a
source-labelled control isolates exactly one reported dependency.

Native synccheck produced no finding for all 30 runs, including the intentional
race mutants. It is not a substitute for racecheck. Most native memcheck runs
warn about `FFTS_BASE_ADDR` reset; these are retained as other findings, not
silently converted to clean memory checks. The GM-overlap mutant additionally
receives an out-of-bounds warning despite recorded allocation/address ranges
appearing sufficient. That warning needs a bounds/ownership audit; it is neither
an established true out-of-bounds access nor a race diagnosis merely because it
mentions the same source.

The Triton overlapping-GM-write mutant also produced racecheck detections in
3/3 captures, with 0/3 for disjoint ownership. All six synccheck captures were
no-finding. Disjoint memcheck runs retain vector-register warnings; overlapping
ones report bounds warnings, which are not automatically accepted as true
out-of-bounds accesses. Current-parser offline validation bound all 108 native
and Triton reports to completed target-kernel analysis, without rerunning tools
or kernels. Both language qualification gates passed the complete three-repeat
positive/negative racecheck matrix. An undetected positive control would instead
produce an explicit inconclusive qualification, never a silent success.

For reproducible use, run an exact-target canary first, use native debug and
sanitizer compilation or a fresh sanitizer-enabled Triton cache, and delimit
tool options from the workload with `--`. The installed tool rejects the bare
`--full-backtrace` flag despite advertising it. Require completed analysis of
the exact kernel for positive as well as negative results: an auxiliary Torch
warning, partial report, successful exit, or quiet log does not qualify detection.
See [the sanitizer workflow](a3-sanitizer-characterization.md).

## Implications for the profiling skill

The defensible rules from this battery are:

1. Inspect the compiled memory path before mapping a tensor load to MTE2.
   Scalarized gathers are a measured counterexample to that shortcut.
2. Inspect late scheduling, not only TTIR: buffering and cross-pipe flags can
   change after identical early IR.
3. Treat high activity as occupancy of a pipe's measurement scope, not proof
   of efficient physical transfer or a terminal optimization condition.
4. Interpret wait-like fields by their documented scope. Neither their sum nor
   a zero-valued generic-looking field is a source-level wait reconstruction.
5. Use the A3 simulator for qualified instruction-order hypotheses, not hardware
   latency. Preserve exact binary, core, lane, PC, units and acquisition status.
6. Use racecheck for the hazard coverage actually demonstrated; retain repaired
   controls and tool warnings. Numerical correctness and profiler counters do
   not establish race freedom.

Remaining hypotheses should be tested before becoming stronger skill rules:

| Hypothesis | Before / after microbenchmark | Evidence that would support it |
| --- | --- | --- |
| Indirect-load scalarization, not just locality, dominates indexed-contiguous cost | Identical contiguous logical addresses supplied through a runtime index tensor / compile-time direct address arithmetic, then a verified specialized indirect lowering | Late IR changes scalar GM loads to an efficient copy/gather; scalar time falls with lower total latency. Existing controls support the path difference, not a universal remedy. |
| Nonmonotonic strides reflect transaction/locality behavior | Strides 2/4/8/16/32/64 in a common fixed allocation and controlled working set / current stride-dependent spans | Same compiled transfer path and stable repetitions, plus relevant bytes/cache/transaction metrics; MTE2 activity alone is insufficient. |
| Native small-dot overhead is transfer/layout/setup, not Cube arithmetic | Current runtime-dimension native control / constexpr-specialized dimensions and an independently controlled output-layout change | Identical Cube work and numerical outputs, lower transfer/FIXPIPE/scalar costs, verified scheduled code. Change one factor at a time. |
| Source wait placement changes realized overlap | Current matched-work early-wait / overlap variants, with explicitly verified simulator ABI and instruction trace | Bound set/wait PCs and pipe intervals move as intended. Source order or a generic wait counter is not enough. |

These are report hypotheses, not automatically scheduled follow-up tasks.

New profiling controls retain all five BasicInfo samples and three pipe
captures per variant; timing spread above 5% lowers confidence without deleting
the evidence. Instrumented sanitizer runs are never performance baselines.
