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

The previous explicit simulator control produced bound events but exited with
a child signal after numerical success. Those events remain partial evidence,
not a clean acquisition or hardware timing. Label-based phase inference is not
a source map; retain core/lane/PC/units and validate any actual source mapping.

## Historical compiler audit and extension controls

Implementation and device validation are in progress. No new conclusions are
accepted yet for striding, irregular indexing, flag/wait placement, small
matmuls, or sanitizer detection coverage. Their final entries must include:

1. Exact controlled code difference and correctness/reference contract.
2. Original or reconstructed compiler evidence and confidence.
3. Matched per-core counter changes, timing distribution, or race diagnostics.
4. Competing explanations, counterexamples, and unsupported observations.
5. Practical agent guidance that follows only from the observed evidence.

New profiling controls retain all five BasicInfo samples and three pipe
captures per variant; timing spread above 5% lowers confidence without deleting
the evidence. Instrumented sanitizer runs are never performance baselines.
