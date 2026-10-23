---
name: ascend-optimization-library
description: Choose and explain Ascend Triton and Catlass C++ kernel optimizations using source patterns, guarded code examples, and paired correctness/profile evidence.
---

# Ascend optimization library

Use this library to choose a transformation, explain why it may help, and state
when its cost or numerical contract makes it unsuitable. Read only the relevant
recipe. A gallery result validates its recorded case and implementation; it is
not a universal dispatch threshold or a claim about every Ascend target.

## Find a mechanism

| Source or profile symptom | Read | Main competing cost |
| --- | --- | --- |
| Row scale repeated inside a dot reduction | [Invariant scaling hoist](references/scaling-hoist.md) | Output-side work and changed rounding |
| GEMM result immediately consumed by an add | [Epilogue fusion](references/epilogue-fusion.md) | Live resources, synchronization and GM staging |
| Nearby query programs reload the same K/V tiles | [Query blocking](references/query-blocking.md) | Padded work and live accumulators |
| Long tokenwise state recurrence with few independent heads | [Chunked recurrence](references/chunk-recurrence.md) | Preparation, workspace and matrix-operand rounding |
| Output-task count just exceeds the observed core count | [Tile balance](references/tile-balance.md) | Layout-dependent operand movement |
| Few M/N tiles and a long K reduction | [SplitK](references/split-k.md) | Partial-result writes, reduction and synchronization |
| Repeated K-stage movement followed by compute | [Multibuffering](references/multibuffering.md) | Additional L1/L0 reservation and fill/drain |
| Several successive output tasks per core | [Next-task preload](references/preload.md) | Bookkeeping and no useful successor on small grids |
| The same A rows are reused across a core's output tasks | [L1 operand residency](references/l1-residency.md) | Capacity and reuse bookkeeping |
| Repeated strided ND movement with inconvenient alignment | [Padding/layout conversion](references/padding-layout.md) | Complete conversion and workspace cost |
| Many cores begin the same multi-tile K traversal | [ShuffleK](references/shuffle-k.md) | Indexing and changed summation order |
| Many output tiles have a plausible operand-locality problem | [Output-program ordering](references/program-order.md) | Mapping cost or worse reuse of the other operand |

The recipes contain code formats, preconditions, falsifiers and gallery data.
[Gallery evidence](references/gallery-evidence.md) defines the timing boundaries;
its JSON companion retains every ten-sample series and original PMU field names.
Only gallery examples belong to this skill snapshot. All 28 original gallery
pairs have correctness and ten-forward identity evidence. Residency, ShuffleK
and the tested output rotation have not demonstrated a timing gain in these
regimes; read those cards as conditional hypotheses, not proved speedups.

## Make a conditional decision

Connect an actual source pattern to a bottleneck hypothesis. Then match shapes,
strides, physical layout, dtypes, required outputs, target capabilities and
compiled resources. Logical shape does not establish physical storage. A tile
that compiles is not yet a correct kernel; an admitted small case does not
prove a larger resource envelope.

Choose **apply**, **avoid**, or **insufficient evidence**. An actor should give
concrete edits, dispatch predicates, numerical guards and a supported alternate
path for the task's remaining valid inputs. An analyzer should name the observed
source/profile evidence, confidence, competing explanation and a falsifying
comparison. Describe the optimization in one short sentence before the detail.
Use applicable kernel-authoring instructions for implementation; these recipes
do not replace the requested public interface or the task's coverage contract.

Keep the paired change interpretable. Preserve unrelated scheduling, layout and
precision choices when possible; disclose coupled changes such as replacing
vector reductions with Cube dots. Preserve packed boundaries, grouped-head
mapping, nonzero initial state, final-state outputs and input immutability.
Do not relax an error threshold to keep a candidate. Reject incorrect or
unbuildable variants separately from correct examples with an unfavorable cost.

## Read the evidence at the right boundary

A case has one independent CPU oracle. Each implementation is checked once,
then executes ten profiled forwards without tensor inspection, discovery calls,
autotuning or hidden warmups. Failed implementation attempts are retained
separately; they are not successful tradeoff examples.

The **device-operation span** is the last proven device task's end minus the
first task's start within a forward. It includes gaps between that forward's
kernels. The **instrumented caller event** is a separate NPU-event interval
around the call while the profiler is active. Neither is unprofiled application
latency. Device busy time and active union are also distinct from elapsed span.
Compare all ten samples and sample deviation at each boundary; do not discard
a slow first sample or turn a small noisy difference into a crossover rule.

Include preparation kernels, conversions, reductions, epilogues and
synchronization in the measured operation. Workspace allocation policy is
format-specific and explicit: a preallocated native workspace cannot support
an allocation-saving claim. Require numerical checks, successful export,
explicit per-iteration task ownership and expected kernel identities before
accepting a comparison. Timing availability, accounting and PMU availability
are separate states.

Preserve PMU column names, units and missing values. Pipe ratios do not measure
bytes, peak throughput or simultaneous overlap; do not sum them as disjoint
fractions. `cube_utilization(%)` is an exported field, not a certified fraction
of theoretical compute peak. Estimate a roofline distance only when operation
count, traffic, elapsed time and a target-supported capacity denominator are
available with explicit assumptions. Otherwise state **not estimated**.

The runnable interface is repository-root `scripts/optimization_pair.py`, with
an adapter, case JSON, output directory, oracle cache and provenance JSON.
`optimization_profile_evidence.py` binds timing/PMU rows to retained trace and
source identities. Follow the available `remote-access` and `ascend-profiling`
workflows for execution and collection; this library grants no extra launches.
