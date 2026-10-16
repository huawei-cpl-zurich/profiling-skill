# Natural-task profiling qualification

## Status and interpretation

All six tasks qualify for the prospective comparison: four as bounded
educational negatives and two as positive diagnostic lessons. All twelve
acquisition batches completed successfully. This is not a scored agent
comparison and does not establish that the profiling skill helps agents.
No natural-suite optimizer has launched. The earlier negative pilot is unchanged.

Before any scored launch, the user clarified: "I think we can keep negative
results as well if they teach us a lesson." Qualification therefore accepts
reproducible, code-linked educational negatives as well as positive effects.
An unexplained null or noisy timing does not establish equivalence. The skill
stays frozen at `294f76f`; there is no mandatory pipe acquisition or credit for
collection alone. The later paired reasoning/correctness rubric is unchanged.
This clarification was recorded before any scored launch; the preliminary
strict-gate assessment is retained in the task history, not rewritten as an
originally successful positive-contrast result.

## Acquisition

Each of six tasks has a baseline and two private scoped interventions. Each
variant passes four seeded numerical checks, unrestricted compute-launch
discovery, five independent BasicInfo timing captures and three PipeUtilization
captures. Independent batches use separate retained jobs and fresh output
folders. Devices are selected dynamically; device migration within this server
is not an exclusion criterion. All samples, failed attempts and diagnostics are
retained, including timing sets flagged above 5% range/median.

The completed evidence contains 36 variant/batch executions, 144 successful
seeded checks, 36 unrestricted compute discoveries, 180 timing samples and
108 pipe captures. Every acquisition has the expected single compute selector.
The source hashes match the generator's baseline and two scoped alternatives.
Independent batches are independent acquisitions, not independent recompilations:
the retained compiler artifacts for a given variant are identical across batches.

Timings below are `msprof op` BasicInfo task durations in microseconds, not
uninstrumented launch latency. Cube and vector rows are kept separate. Pipe
activity ratios are not capacity measurements or additive latency components.
Compiler intermediates describe lowering, not final-binary equivalence.

| Task | Family | Batch 1 baseline / intervention 1 / intervention 2 | Batch 2 |
|---|---|---:|---:|
| task01 | Gather, four nonlinear rounds | 2541.59 / 2565.15 / 2570.53 | 2567.93 / 2544.63 / 2543.55 |
| task02 | Gather, 32 nonlinear rounds | 624.37 / 625.89 / 626.53 | 614.33 / 615.95 / 612.25 |
| task03 | Masked rows, width 1024 | 94.08 / 94.58 / 94.02 | 98.04 / 87.28 / 96.42 |
| task04 | Masked rows, width 4096 | 25.30 / 28.86 / 23.60 | 24.30 / 28.40 / 26.10 |
| task05 | Tiled matmul, four epilogue rounds | 79.54 / 96.84 / 54.80 | 73.46 / 97.32 / 52.54 |
| task06 | Tiled matmul, 16 epilogue rounds | 57.90 / 68.04 / 50.02 | 59.28 / 66.94 / 52.46 |

Interventions are warps8 / recurrence unrolling for gather, predicated input
loads / warps8 for masked rows, and BM16 / BK64 for matmul. Only declared task
inputs and contracts will reach optimizer agents; interventions, this report
and qualification evidence remain outside their isolated workspaces.

[Retained sample data](data/natural-task-qualification.json) includes every
timing sample, spread/confidence flags, source/archive/compiler-artifact hashes,
correctness and discovery counts, and selected pipe summaries. Those summaries
preserve each exported core group, the row median within each capture and the
full row range; `NA` metrics are omitted, never zero-filled. Full CSVs and logs
remain in the compact retained archives. No acquisition result is pending.

## Final task decisions

| Task | Decision | Reproducible lesson | Timing limitation |
|---|---|---|---|
| task01 | Educational negative | Warp hint preserves retained lowering; recurrence unrolling changes vector activity but leaves scalarized gather | Small apparent gains/regressions reverse |
| task02 | Educational negative | Same gather structure persists; unrolling modestly lowers vector/stall activity in the longer recurrence | No robust overall gain; not equivalence |
| task03 | Educational negative | Source predication retains dense-copy-plus-select lowering and adds vector activity | Predicate's timing direction changes across batches |
| task04 | Educational negative | Wider regime repeats dense-copy-plus-select and extra vector activity; warp hint preserves early IR | Predicate medians worsen, but samples are noisy; warp timing direction reverses |
| task05 | Positive diagnostic | BK64 halves K-loop iterations, reduces Cube-side and reported wait activity, while vector arithmetic stays nearly fixed | Instrumented durations; several noisy sample sets |
| task06 | Positive diagnostic | Longer K regime repeats the tiling/activity direction with unchanged vector arithmetic | Smaller timing effect and heterogeneous per-core work |

## Code-to-profiler lessons

### Gather: distinguish the changed work from the unchanged access path

Task01's warp and unrolling timing differences reverse direction between
batches: roughly +0.93% then -0.91%, and +1.14% then -0.95%. The baseline
itself shifts about 1%. Neither intervention demonstrates a robust gain or
regression in these acquisitions; this is not proof of exact equivalence.

Task01 captures retain overlapping ratios: scalar approximately 0.968-0.971,
vector 0.022-0.025, MTE2 0.017-0.019. TTAdapter shows the indirect gather
lowered to a 256-iteration loop with per-element `memref.load`; unrolling the
nonlinear recurrence leaves that gather intact. This supports an unchanged
scalarized-gather hypothesis. It does not prove scalar capacity saturation,
exclude memory latency, or isolate a physical latency contribution.

Activity is not completely unchanged. Across all six captures per variant,
task01's median vector time is 57.959 us for baseline/warp8 versus 56.542 us
for unrolling. Task02's baseline vector medians are 92.110-92.113 us versus
91.353 us for unrolling; its scalar-vector-stall medians fall from
67.769-68.717 us to 65.169-66.159 us. The compiler expands four or 32
recurrence stages, while retaining the gather loop. These are repeatable
diagnostic changes, not microseconds that can be subtracted from task latency.

Task02 independently repeats the lesson: static unrolling changes its timing
median by +0.346% then -0.339%; warp8 changes it by about +0.25%, smaller than
the observed sample dispersion. Greater vector activity than task01 is
consistent with more nonlinear rounds, but different shapes/table sizes prevent
that cross-regime observation from isolating recurrence cost.

Baseline versus warp8 TTIR and TTAdapter bodies are identical after removing
only location metadata in both tasks and both batches. This does not prove
binary identity or that the compiler universally ignores the hint.

### Masked loads: source predicates are not evidence of sparse transfers

Task03's predicated-load variant retains a full `memref.copy` followed by
`arith.select` in TTAdapter, in both acquisitions. Task04 repeats
the corresponding full-width copy at width 4096. The source-level load mask
does not establish that physical memory transfers became sparse.

Task03's higher vector activity and lower MTE3 activity repeat, but its timing
response does not: near-equal medians in batch 1 versus a lower predicated-load
median in batch 2. Do not turn either observation into "masking never helps" or
"the mask saved HBM traffic." A ratio change alone does not measure bytes, and
intermediate IR is not final-binary proof. This is a useful calibration lesson,
not an isolated causal explanation of the timing difference.

The added vector response repeats in both regimes. Across six captures,
task03's median vector time is 28.482-28.603 us for baseline versus 30.945 us
with predication. Task04's is 7.364-7.367 us versus 7.827 us. Gain loading,
both exponentials and both full-width reductions remain in retained lowering.
The masked variant's MTE2 activity does not collapse proportionally to the
source mask density. This corroborates persistent execution, not a physical
byte count or a proof of which pipe determines latency.

Task04's predicated median is higher in both batches, but those timing sets
are noisy. Its warp8 timing advantage reverses despite retained early IR and
nearly unchanged vector activity. Keep both results; neither justifies a
precise latency penalty, equivalence, or a universal claim about masking.

Baseline and warps8 TTAdapter module bodies are identical after removing only
location annotations in both tasks and both batches. That nominal launch-hint
change does not change these retained intermediate
operations. It does not prove identical final binaries or globally establish
that the compiler ignores the hint.

### Matmul: tile changes have distinguishable cost and resource signatures

For task05, BK64 is faster and BM16 slower in both acquisitions, with disjoint
within-batch timing ranges despite several noisy sample sets. BK64 changes
TTIR's K loop from eight iterations of K32 to four of K64 while preserving
the output tile and fused epilogue. BM16 changes the output tile and program
count instead; both changes are substantive, not removing required work.

Task06 repeats the direction in a different regime: BK64 changes 64 K32 loop
iterations to 32 K64 iterations, preserving the sixteen-round fused epilogue.
BM16 instead retains the reduction loop and changes the output tile/program
count. Its median is slower in both batches; BK64 is faster in both.

The following are ranges of within-capture core-group medians across six
captures, in microseconds. They are separate exported observations, not
additive phases, latency savings or whole-device utilization.

| Task/variant | Cube0 MTE2 | Cube0 scalar-MTE1 stall | Vector0 arithmetic | Vector0 wait_id4 |
|---|---:|---:|---:|---:|
| task05 baseline | 54.482-64.356 | 49.121-58.979 | 21.785 | 47.967-53.743 |
| task05 BM16 | 69.243-73.309 | 65.767-70.517 | 32.703 | 55.277-67.866 |
| task05 BK64 | 29.966-33.959 | 27.672-32.533 | 21.745-21.773 | 22.938-29.743 |
| task06 baseline | 21.836-24.054 | 19.617-21.659 | 3.569 | 24.321-26.432 |
| task06 BM16 | 43.983-46.149 | 43.087-44.206 | 7.862 | 41.334-43.843 |
| task06 BK64 | 15.213-17.142 | 13.853-15.637 | 3.569 | 18.301-20.332 |

Cube-side activity and reported vector waiting decrease with BK64, while
vector arithmetic time stays nearly fixed. A higher vector *ratio* therefore
does not establish more vector work, vector saturation or a vector bottleneck.
The data also retains vector1 separately; the table does not merge AIV groups.
Do not assign `wait_id4` to a particular synchronization dependency without
additional mapping evidence.

Task06's broad per-core ranges overlap between variants; cores need not have
equal work. Its group medians and repeated timing direction support the bounded
diagnostic lesson, not uniform per-core behavior or an inferential confidence
interval. Neither task's diagnostic timing replaces minimally instrumented
performance measurement.

The educational contrast includes BM16: higher vector activity does not by
itself mean a faster kernel. Code, amount of work, timings and execution
structure must be interpreted together.

## Evidence locations and limits

Archives are named `evidence-taskNN-batchB.tar.gz` in retained controller
storage; machine-specific locations and durable handles remain in the central
task record. Within each archive:

- `batch-B/summary.json` and `batch-B/variant-N/result.json`: capture results.
- `batch-B/variant-N/basic-I/report/OPPROF_*/OpBasicInfo.csv`: exact timing row.
- `batch-B/variant-N/pipe-I/report/OPPROF_*/PipeUtilization.csv`: all core rows.
- `batch-B/variant-N/correctness/`: raw seeded checks, commands and exit status.
- `variant-N/compiler-evidence/candidate/{ttir,ttadapter}.txt`: retained IR.

Useful compiler pointers, with paths relative to each variant's compiler folder:

- Gather baseline/warp: `ttadapter.txt:33` starts the scalar gather loop;
  static variants retain it at line 30. Baseline recurrence is at line 45.
- Masked baseline: `ttadapter.txt:49` is the dense x copy; predicated variants
  retain it at line 50 and select at line 52, for both widths and both batches.
- Task05 `ttir.txt`: baseline loop/dot lines 44/64, BM16 lines 48/68,
  BK64 lines 45/65. Task06 corresponding lines 43/63, 46/66, 45/65.

These are compiler-stage citations, not instruction timelines or final-binary
equivalence. No A5 capability, traffic-byte multiplier or capacity ceiling is
inferred from them. Only supported A3 captures were used.

## Operational evidence and readiness

The initial acquisition attempt failed before any kernel because a bounded
Popen-compatible helper was incorrectly called with `capture_output`. The fix
and regression are retained; it is a harness bug, not a device failure. Eight
qualification tests passed natively on A3. The final assembled local suite
passed 1925 tests and 15 subtests before the documentation-only clarification.

`msprof --version` is unsupported here (exit 255); actual `msprof op` captures
succeed. Its diagnostic is retained rather than inventing a profiler version.
Observer interruptions are recovered through the original retained handles;
they do not authorize a duplicate workload or classify a kernel as failed.
Compact per-task artifacts are about 184-284 KB for the first completed batches,
including logs, CSV/JSON and compiler text, without vendor binary capture trees.
The interrupted task02/task04/task06 second batches subsequently completed
with exit 0 and were retrieved through their original handles. No duplicate
workloads were dispatched and no noisy samples were removed.

Qualification decisions and their evidence are now frozen in this report and
the accompanying data. Three independent family audits checked the raw CSVs,
correctness logs and compiler artifacts; all six decisions passed the amended
gate. No replacement tasks or skill retuning were used; no pull request was
merged and no skill was installed.

The next step is the twelve-agent comparison described in
[the natural-use protocol](natural-profiling-evaluation-protocol.md): same
ordinary prompt/model/tools, frozen skill `294f76f` versus no skill, 1200 seconds
including waiting, a 120-second finishing reserve, at most three concurrent
agents in four balanced waves, using the validated prerequisite assembly.
Keep diagnostics and these private interventions
outside both isolated agent workspaces. No agent launch is part of this report
finalization.

Readiness is not proof of necessity or usefulness. Timing-only tuning could
select BK64; compiler inspection alone could reveal dense copies or scalarized
gather. Pipe evidence adds observed activity and constraints on explanations.
The paired comparison must test that incremental value rather than assume it.
Retain ties, failed edits and lack of skill benefit alongside wins; score actual
evidence use and calibration, not collecting metrics or citing this report.
