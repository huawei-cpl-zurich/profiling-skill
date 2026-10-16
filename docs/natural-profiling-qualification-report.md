# Natural-task profiling qualification

## Status and interpretation

Qualification is in progress. This is not a scored agent comparison and does
not establish that the profiling skill helps agents. No natural-suite optimizer
has launched. The earlier negative optimization pilot remains unchanged.

Before any scored launch, the user clarified: "I think we can keep negative
results as well if they teach us a lesson." Qualification therefore accepts
reproducible, code-linked educational negatives as well as positive effects.
An unexplained null or noisy timing does not establish equivalence. The skill
stays frozen at `294f76f`; there is no mandatory pipe acquisition or credit for
collection alone. The later paired reasoning/correctness rubric is unchanged.

## Acquisition

Each of six tasks has a baseline and two private scoped interventions. Each
variant passes four seeded numerical checks, unrestricted compute-launch
discovery, five independent BasicInfo timing captures and three PipeUtilization
captures. Independent batches use separate retained jobs and fresh output
folders. Devices are selected dynamically; device migration within this server
is not an exclusion criterion. All samples, failed attempts and diagnostics are
retained, including timing sets flagged above 5% range/median.

Timings below are `msprof op` BasicInfo task durations in microseconds, not
uninstrumented launch latency. Cube and vector rows are kept separate. Pipe
activity ratios are not capacity measurements or additive latency components.
Compiler intermediates describe lowering, not final-binary equivalence.

| Task | Family | Batch 1 baseline / intervention 1 / intervention 2 | Batch 2 |
|---|---|---:|---:|
| task01 | Gather, four nonlinear rounds | 2541.59 / 2565.15 / 2570.53 | 2567.93 / 2544.63 / 2543.55 |
| task02 | Gather, 32 nonlinear rounds | 624.37 / 625.89 / 626.53 | Pending |
| task03 | Masked rows, width 1024 | 94.08 / 94.58 / 94.02 | 98.04 / 87.28 / 96.42 |
| task04 | Masked rows, width 4096 | 25.30 / 28.86 / 23.60 | Pending |
| task05 | Tiled matmul, four epilogue rounds | 79.54 / 96.84 / 54.80 | 73.46 / 97.32 / 52.54 |
| task06 | Tiled matmul, 16 epilogue rounds | 57.90 / 68.04 / 50.02 | Pending |

Interventions are warps8 / recurrence unrolling for gather, predicated input
loads / warps8 for masked rows, and BM16 / BK64 for matmul. Only declared task
inputs and contracts will reach optimizer agents; interventions, this report
and qualification evidence remain outside their isolated workspaces.

[Retained sample data](data/natural-task-qualification.json) includes every
downloaded timing sample, spread/confidence flags, source hashes, correctness
counts and pipe-row counts. Missing second-batch results are explicitly pending,
not silently dropped, zero-filled, or classified as failed kernels.

## Lessons supported so far

### Gather: unchanged relevant lowering is an informative negative

Task01's warp and unrolling timing differences reverse direction between
batches: roughly +0.93% then -0.91%, and +1.14% then -0.95%. The baseline
itself shifts about 1%. Neither intervention demonstrates a robust gain or
regression in these acquisitions; this is not proof of exact equivalence.

All captures retain overlapping activity: scalar approximately 0.968-0.971,
vector 0.022-0.025, MTE2 0.017-0.019. TTAdapter shows the indirect gather
lowered to a 256-iteration loop with per-element `memref.load`; unrolling the
nonlinear recurrence leaves that gather intact. This supports an unchanged
scalarized-gather hypothesis. It does not prove scalar capacity saturation,
exclude memory latency, or isolate a physical latency contribution.

Task02's first batch has greater vector activity, consistent with 32 rather
than four nonlinear rounds. That cross-regime observation cannot substitute
for its own independent intervention comparison.

### Masked loads: source predicates are not evidence of sparse transfers

Task03's predicated-load variant retains a full `memref.copy` followed by
`arith.select` in TTAdapter, in both acquisitions. Task04's first batch shows
the corresponding full-width copy at width 4096. The source-level load mask
does not establish that physical memory transfers became sparse.

Task03's higher vector activity and lower MTE3 activity repeat, but its timing
response does not: near-equal medians in batch 1 versus a lower predicated-load
median in batch 2. Do not turn either observation into "masking never helps" or
"the mask saved HBM traffic." A ratio change alone does not measure bytes, and
intermediate IR is not final-binary proof. This is a useful calibration lesson,
not an isolated causal explanation of the timing difference.

Baseline and warps8 TTAdapter module bodies are identical after removing only
location annotations in both task03 batches and task04's first batch. That
nominal launch-hint change does not change these retained intermediate
operations. It does not prove identical final binaries or globally establish
that the compiler ignores the hint.

### Matmul: tile changes have distinguishable cost and resource signatures

For task05, BK64 is faster and BM16 slower in both acquisitions, with disjoint
within-batch timing ranges despite several noisy sample sets. BK64 changes
TTIR's K loop from eight iterations of K32 to four of K64 while preserving
the output tile and fused epilogue. BM16 changes the output tile and program
count instead; both changes are substantive, not removing required work.

BK64 repeatedly has lower Cube MTE2 activity, higher vector activity and lower
vector `wait_id4` time. Batch2's per-capture Cube MTE2 means are approximately
0.807-0.850 baseline versus0.663-0.776 BK64; vector0 activity0.325-0.330 versus
0.480-0.503; vector0 wait48-52us versus27-30us. Keep these separately labeled
observations: they are not a sum of saved microseconds, proof of saturation,
or proof of a particular synchronization dependency.

The educational contrast includes BM16: higher vector activity does not by
itself mean a faster kernel. Code, amount of work, timings and execution
structure must be interpreted together.

## Operational evidence and remaining gate

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

Finish independent batches for task02/task04/task06, retain exact samples and
code-linked interpretations, and freeze the qualification decisions before
starting the twelve-agent comparison. Positive and negative optimization
outcomes must both remain visible in that later report. Neither these
characterizations nor successful acquisition alone establish skill usefulness.
