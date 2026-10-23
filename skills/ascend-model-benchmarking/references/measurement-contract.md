# Measurement contract

## Primary profile-comparison protocol

For profile comparisons, including “profile ten”, use
`scripts/model_benchmark_profiled.py`. For each case:

1. The trusted provider generates inputs and the expected output with one
   reference forward. Save that private case before compiling the candidate.
2. Compile the candidate with fullgraph and validate its output exactly once.
   Correctness and compilation are outside the measured capture.
3. Capture exactly ten candidate forwards in one active profiler step and ten
   eager PyTorch baseline forwards in one separate capture. There are no extra
   warmup, discovery or unprofiled timing calls. Candidate capture comes first.
4. Reuse the private case and baseline capture for subsequent candidate
   configurations. Reuse invokes neither the provider nor the baseline; each new
   candidate still gets its own one correctness call and ten profiled forwards.

Expected outputs and candidate inputs use independent storage restored from the
CPU cache before correctness validation. Baseline inputs are independently
restored after candidate capture. All cache transfers/copies occur outside
profiling. Measured forwards assume read-only inputs: do not add tensor I/O
inspection, post-profile correctness checks or extra validation forwards during
or after the ten measured calls. Preparation and capture still check compiled
execution and reject active compilation. Cache identities bind stable private
baseline/provider/validator source files, exact case data and baseline results;
candidate code must live in a separate source file.

These measurements are **profiled device-event timings**, not warmed unprofiled
wall-clock latency. Retain all ten event samples in milliseconds and report their
arithmetic mean and sample standard deviation (`ddof=1`). The ten calls share one
capture; they are not ten independent full acquisitions or ten fresh processes.
Save and flush the timing evidence before completing the capture/export. A trace
export or normalization failure does not discard valid event timings. Report
export, accounting and kernel-identity status separately; timing completeness
alone does not establish successful profiling or verified submitted-kernel identity.

When trace accounting succeeds, retain each iteration's device and AiCPU task
sums and their combined all-kernel sum. These task-duration sums are separate
from elapsed device-event timings: overlapping tasks can make a sum exceed elapsed
time. Use explicit CPU/NPU correlations and accounted task iteration ownership,
not repeated names, fixed launch counts or time-window guesses. Expected kernel
names must match every candidate iteration; unknown selectors remain observed-only.

Compare matching case IDs and retain source/device/runtime/clock/load evidence.
A cached baseline from another configuration remains qualified until those
conditions are established. Report incomplete or user-truncated cohorts explicitly,
with per-arm sample counts and the common paired cohort for comparisons; do not
silently replace failures or omitted slower cases. Preserve every valid sample.

For explicitly requested warmed, unprofiled latency instead, use the separate
workflow below. Its twenty warmups, three batches and ten fresh-process repeats
must not be applied to a profile-comparison request.

## Explicit warmed unprofiled latency workflow

For explicitly requested warmed latency, use `scripts/model_benchmark_repeated.py`.
The following warmup, batch and fresh-process requirements apply only to that
workflow. Latency is unprofiled, synchronized full Model-forward wall time: synchronize,
read a monotonic clock, call the compiled Model, synchronize completion, read the
clock. Include allocations/host dispatch; exclude input generation/transfers,
correctness, compilation and 20 warmups. Three batches of 50 calls alternate
baseline/candidate order. Preserve all samples; report median, p10/p90, batch
medians and relative spread. Report empty synchronization overhead, never silently
subtract it. Warm means steady-state execution, not an assertion that all data
fits in cache. Inputs must remain immutable across paired calls.

`latency.execution_evidence[arm].compiled_executions` and `compile_count` are
preparation snapshots, not totals for the active measurement calls. The controller
checks active compiled execution and compilation separately. Cold preparation is
retained as `latency.execution_evidence[arm].cold_prepare_ms`; it includes Model
construction, compilation, first execution and synchronization, never warm latency.

Busy-time comes from a separate warmed replay of the same compiled callable and
inputs: 20 warmups and 50 explicitly correlated active iterations. Sum durations
of each iteration's device and AiCPU tasks for all-kernel busy-time; retain
device-only and AiCPU totals separately. Compute the median of each iteration's
sum, not the sum of the two medians. The primary busy-time speedup uses all-kernel
medians; a separate device-only speedup is labeled explicitly. Overlapping device
tasks can make this sum exceed elapsed time. Union-of-device-activity is an
additional diagnostic, not end-to-end latency. Keep AIC/AIV components distinct.

Acquisition must establish completed export and unambiguous CPU/NPU correlation.
Use actual task identities and iteration markers, never repeated kernel names or
an assumed fixed number of launches. Empty or ambiguous captures are failures.
The generic accounting helper accepts normalized records; it is not a vendor
trace parser and does not independently prove that export lost no records.

### Warmed acquisition filtering and comparisons

The reused CANNBot threshold is `duration > max(50 * per-name median, 10 us)`.
It flags suspicious task rows only. All primary totals retain every valid row,
including recurring casts or slow operations. Exclude warmups/host/unrelated work
only through explicit retained acquisition scope, not a duration heuristic.

Retain physical device identity, versions, requested and observed clocks/load,
capture order and exact measurement policy. Stable 800 MHz and stable 1800 MHz
are not comparable clocks. Missing telemetry qualifies results. This warmed-latency reproducibility
workflow uses ten complete paired repetitions, retaining noise rather than
requiring two stable runs. Greater-than-5% batch-median shifts are informational
flags to investigate, not an admission gate or permission to retry until favorable.

### Warmed between-repetition statistics

Use ten fresh child processes, each executing the entire correctness, compilation,
warmup, three latency batches and separate profiler replay protocol above. Hold
the same external physical-device selection/lease across children; preparation is
still excluded from every measured window. Providers must reproduce the same
cases/configuration in every child, not depend on state set in the parent process.

For each configuration/case, report the arithmetic mean and sample standard
deviation (`ddof=1`) across per-repetition latency medians (ms) and correlated
all-kernel busy-time medians (us); device-only and AiCPU statistics remain separate.
Also retain per-repetition paired speedup ratios and their mean/sample deviation.
Do not pool the 150 per-call latency samples or treat the three batches as three
repetitions. A single successful repetition has undefined sample deviation, not
zero. Retain all raw run directories and list requested, successful, failed and
unattempted counts. Fewer than ten successful repetitions is incomplete; optional
additional measurements are a separately identified series, never silent replacements.

Within-run acquisition failures do not contribute partial case results to the
complete paired series. Summaries of available successes are explicitly provisional
when the series is incomplete. Statistics do not establish comparable clocks/load;
retain per-run qualification and verify conditions across repetitions separately.

Source and fixture providers must be versioned separately from candidate code.
Correctness validation covers ordinary runtime input variation as well as shapes;
passing fixed fixtures is not permission to specialize on their accidental values.

## Eager reference for the warmed workflow

The warmed controller accepts `--baseline-execution eager|compiled` (default `compiled`).
Candidates always require compiled fullgraph execution and submitted-kernel identity.
Use `eager` explicitly for a slow PyTorch reference; unsupported compiled references
fail rather than falling back. Measure eager and compiled reference comparisons in
separate evidence directories and label their execution modes in final tables.
Eager references use eval/inference mode, correctness validation, the same 20-call
warmup, three batches of 50 synchronized full-forward samples, and a separate
50-iteration correlated profile with 20 warmups. They require no graph backend.
Active compilation detection and input mutation checks still apply to both arms.
The repeated controller forwards this option unchanged, records execution modes,
and rejects mixing modes across repetitions. Ten full repeats, raw samples, failure
counts, mean and sample standard deviation are unchanged.
