# Measurement contract

## Two metrics, two acquisitions

Latency is unprofiled, synchronized full Model-forward wall time: synchronize,
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

The torch-npu 2.10 trace adapter recognizes the retained zero- or 0.02-microsecond-duration
`PLACE_HOLDER_SQE` schema only when its name, sentinel model/subtask/connection
IDs, task ID 0, positive real batch ID and matching stream/thread coordinates agree. It
retains the complete original record in the administrative exclusion audit;
that record contributes no compute time. A different shape or a flow targeting
this placeholder remains invalid. Unknown task types, missing iteration markers
and incomplete CPU/NPU links still fail closed.

## Filtering and comparisons

The reused CANNBot threshold is `duration > max(50 * per-name median, 10 us)`.
It flags suspicious task rows only. All primary totals retain every valid row,
including recurring casts or slow operations. Exclude warmups/host/unrelated work
only through explicit retained acquisition scope, not a duration heuristic.

Retain physical device identity, versions, requested and observed clocks/load,
capture order and exact measurement policy. Stable 800 MHz and stable 1800 MHz
are not comparable clocks. Missing telemetry qualifies results. A reproducibility
measurement uses ten complete paired repetitions, retaining noise rather than
requiring two stable runs. Greater-than-5% batch-median shifts are informational
flags to investigate, not an admission gate or permission to retry until favorable.

## Between-repetition statistics

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

Retained GDN `MIX_AIV` ReduceSum events have verified `torch_to_npu` start/finish
links to CPU `aclnnReduceSum` records. They count as device compute under the same
complete-flow and explicit-iteration requirements as other validated task types.
This recognition does not recover missing markers or establish vector/cube
utilization from the task label.

Timing-event commands exported as `EVENT_RECORD` are administrative, rather
than compute, only for the observed 0 or 0.02 µs schema: matching name, absent
category, sentinel model/subtask IDs, real batch/task/connection IDs and
matching stream/thread coordinates. Every command still requires a unique,
complete `torch_to_npu` flow from an exact CPU `record_event` operation. The
exclusion audit retains its original device record, host operation and link.
These flows count in `paired_flows`, while `observed_tasks` counts compute
records. Timing events cannot satisfy an iteration's compute-work requirement.

## Explicit eager reference

The controller accepts `--baseline-execution eager|compiled` (default `compiled`).
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
