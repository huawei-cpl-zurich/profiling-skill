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

## Filtering and comparisons

The reused CANNBot threshold is `duration > max(50 * per-name median, 10 us)`.
It flags suspicious task rows only. All primary totals retain every valid row,
including recurring casts or slow operations. Exclude warmups/host/unrelated work
only through explicit retained acquisition scope, not a duration heuristic.

Retain physical device identity, versions, requested and observed clocks/load,
capture order and exact measurement policy. Stable 800 MHz and stable 1800 MHz
are not comparable clocks. Missing telemetry qualifies results. A reproducibility
smoke requires two consecutive complete matmul runs, with medium/large shapes
showing no unexplained greater-than-5% batch-median shift. Diagnose failures;
do not retry until a favorable sample appears.

Source and fixture providers must be versioned separately from candidate code.
Correctness validation covers ordinary runtime input variation as well as shapes;
passing fixed fixtures is not permission to specialize on their accidental values.
