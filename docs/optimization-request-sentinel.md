# Request-scalar sentinel replacement before token expansion

This prepared Triton comparison uses the pinned source kernel's replacement of
one scalar per request before its masked32-lane broadcast. The educational
baseline expands the unchanged request scalar, then launches a separate
in-place token replacement kernel. Both baseline kernels belong to every API
call. This is a known fusion/hoisting technique applied to this operator; the
baseline is an ablation, not an upstream-provided implementation.

The bounded component accepts owned CPU int32 request values and cumulative
endpoints for preparation. It verifies nonnegative monotone endpoints,
intervals0..32, exact nonempty final token extent, layouts, scalar int32 bounds
and no aliases before backend/cache access or device transfers. Empty batches,
all-empty output and trailing unwritten output are rejected. These stronger
checks are the study's domain; the upstream allocating wrapper does not perform
them. Its exact source, MIT license and complete thirdparty notice are retained.

Both arms reuse private preallocated output and prepared immutable inputs. CPU
prefix construction, input transfer, output allocation and independent reference
generation are outside the forward. Output is poisoned once before correctness,
then every API must completely overwrite it.32-int32 halos surround each input
and output, with aligned128B logical pointers and actual ND storage audits.
A common device context keeps compilation/launches on the prepared tensor's
device and preserves a caller's changed device selection. No global native
launch registry or separate prepared-arm close is involved.

The prospective matrix has four main pairs and two guards. The65,536-request
case deliberately has only512 output tokens: request-scalar replacement still
runs in empty requests while the baseline token pass touches few values. Both
sentinel and nonmatching values occur in empty and active requests. The guards
cover mixed0/1/31/32 lengths and exact int32-extrema replacement. The independent
CPU reference uses Python integer list expansion/replacement; all outputs,
whole input backing, halos and paired output bits are checked once. Cases,
configuration and replacement density are fixed before compilation or data.

The CLI is exploratory single-case execution, not the controlled all-case
qualification collector. Scientific outcomes and any profiler diagnostics are
recorded separately. Profiled event durations and device-operation spans include
instrumentation and omit allocation/setup; neither is ordinary end-to-end
latency. Missing metrics are unavailable rather than zero; no HBM traffic,
occupancy or roofline claim is inferred from logical work or ELF differences.

The bounded A3 run passed all12 correctness checks before80 profiled APIs in
eight captures; guards had no profiles. Raw traces show two kernels per baseline
API and one per candidate API. Four mean diagnostic device spans (microseconds)
were79.2234→1.76405,77.3836→5.1901,90.64765→21.9524, and223.1682→219.4925.
The large small-case span ratios include the baseline's inter-kernel host-enqueue
gap under profiling. The sparse-request cost probe changed little; these are
not ordinary-latency speedups. All ten samples, their variation, actual task
counts and available PMU rows remain in `P13-profile-details.json.gz`.

`P13-result.json` binds the prospective matrix, exact measured/current sources,
all compile artifacts, six once-cached fixture payloads, raw reports, timeline
checks and device release proof. CPU-inferred empty/matching request counts
are labeled separately from observed kernel counts; no internal dynamic branch
coverage or machine comparison-count claim is made.

All values below are means in microseconds from the same retained captures.
Task-time sum equals active union here because tasks do not overlap. The gap
column measures idle time between the operation's first and last device task;
it excludes time before the first enqueue and after the last task. Task time
also includes device movement/dispatch and is not isolated arithmetic time.

| Case | Task sum / active union, before→after | Inter-kernel gap, before→after | Device span, before→after |
|---|---:|---:|---:|
|01 small cost probe|3.11005→1.76405|76.11335→0|79.2234→1.76405|
|02 benefit probe|7.25805→5.19010|70.12555→0|77.3836→5.19010|
|03 benefit probe|29.03480→21.95240|61.61285→0|90.64765→21.95240|
|04 sparse cost probe|223.14040→219.49250|0.02780→0|223.1682→219.49250|

The result is fusion from two device kernel tasks to one per API and the
associated instrumented spans. The large small-case span ratios largely
reflect disappearance of an inter-kernel enqueue gap; task-time reductions
are substantially smaller. The sparse cost probe is near flat. Mean
directions alone establish neither statistical significance nor ordinary
latency improvements.

Address proof distinguishes pointer arithmetic from memory accesses. The
expansion's `end+31` pointer expressions fit the32-element output halo. The
baseline BLOCK256 token kernel can form false-mask pointers beyond that halo;
its strict `idx<N` mask keeps actual accesses inside the output, and padded
indices/byte offsets remain signed32-safe under the bounded domain. The
prospective end+31 wording describes the expansion rather than every masked
token-kernel pointer. No block size, halo, source, fixture or measurement was
changed after acquisition. The independent CPU reference applies its scalar
conditional before list repetition, which is algebraically identical to
elementwise replacement after repetition for exact int32 values.
