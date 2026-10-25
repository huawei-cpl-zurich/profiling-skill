# Shared A/beta in prepared WY W/U recomputation

This bounded component compares educational split W-then-U producers with the
source fused dual-output producer. Both required matrices, all eight heads,
all feature tiles, FP32 casts, beta products, W gating, dots and FP16 output
stores belong to every prepared API. No output or downstream-required step is
dropped. Shared source operands may reduce repeated loads/launches, while
coupled liveness can increase pressure; these are hypotheses, not traffic or
occupancy measurements.

Admission is explicit: B=1, one positive packed sequence with cu=[0,T],
H=Hg=8, K=V=128, BT=BK=BV=64. K/V/A/beta are physically dense FP16; beta and
FP32 cumulative g are already head-major. Every A row owns all64 columns,
including tail chunks. Common precomputed A and exact chunk-index metadata
are inputs. A inversion, prefix/gate accumulation, transposes/contiguous copies,
index construction and allocation/transfers are excluded from this component
in both arms; no complete GDN or public-wrapper latency claim is made.

The original source has unresolved dense-path undefined i_t, ignored B>1,
and empty-sequence index-helper behavior. Those paths, multiple sequences,
grouped heads, other K/V dimensions, non-dense layouts, bad metadata/dtypes
and aliases are rejected before backend/cache access. They are not silently
repaired. Open miss b26f6e86f6733351659ed05cee804a0b remains open. The upstream
NPU test covers this B1/H8/D128 single-varlen path with tails; it does not
establish standalone W/U correctness on the selected runtime.

The common extraction fixes IEEE policy before data. Original U
allow_tf32=False is retained; W receives the same explicit setting to preserve
the upstream test's TRITON_F32_DEFAULT=ieee intent. Both arms use four warps,
three stages and enable_fp_fusion=False. The original source and one-line W
precision patch are retained. IS_VARLEN=True is supplied directly, matching
the source cu-present heuristic on the admitted domain; dense behavior is
not changed. An auditable generator produces all three attributed bodies.
Actual compiled FP32 dot types and IEEE policy must be retained before any
scientific reference generation. An explicitly printed IEEE attribute or an
elided default with actual CompilerMetadata.default_dot_input_precision=ieee
qualifies; explicit contradictions, unknown defaults and non-FP32 dots reject. Neither TF32 advantage nor mixed dot policy
is permitted.

The independent CPU FP64 reference models FP32 beta products, FP32-rounded
ideal exp(g), the FP32-rounded second W product, FP64 matrix sums and final
FP16 casts. Both output matrices use fixed atol2e-4/rtol2e-3; exp approximation
and FP32 accumulation preclude an invented bitwise primary rule. Paired bits
and zero signs are secondary diagnostics. Lower-triangular precomputed A has
unit diagonal and small dyadic off-diagonals with a conservative infinity-norm
condition bound below1.29. K/V/beta and negative cumulative g are nontrivial.

Private outputs are poisoned once, then reused. Each output and its128B halos
is copied to CPU once per correctness check; both full matrices, all input
backing and aliases are checked. True memory accesses are proved independently
of padded false-mask pointer expressions; all integer/byte arithmetic remains
bounded. Compilation and launches use the prepared tensor device and restore
a caller's device selection. Compile prototypes reject execution even after
cached qualification, and all-component compilation publishes atomically.

The CLI is an exploratory single-case component run, not the controlled
all-case collector. Instrumented event/task times, active union, inter-kernel
gaps and complete operation spans are distinct metrics. No ordinary-latency,
physical-traffic, isolated-compute, occupancy or roofline claim is inferred.
Full per-file Apache/FLA MIT notices, complete Apache2/MIT text and complete
thirdparty notice are retained with the selected source/helper/caller/test.

The original inspection required a printed default and stopped after the first
accepted W binary, before any scientific reference or numerical API. Its raw
IR uses FP32 operands/output and elides the IEEE default; actual structured
compiler metadata explicitly declares IEEE. Prospective P05-v2 corrects only
that inspection representation before first math. The original failed gate,
artifacts and zero counts remain retained. Its accepted W program/cache is
reused; kernels, provider, cases, tolerances and flags are unchanged. Reported
shared/workspace metadata are retained without reinterpreting them as formal
physical UB, peak usage or occupancy proof. Any backend-required workspace
activity remains inside the complete JIT launch; explicit output/input setup
allocation is the excluded cost, and no allocator-saving claim is made.

The generic public capture hook now scopes event/profiler setup, forwards and
synchronous teardown/export to the prepared device, restoring the caller's
selection even on failure. This publication-only host fix is separately hashed
from the measured host interface. The controlled collector already supplied
its own device0 capture callback, so no kernel, provider, fixture or scientific
forward was changed or repeated. Its regression test observes setup, forward,
teardown and exception restoration under a mocked caller-device switch.

The complete fixed matrix passed twelve correctness checks before eighty
profiled APIs in eight captures. The following means are microseconds from
those same retained ten-sample captures; task sum equals active union within
rounding because no device-task overlap was observed.

| T / role | Task sum / active union before→after | Inter-kernel gap before→after | Device span before→after |
|---|---:|---:|---:|
|6 cost probe|21.5944→18.6264|110.5422→0|132.1366→18.6264|
|64 cost probe|23.9063→21.89245|117.13845→0|141.04475→21.89245|
|128 benefit probe|24.94045→22.1264|116.8805→0|141.82095→22.1264|
|2048 benefit probe|106.8542→106.0142|107.254→0|214.1082→106.0142|

Observed device tasks decrease from two to one per API. Span ratios are
primarily gap-driven; the largest case's task time is near flat. Task time
includes device movement/dispatch as well as arithmetic. These mean directions
do not establish ordinary-latency speedups or statistical significance. All
ten samples and available PMU rows remain retained without first-sample
removal. Only this explicit supported-path prepared component is qualified;
general source hazards and the open admission miss remain unresolved.
