# W004 W12: guarded forty-step Jacobi

This independent FP32 example replaces a serial single-RHS Thomas solve with a
GPU certificate and forty simultaneous Jacobi updates from zero. It passes the
fixed forward-accuracy gate on all six frozen cases. Avoid this implementation
for the four measured Ascend910B4-1 workloads: its forty launch boundaries cost
more than the serial work they remove.

| Case | m | Before mean span (µs) | After mean span (µs) |
|---|---:|---:|---:|
| Gallery benefit hypothesis | 2048 | 1071.284 | 6764.956 |
| Gallery cost hypothesis | 128 | 259.518 | 7148.896 |
| Heldout benefit hypothesis | 1536 | 776.850 | 7380.748 |
| Heldout cost hypothesis | 256 | 313.076 | 7208.280 |

These are full profiled device-operation spans from first to last device work,
including resets, certificate, forty passes, conditional fallback dispatch and
inter-stage gaps. They are not unprofiled application latency. Active duration
and gaps are retained separately in the [compact data](data/optimization-w004-guarded-jacobi.json).
PMU fields describe the exported per-task measurements; they establish neither
physical traffic nor a whole-device roofline. The four benefit/cost labels are
prospective hypotheses, preserved despite the negative results.

The [frozen matrix](../examples/optimization-library/cases/wave004/W12.json)
SHA256 is `85feb98c0fbafaf152a8bfe0135033244956927a0f50de5e0bec1af80272aefd`.
Inputs are contiguous FP32 `dl,d,du[m]` and one RHS stored as `[1,ldb]`, equivalent
to column-major logical `[m,1]`. The original coefficients/RHS remain immutable;
output has separate storage and preserves padding bitwise. Frozen coefficients
use signed uniform off-diagonals in ±.25, `d=2+abs(dl)+abs(du)`, and RHS in ±.125.
The supported solver scope requires a no-pivot-safe Thomas system; this example
provides no pivoting or extended-exponent fallback guarantee.

The after API checks the current GPU coefficients and active RHS on every call.
Its certificate requires `128<=m<=6144`, finite values, strict
`abs(d)>2*(abs(dl)+abs(du))`, diagonal magnitudes in `[2^-24,2^24]`, RHS magnitude
at most `2^24`, `maxabs(b)/minabs(d)<=4`, and conservative finite-sum bounds.
GPU conditionals select forty old/new ping-pong updates or the unchanged parent
Thomas arithmetic. No input-dependent certificate, factors or results are
cached between calls; no host certificate synchronization is needed. All
necessary resets/copies and launches remain inside the operation. Both arms
compile with `num_stages=1,multibuffer=False,enable_fp_fusion=False` and use the
same supported FP32 division operator.

The contraction certificate bounds exact-arithmetic truncation by `2^-40`;
this is not a proof of FP32 roundoff. CPU FP64 dense solves of the represented
FP32 inputs independently gate every output with `atol=1e-5,rtol=1e-4`. The data
retains maximum forward error, relative L2 error, FP64 residual infinity norm
and componentwise backward error. The boundary-impulse guard has backward
error 1.0 in both arms in tiny/underflow rows while its maximum forward errors
remain about `1.94e-9` and `3.42e-9`. This limitation is retained; residual alone
never establishes correctness. The strict-twofold-dominance guard stays safe
for Thomas. Five separately counted certificate-only NPU component tests verify
that guard's false bit, the impulse guard's true bit, and rejection of current
RHS magnitude, diagonal range and nonfinite-coefficient changes.

Run the public pair CLI with a nested committed case:

```bash
PYTHONPATH=examples/optimization-library/triton python scripts/optimization_pair.py \
  --adapter guarded_jacobi:factory \
  --case examples/optimization-library/cases/wave004/W004-W12-01.json \
  --output OUTPUT --cache CACHE --provenance PROVENANCE_JSON
```

The retained A3 run used physical die 4 mapped to logical 0 after fresh healthy,
idle checks and a device lease. It compiled every resource in all twelve arms
before any full API invocation, generated six CPU oracle caches once, checked
twelve arm/case outputs once, and captured ten complete calls per arm for four
pairs (80 calls, eight captures). The two guards made no captures. There were no
warmups, autotuning, extra timing runs or output validation inside profiling.
Standalone certificate tests add five component calls, zero full APIs/captures.

Provenance is **SOURCE-DOCUMENTED**, not a transcription of upstream device code.
The source proposal pins ops-sparse `bfc7c19d83ce246b03c65dc916ed4753e5467cf3`
under CANN Open Software License Agreement 2.0. Algorithm documentation is
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/gtsv2/README.md:L26-L34`;
architecture layout is `#sparse/gtsv2/arch22/gtsv2_tiling_data.h:L18-L42`, numerical
tests `#test/gtsv2/arch22/gtsv2_test.cpp:L368-L414`, and public host launch boundary
`#sparse/gtsv2/arch22/gtsv2_host.cpp:L143-L188`. The Jacobi device body was unseen.
Source-path miss `11501d136a88a71c6d38ce1feaa9647e` and runtime-contract miss
`58c446d63c6a796ae5f435fb458b8668` remain open. This does not reproduce native
extended-range behavior or establish a universal dispatch threshold.

An initial collector failed before any API because its generic output name
collided with a previous experiment. A read-only audit found one collector and
its staging archive replaced, with all 84 original kernel/helper files intact.
The exact two originals were restored from a verified original archive, all 85
source members and the archive were checked, and the accidental versions were
preserved separately. Unique W12 paths hold the successful run. Full vendor
reports remain retained remotely; the compact archive binds their inventory,
source/cache binaries, sample traces, counters, operational failure and tests.
