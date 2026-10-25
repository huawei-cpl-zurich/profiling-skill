# Prepared Compressor output zeroing pair

`compressor_zero:factory` compares full prepared Compressor calls. Both arms
project BF16 inputs with the same source dual-projection strategy for at most
64 tokens and separate projections otherwise, using common BM32/BN64/BK128/GROUP8
configuration. The source pooling and separate cyclic-cache update remain
unchanged. The only candidate change omits `output.zero_()` when every returned
row is proved to be written. This is a bounded source-backed experiment, not a
claim about vendor autotuner throughput or a new unique optimization mechanism.

The supported domain is contiguous BF16 X with hidden size 5120, projection
weights [512,5120], BF16 norm weights [512], and private FP32 cache [B,16,1024].
B is 1–4, ratio is 2/4/8 and ratio < S <= 2048. Uniform used lengths must exceed
the ratio and not exceed S; start positions are 0 through ratio-1. Distinct
cache-block mappings are required. Packed input is supported for B=1. Invalid
host metadata and CPU fixture shape/dtype/layout/alias are rejected before
backend import or fixture caching. This bounded API does not implement all
upstream Compressor shapes or metadata variants.

Zero omission requires uniform BSND, used_len=S, aligned start and S divisible
by ratio. With no residual, source task0 takes the clean path: task t owns group
range [t*GROUPS_TILE,(t+1)*GROUPS_TILE), masked at S/ratio. These disjoint ranges
cover every returned row and all 512 columns exactly once. Batch output bases
are disjoint. Packed, partial and residual cases retain source zeroing every
call. The source packed allocation can have an unwritten extra row, which must
remain zero. Projection outputs, all cache slots and the entire returned output
including padding are validated.

Inputs and metadata upload, private workspace/output allocation and initial
cache copies occur in preparation; storage is reused across forwards. Allocator
latency and repeated metadata upload are excluded in both arms. Output interiors
are poisoned once in preparation, with output/cache canaries. The baseline and
fallback candidate call Torch `Tensor.zero_()` on every API invocation. The
complete API includes zeroing where required, both projections, pooling and cache
writes. Profiles inspect no outputs and perform no cache reset or validation.

The CPU provider independently computes integer dot products from bounded q/128
inputs. Every partial integer sum is below 2^24, so the FP32 projection is exact.
It computes stable FP64 pooling, rounds the pooled value to BF16 before RMS,
uses the full source caller epsilon 1e-20, and rounds the result to BF16.
Output atol=0.03125/rtol=0.015625 is fixed before collection; projections and the
whole cache require exact FP32 bits, and paired outputs require exact BF16 bits.
The frozen v2 matrix is [T11.json](../../../experiments/optimization-wave008/T11.json).
The original wrong-epsilon matrix was never executed.

The single-case CLI is exploratory; it does not claim controlled qualification:

```bash
python examples/optimization-library/triton/compressor_zero.py \
  --case case.json --cache oracle-cache --output run-evidence \
  --provenance provenance.json --correctness-only
```

A controlled study requires all 12 input-only compile/resource/ABI screens before
six once-cached CPU fixtures, all 12 correctness gates before any profiling, then
exactly ten complete API forwards per arm on four main pairs. Guards have zero
profiles. Compile prototypes reject execution, and public run performs atomic,
cached compilation before its first forward.

Torch zeroing is external to Triton. Identical own JIT binaries across arms are
expected and do not prove this host API change is a no-op. The controller binds
installed runtime libraries and retains the actual zero selector, task type and
per-iteration device-operation count. Whole-operation spans include every linked
zero operation even when it is a non-kernel command; no standalone zero ELF or
PMU counter is invented. The shared normalizer supports known kernel task types;
if zeroing exports MEMSET/MEMSET_ASYNC, controlled collection registers those
exact types as device work and binds that operational registration separately.
The generic CLI requires compatible accounting for its actual runtime schema.
Unknown types fail closed. Codec repairs use retained raw captures without
repeating completed forwards. Profiled device spans are diagnostic and exclude
first enqueue/host overhead; they are not ordinary application latency.

Selected vendor bodies preserve the pinned source at
`7acb9c40a6ca827e885d902b7eaee5d45457b230`; only the projection autotune decorator
is omitted for the common fixed configuration. See
[license and notices](compressor_zero_NOTICE.txt). No cache epilogue fusion,
projection strategy optimization or ratio2 pooling optimization is combined.

## Retained bounded A3 result

The single controlled collection completed all 12 compile screens before six
once-cached CPU payloads, then all 12 numerical gates before 80 profiled API
calls in eight captures. Both guards passed with zero profiling. All paired
outputs matched bitwise; projection/cache bits, unwritten rows, input immutability
and canaries passed. Five functional host tests also passed on the selected A3
runtime. No warmup, separate timing run, scientific replay or profiling output
inspection was performed.

The actual zero selector was `aclnnInplaceZero_ZerosLikeAiCore_ZerosLike`, task
type `AI_VECTOR_CORE`, already supported by the normalizer. Every baseline
forward contained one zero task; every main candidate contained none. Device
operations per call changed 5→4 for cases 01/03/04 and 4→3 for case 02. No MEMSET
registration capability was exercised. Own JIT chains were byte-identical across
arms, as expected for this host zero-call omission.

| Case | Before span µs | After span µs | Before/after |
|---|---:|---:|---:|
| 01 | 382.890 | 285.985 | 1.339 |
| 02 | 334.145 | 208.398 | 1.603 |
| 03 | 603.690 | 537.276 | 1.124 |
| 04 | 416.274 | 283.140 | 1.470 |

These are means of ten profiler-instrumented full device-operation spans, with
all required zero tasks and inter-task gaps included. The diagnostic ratio
geomean is 1.372. First-sample variability and all ten raw samples are retained;
these figures do not establish ordinary application latency or allocator gains.
The [portable result](../../../experiments/optimization-wave008/T11-result.json)
includes raw PMU columns and identity-bound rows, source/compiler/binary/cache
bindings, actual operation counts and receipt limitations. All 431 compact-file
hashes were verified, and raw trace accounting, spans and PMU joins were
independently reproduced locally. Coverage matched the separate inventory.
The selected device was released with no live owned processes, healthy idle
status, empty device processes and zero ECC errors; no reset was used.

The subsequent [host compatibility receipt](../../../experiments/optimization-wave008/T11-publication-host.json)
separately binds the published wrapper and tests. Projection buffers now explicitly
use FP32 regardless of the caller's global dtype, and CPU diagnostic quantile
probabilities match the error tensor's dtype/device. The changed-default-dtype
regression passed locally and in a CPU-only A3 gate. Vendor/JIT/provider source,
original measured wrapper hashes, fixtures and all scientific results remain
unchanged; no compilation, NPU execution or profiling was repeated for this fix.
