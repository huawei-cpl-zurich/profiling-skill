# W004 W11: shared tridiagonal factors

This pair independently implements no-pivot FP32 Thomas elimination. The before
arm recomputes factors for each RHS inside its fused forward/backward solve. The
after arm factors A once on every API call, then solves all RHS using read-only
factors. Both arms use the same RHS ownership and FP32 arithmetic order. The
additional factor launch and global factor storage can outweigh saved work.

The native host launcher calls factor once followed by solve-all, with a separate
single-RHS branch. This establishes the ownership idea, not the unseen device
helper's arithmetic. The Thomas recurrence is independently implemented here;
there is no claim that the native helper was copied, executed, or benchmarked.
Filtered source receipts from query `80ab2af57e2442dfb3acc6f0759e2a43`:

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/gtsv2/arch22/gtsv2_host.cpp:L1-L200`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/gtsv2/arch22/gtsv2_kernel.h:L1-L210`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/gtsv2/README.md:L1-L180`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#test/gtsv2/README.md:L1-L220`

Unresolved native-helper source miss `11501d136a88a71c6d38ce1feaa9647e` remains
open. Source scope therefore stays at host factor reuse and documented Thomas.

The frozen matrix is `examples/optimization-library/cases/wave004/W11.json`,
SHA256 `d2530df0d3c34c19318b1c648ea6f359c74f67992fa327d8ffdcb6165fab2e66`.
The four fixed `(m,R)` cases are `(512,64)`, `(128,1)`, `(768,48)`, and `(64,2)`,
with `ldb=m+8`. Contiguous storage `[R,ldb]` represents logical column-major
`B[m,R]`. Inputs use diagonal 4, seeded off-diagonals in ±.25 and RHS in ±.125.
Both APIs return separate storage, copy padding sentinel 98765.25 inside every
call, preserve original A/B, and rebuild factors every call. Preallocation is
outside the API; no input-dependent factor caching survives across calls.

The independent CPU oracle uses FP64 dense solve. All active output must pass
`atol=2e-5, rtol=2e-4`, remain finite, and satisfy relative infinity residual
`||Ax-b||inf/max(||b||inf,1e-30) <= 2e-5`. Padding is checked bitwise. The
nonsingular zero-first-pivot guard executes a real CPU dense solve per call,
without reading the oracle. Its two arm checks have zero NPU forwards and zero
captures. The `(17,3,24)` guard checks tail rows and padding without captures;
empty RHS is exercised by functional CPU tests.

Use the actual nested case JSON with the pair CLI:

```bash
PYTHONPATH=examples/optimization-library/triton python scripts/optimization_pair.py \
  --adapter shared_tridiagonal:factory \
  --case examples/optimization-library/cases/wave004/W004-W11-01.json \
  --output /tmp/w11-pair --cache /tmp/w11-oracles --provenance /tmp/provenance.json
```

A campaign must compile all arms/resources without forwards before scientific
execution. The fixed protocol creates six CPU oracles once, checks each arm once
per source version, and profiles ten full API calls for each of the four pairs.
It has no warmup, autotune, extra timing, or profiled output inspection. Serial
loops use dynamic induction with `num_stages=1`, `multibuffer=False`, and FP
fusion disabled for both arms. Supported FP32 division replaces an unavailable
`tl.div_rn`; it has identical operation order on both sides.

Profiled whole-device operation span includes every copy, factor, solve and
inter-stage gap. Summed active work and gaps remain separate diagnostics. These
instrumented captures establish no uninstrumented application latency, physical
traffic, bandwidth ceiling, or roofline conclusion.

## A3 result

All four pairs and two correctness-only guards passed on gz-a3 physical die 5
(Ascend910B4-1). The independent CPU and device checks satisfy the frozen
precision, residual, immutability, and padding gates. Each arm has ten profiled
forwards. The whole-device span results are negative for this shared-factor
implementation:

| m | RHS | Before mean µs | After mean µs | Before / after |
|---:|---:|---:|---:|---:|
| 512 | 64 | 806.704 | 1124.860 | 0.717 |
| 128 | 1 | 225.128 | 397.582 | 0.566 |
| 768 | 48 | 703.068 | 858.306 | 0.819 |
| 64 | 2 | 208.096 | 375.178 | 0.555 |

The additional factor stage and global sharing fail to improve the full
operation in these captures. The frozen gallery/heldout benefit labels remain
prospective hypotheses. No cases, seeds, compiler options, or sample exclusions
were tuned after measurement. See
[data/optimization-w004-shared-tridiagonal.json](data/optimization-w004-shared-tridiagonal.json)
for all samples, active/gap accounting, per-task PMU fields, source/binary
identities, archive hash and durable handles.

Collector v1 had a marker-directory collision before the first profile API
entry. Its two completed correctness checks, original six oracle fixtures,
compiled binaries, and zero-entry marker were retained. Collector v2 used
sibling marker directories and resumed the first pair's captures only after
verifying unchanged source/binary digests; it reused the completed first-pair
checks and six cached oracles. It then checked and captured the remaining
cases normally. This operational negative is retained alongside successful
captures. The corrected collection completed exit 0 at
`remote:gz-a3:job:536b0b9d-bdfd-4f6f-91d7-a75f60d9f0fa`.
