# Bounded supplied-metadata chunk-sort map

This pair adapts `_make_chunk_sort_map_kernel`, lines 27–64 of
`mindspeed_ops/arch32/triton/sort_chunks_by_idx.py`, from MindSpeedOps revision
`4efe2b675c2d51517c3fdd14c721020b8d41e241`. The upstream source has Huawei and
NVIDIA copyright notices and the Apache-2.0 license, retained in
`references/mindspeed-ops/LICENSE`. The filtered research receipt is query
`30c29d0af3324ab9b48c4c685e29a32c`, snapshot
`92fb5efaf51aa786cdd8d1a5869dd952486dffcdb039897d790393947a2aad54`.
This is a new operator application, with no mechanism novelty claim.

The prepared API receives four contiguous, owned int32 metadata arrays:
`split_sizes[S]`, `input_cumsum[S+1]`, `inverse_sorted_indices[S]`, and
`output_cumsum[S+1]`. The prefixes start at zero, the split sizes sum to N,
and the inverse permutation and output prefixes must agree exactly. Zero
length chunks are valid. One forward produces a private int32 map of N rows
using one kernel. CPU metadata delivery, prefix/inverse construction, row
movement, and fresh output allocation are outside this component's profile.
The prepared output storage is reused between forwards.

Both arms use the source masks, metadata loads, source grid
`min(queried_AIV_cores,N)`, block size `ceil(N/grid)`, and the real source
candidate `SUB_BLOCK_SIZE=64`. Autotuning is disabled. The only arithmetic
delta retains int32 split sizes, cumulative sums, token comparisons, and
map arithmetic instead of source float32 conversions. Compilation uses
`num_stages=1, multibuffer=False, enable_fp_fusion=False` in both arms.

Admission requires `0 < N <= 2**23-128`, `0 < S <= 128`, and an actual
queried AIV count in 1–64. N below 2**24 alone is insufficient: the source
expression adds output prefix to token before subtracting input prefix.
For example float32 `9000000 + 9000001 - 9000000` yields 9000000.
Under the common bound, all virtual tail tokens are less than N+128 and
output prefixes are at most N, so every addition stays below 2**24. All
integer paths fit int32. The pair makes no claim about a wider integer domain.

`experiments/optimization-wave007/E08.json` freezes four profile cases and two
correctness guards. The large guard checks the complete identity map near
the common bound, once per arm without captures. The independent CPU oracle
uses vectorized per-chunk ranges; its payload is cached separately from the
kernel source. Source, compiler artifacts, fixture hashes, failures, all ten
samples per captured arm, and supported raw profiler fields are retained.
Profiled operation spans are diagnostic and do not establish ordinary
latency, physical memory traffic, occupancy, or roofline behavior.

The public adapter is `moe_map:factory`, usable with
`scripts/optimization_pair.py` after adding
`examples/optimization-library/triton` to `PYTHONPATH`. Cases may supply
parameters nested under `parameters` or directly alongside `seed`, `role`,
and descriptive `expected` metadata. Invalid cases are rejected before oracle
cache creation. The first public launch qualifies compilation first, even
when the caller does not use the compile-only screen.

The retained A3 run passed all six pairs, including the full precision guard:
12 correctness forwards, 80 profiled forwards, eight captures, and zero warmups.
The mean diagnostic operation spans in microseconds were:

| Case | Source float32 | Candidate int32 |
| --- | ---: | ---: |
| 01 | 49.358 | 49.368 |
| 02 | 87.250 | 92.694 |
| 03 | 24.942 | 24.988 |
| 04 | 44.654 | 48.578 |

These observations do not establish an improvement. All twelve compiled arms
had distinct source/candidate `.text` before correctness. The initial compile
attempt failed before any device forwards due to an incorrectly runtime-typed
SUB_BLOCK_SIZE; that failure is retained in the receipt. Its CPU fixture cache
was reused after restoring the source constexpr declaration. All six CPU
payloads were generated once. The tested runtime imported Triton 3.2.0,
triton-ascend 3.2.2, CANN 9.1.0, on 910B4-1 with 40 queried AIV cores.
Distribution metadata separately reported Triton 3.5.0; the receipt explicitly
separates this from the actual imported module version.

A review follow-up fixes explicit `sorted_indices` supplied with generated
split sizes: that permutation is now honored and validated instead of being
replaced by the default order. Nonfrozen provider and external CLI tests cover
reverse order and invalid permutations before cache/device preparation.
The original six frozen payloads, measured provider binding, kernel binaries,
and profiling receipt remain unchanged. `E08-publication.json` records the
updated provider hash and the source proof that the frozen-case branches retain
their original behavior; the follow-up gate runs CPU tests only.
