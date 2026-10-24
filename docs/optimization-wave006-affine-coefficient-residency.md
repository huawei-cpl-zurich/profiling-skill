# Wave006 N06: affine coefficient residency

This standalone AIV study applies existing coefficient reuse to FP32 affine
rows. It does not introduce a new mechanism or benchmark complete layer norm.
Its immutable cases and numerical contract are in
[the N06 matrix](../experiments/optimization-wave006/N06.json).

The source motivation is actual repeated gamma/beta CopyIn in
`Ascend/ops-rec` commit `e489c0bf6a2d5f2d05b3205fcacc01b62a908b10`,
`ascendc/mixing/token_mixing/op_kernel/token_mixing.cpp:146–162,197–205,273–275`
(Apache License 2.0), and the retained gamma/beta lifecycle in `cann/ops-nn`
commit `7ed342d28b44b6291aceaf747796028c48d7304c`,
`norm/layer_norm_v4/op_kernel/layer_norm_v4_transpose.h:87–109,152–192,744–748`
(CANN Open Software License Agreement 2.0). The implementation here is an
independent isolated adaptation, using dedicated TBuf ownership instead of
copying the original queues or full normalization algorithm.

Stable source citations:
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec`
and `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn`.
Filtered source receipts are associated with queries
`207c5923eed845cf8a9cc1d498401f5f` and `b339eb85c773462fbfc45ecb8dbc2587`.
Unresolved discovery misses `3d89ba69e929e73c7e63081c196f6f47` and
`1db2b3b7edf2e0eac4137ab052d22b41` remain unresolved.

Each core processes a contiguous ceil-divided row range with one row per
iteration. The grid is min(rows, queried available AIV cores), identical in
both arms. Three disjoint, equally sized UB buffers are reserved once. The
before arm copies gamma and beta for every row; the after arm copies them at
each active core's entry inside every full API call. Both use identical exact
byte transfers, Mul, PIPE_V completion, Add, output transfers and final
completion barriers. Width is bounded to 1024 FP32 elements; UB allocation
uses width rounded up to eight elements and is qualified against queried UB.

The CPU oracle rounds represented-value FP64 multiplication to FP32 before
adding in FP64 and rounding to FP32 again. Primary tolerances are fixed at
atol=rtol=2e-6. Correctness retains output bit hashes and checks paired agreement. The source
uses separate Mul and Add with explicit PIPE_V completion. Instruction-level
equivalence remains unverified because this SDK does not retain readable assembly;
observed bit agreement applies only to the fixed correctness inputs.
The compound GA guard reads/checks the first result before updating the same
caller-owned gamma/beta pointers, then issues a second complete API call into a
separate output. The expected output tuple is generated once. The odd-width
GB guard exercises exact byte tails, immutable inputs and output canaries.
Guards have no captures.

Run the public pair CLI with `--adapter affine:factory` and
`AFFINE_DIRECTORY` pointing to SDK-built libraries. `compile_all` requires the
queried AIV count and UB capacity. Preparation verifies shape/arm/grid exports,
binary and source digests before launch. Each Adapter owns its prepared state;
no coefficients or mutable ABI shape state are shared across cases.

Collection uses one correctness operation per arm, then exactly ten complete
profiled forwards per arm for the four main cases, without warmups or separate
timing. These spans are profiler diagnostics, not ordinary latency or physical
traffic measurements. Logical coefficient request bytes are
`2 * width * sizeof(float) * rows` before and
`2 * width * sizeof(float) * active_cores` after. PMU percentages cannot establish
physical DRAM bytes, bandwidth, occupancy or a full layer-norm speedup.

## Retained collection

All six once-per-arm correctness operations passed the fixed tolerance, exact
oracle comparison, input immutability and output canaries. Both arms produced
identical bits for these fixed inputs, including both GA outputs. This is
observed agreement, not a universal compiler equivalence proof.

Each entry below has all ten diagnostic device operation spans retained per
arm. There are no uninstrumented latency measurements.

| Case | Shape | Before mean (us) | After mean (us) |
| --- | --- | ---: | ---: |
| gallery benefit | 4096 x 256 | 157.988 | 38.534 |
| gallery cost | 1 x 64 | 1.666 | 1.700 |
| heldout benefit | 3072 x 512 | 131.030 | 30.410 |
| heldout cost | 2 x 128 | 1.474 | 1.440 |

The large cases reduced diagnostic spans; the one-row case increased slightly.
The two tiny cases provide no strong benefit claim. All original PMU columns,
ten samples, exact kernel ownership, compiler/dependency/binary/source/fixture
bindings and all twelve prelaunch AIV ELF audits are retained in a compact
archive with SHA256
`f7e724453fce58002147df771ceedf36cd5127c6a79c33f9bfbc00adae3c1549`.
Its 281 file bindings were independently verified after transfer. Full vendor
report trees remain on the validation target.

Six scientific oracle payloads were generated once. Collection used twelve
outer correctness operations, fourteen native correctness API components
(including GA's two per arm), eighty profiled forwards in eight capture contexts,
and four explicit caller coefficient updates. Guards had zero captures. Total
native API forwards: 94; warmups and scientific replays: zero. Two failed
controller qualification stages preceded all science (enum spelling and missing
assembly); their source, logs and artifacts remain retained and their scientific
forward count is zero. Controller corrections reused the original binaries.
