# Reuse request row metadata across index tiles

Qualification failed on guard GB in the educational BEFORE arm. The source
AFTER arm was not run for that guard, and the cause remains unresolved.
Default public numeric execution and CPU scientific payload generation are
disabled. Only explicit input-only compile inspection is exposed.

This prepared exact-int32 component compares an educational program per
(token, column tile) with the pinned source's row-grouped program. The source
queries the actual vector-core count for its grid, loops over assigned rows,
and reuses request/prefill metadata while looping over each row's column tiles.
The difference includes program scheduling and parallelism as well as metadata
reuse; it is not a pure load-hoisting or vendor baseline comparison.

The copied source kernel is _convert_req_index_to_global_index_kernel in
index/convert_index.py at 7acb9c40a6ca827e885d902b7eaee5d45457b230, with
[license notices](../examples/optimization-library/triton/request_metadata_NOTICE).
Both arms use BLOCK_SIZE64, BLOCK_N128, identical vector lookup/masking/int32
arithmetic and compiler flags. Source FP32 casts remain confined to sign
predicates. No float round trip is used for addresses or output arithmetic.

Inputs are owned contiguous int32 request IDs, block table and token indices,
plus prefill IDs and three fixed workspace starts when enabled. The host
rejects invalid request/prefill IDs, layouts, dtypes, page values, topK tails
and overflowing pointer/output arithmetic before device movement. topK must
be a positive multiple of128 because the source has no column-tail mask.
Negative token indices and derived out-of-table blocks yield -1, including
prefill rows. Otherwise prefill returns workspace_start+token_index and decode
returns block_table[request,token_index//64]*64+token_index%64.

An independent CPU int64 indexing provider checks every output bit. The
[frozen matrix](../experiments/optimization-wave009/R05.json) predeclares case01
as a cost-risk control with one column tile, and cases02–04 as potential
benefits with multiple tiles. A cost-risk label does not promise a slowdown.
One guard checks negative/last-valid/first-invalid boundaries with mixed
prefill/decode rows. Another uses an actual 262146-column table and valid int32
indices above2^24; it does not claim the source has a precision bug.

Outputs are private int32 buffers reused in both arms; allocation is excluded.
Default Adapter.make_case/factory, prepare and probe calls reject before
scientific payload generation, compilation or input movement. Explicit
Adapter(compile_inspection=True).probe permits compile_only; its run rejects.
The standalone request_metadata_cli.py rejects numeric execution before cache
access; --compile-inspection permits unqualified input-only inspection. Controlled qualification additionally screens all12 input-only arm
cells before generating six expected payloads once, gates all12 numeric calls
before80 API profiles, and holds both host export locks through capture and
synchronous analysis. Guards receive no captures; there are no warmups,
autotuning, extra timings or profiled I/O checks.


The retained run compiled all twelve arms and generated six cached CPU payloads
once. Both arms passed cases01–04 and guard GA (ten checks). Guard GB's BEFORE
arm failed its eleventh numeric check; canaries, full input immutability and
nonaliasing passed. GB AFTER was unattempted. There were zero profiles or
captures, so no timing comparison or speedup is available.

The original validator retained a boolean mismatch, not actual tensor values,
mismatch count or coordinates. Those samples remain unavailable. Retained
TTIR/adapter operations use integer division, remainder and addition with FP32
conversion only for sign checks; no source, compiler or infrastructure cause
has been established. In particular, this is not evidence that the source
AFTER arm is incorrect or that its sign casts create a precision bug.

The [failure evidence](data/optimization-w009-request-metadata.json) preserves
all196 verified archive hashes, six immutable fixture bindings, source/compiler
artifacts and exact call counts. Full-inventory coverage audit is an expected
failure for the missing guard qualification, not a passing coverage claim.
A separate host-only publication guard and improved mismatch diagnostics were
tested on CPU mocks. Future diagnostics use one CPU transfer for count and
first mismatch details; they do not recover the original missing actual values.
Measured and publication host hashes are distinct; JIT/provider/cases/tolerances
remain unchanged. Six functional host tests passed locally and on A3 without
any real compilation, NPU initialization or frozen scientific replay.
