This is an UNQUALIFIED compile-inspection study, not a runnable numerical or
performance example. R06 attempted to compare the exact source causal_conv1d_update kernel at
7acb9c40a6ca827e885d902b7eaee5d45457b230 with direct state-tap addresses.
The original kernel remains verbatim with Huawei/vLLM/Tri Dao notices and MIT.
Only FP32 seq1/width4/state3 is admitted. Its default validate_data=False consumes
physical contiguous [slot,channel,tap] state, strides[D*3,3,1]. The dormant
validate_data=True feature-stride assertion is not the supported kernel layout.
An index-only CPU proof and functional nonfrozen tests establish the reshape
permutation as channel-major→tap-major and its inverse; no scientific expected
payload is produced by that proof.

Both arms preserve the source DIM_BLOCK picker, grid(B,1,1), alignment16, four-tap
axis tl.sum, bias and result/(1+exp(-result)). D2064 uses block16/129loops and
D4096 block1024/4loops. No retile, alternative SiLU or reduction tree is included.
The candidate loads old taps at channel*3+tap, forms the same four-row tap matrix,
loads weights at channel*4+tap, shifts state to [old1,old2,OLDx], and then overwrites
x. Nonnegative indices are unique to prevent state races; -1 skips the whole row.
All active addresses stay within contiguous x[B,D], state[slots,D,3], weight[D,4]
and optional bias[D]; maximum offsets are below signed-int32 bounds. DIM_BLOCK divides
D, so no padded channels are accessed. Mutable x/state buffers each have
32 FP32 sentinels on both sides; weights/bias/indices are independent and immutable.

The public output aliases and mutates x, and state mutates independently. Each
arm owns independent initial buffers transferred once at preparation. Correctness
runs one update, using an independent FP64 scalar oracle from INITIAL values,
including old x saved into state before it is overwritten. Non-SiLU values are
exact dyadic numerics; SiLU uses fixed atol2e-6/rtol2e-5. State is bit-exact across
all slots, including unselected slots. GA skips x row0 and untouched states with
indices[-1,4,1]; GB exercises D16. Halo, alias, shape/dtype/stride, immutable
parameter and paired output-bit diagnostics are checked in the sole validation.

If compilation/code identity permits measurement, the ten profiled updates start
from each arm's ONCE-correctness-advanced x AND state. Both continue evolving
without resets, free copies, extra I/O inspection or CPU reference trajectories.
These are stateful repeated updates, not independent fixed decode-token latency.
Only four main pairs have captures (80 API calls/8 captures); guards have none.

All twelve input-only compile/resource/effective-code probes must pass BEFORE six
scientific CPU expected payloads are generated once. Probes cannot run and compile
qualification commits transactionally. Unsupported source arange/view/insert_slice/
extract_slice lowering is retained as a terminal unsupported baseline; source math
or tile choices will not be rewritten to manufacture a comparison. Compiler no-op
or inconclusive effective identity can stop before math. No numerical/profile
claim is made until the retained collector completes its required gates.

Three nonfrozen functional CPU/mock tests cover old-x state shift, skipped/unused
slots, the physical index permutation, indices/layout/dtype/numeric admission,
no-launch probes, atomic failed compilation/retry, output alias/canaries,
float64/float16 defaults, repeated Adapter orders and genuine state evolution,
cache/codec/role handling and external-cwd CLI rejection. Source JIT calls are
mocked in these tests. Default Adapter/factory oracle generation, prepare, and CLI numeric paths reject
before CPU expected generation/cache/backend/transfers. Only explicit compile
inspection is public: Adapter.compile_probe or CLI --inspect-compile, whose
run always raises and whose compile_only writes source-bound attempt/binary/error
artifacts without claiming numerical qualification. Private prepared logic is
exercised only by CPU-backend mocks in nonfrozen tests. Prepared allocations/transfers and CPU admission are outside both arms;
no ordinary latency, allocator, physical traffic, occupancy or roofline claim.

Native BZ-A3-2 acquisition stopped at the FIRST source compile qualification,
W009-R06-01/before: AttributeError, triton.language has no insert_slice. This
occurred in the JIT dependency visitor before codegen. No source helper was
rewritten or tile retuned. Installed imported Triton3.2.0 (distribution3.5.0),
triton-ascend3.2.2, Torch2.10.0+cpu/Torch-NPU2.10.0 and CANN9.1.0 were retained on
observed9382 Ascend910V1 with24Cube/48vector cores. This is deterministic missing
source/backend capability, not an infrastructure retry or hardware-fault claim.

Exactly one compile qualification was attempted, zero kernels compiled. The
other eleven cells, all six scientific expected payloads, all numerical forwards
and all profiles/captures remain UNATTEMPTED. Startup phase/counters and absence
of oracle/correctness/capture artifacts prove zero scientific activity. No timing,
PMU, effective-code identity, resource-fit, numerical or speedup claim is available.
The full original six-case inventory audit is expected to FAIL; it retains false
correctness/compile flags and actual unsupported/unattempted routes, never a
fake passing coverage record.

Failure compact33files/62,275B SHA256
9d5d05ed378033d2c837c374c388d217dc3b7101591daf8020229eceb80a201a
was fully hash-verified. Original source/provider/JIT/frozen matrix and failure
remain unchanged. R06-publication.json separates attempted adapter hashes from
failclosed host-publication hashes. Four nonfrozen CPU/mock tests, including new
default rejection and inspection artifact/capability-error checks, passed a
separate CPU-only native A3 gate with no actual compile, numerical NPU calls,
scientific expected generation or profile replay. Device lease and both export
locks were confirmed free, no live workload remained, and health/process/ECC
checks were clean after the failed attempt. This PR intentionally remains draft.
