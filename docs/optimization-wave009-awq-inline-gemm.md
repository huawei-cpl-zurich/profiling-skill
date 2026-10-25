# Exact AWQ format inside split-one GEMM

Compiler/type qualification failed before any scientific math. The first
BEFORE materialized GEMM triggered a frontend accumulator-type assertion.
Default CPU payload generation and numeric execution are disabled; explicit
input-only compile inspection is available. AFTER was not compiled or run,
and no numerical or timing comparison exists.

This prepared component compares dequantizing the entire packed AWQ matrix
on every API call followed by materialized-B GEMM, with the pinned source's
inline AWQ GEMM. Both use BM16/BN32/BK32, SPLIT_K1 and identical flags. The
source zero allocation and degenerate split-one sum are excluded in both arms;
private C and required B workspace are reused. This is an exact-format/typed
source variant of the earlier natural-INT4 study, not a new fusion mechanism
or a vendor autotuner/fresh-Python latency claim.

The source is gemm/awq_triton.py at
7acb9c40a6ca827e885d902b7eaee5d45457b230. Its dequantization and inline kernels
retain Huawei/vLLM attribution and [notices](../examples/optimization-library/triton/awq_inline_NOTICE).
Nibble permutation is [0,4,1,5,2,6,3,7], with no zero-point +1. The source's
FP16 C-element accumulator declaration and DEFAULT tl.dot signature are
preserved exactly, including the materialized-B baseline's dot policy.
No accumulator promotion, dot precision argument or tile change is permitted
to make the source compile.

Inputs are owned contiguous FP16 A, int32 packed weights and zero points,
and FP16 scales. A is integer/64 with integer in[-8,8]; scales are integer/256
with integer in[4,16]. Groups32/64/128 divide K and align BK32, or one full
group equals K. N is a multiple of8 and K<=512. Tail guards include M17/N136
and a full K33 group with masked-zero A tail. Signed packed words and all
nibbles are included. Host validation rejects aliases, invalid layout/dtype,
nonfinite/domain violations and padded pointer overflow.

The [frozen matrix](../experiments/optimization-wave009/R09.json) predates code.
All input-only compile/resource/ELF cells and actual typed dot/loop-carry
evidence for both arms must qualify before scientific CPU expected generation.
MIX/CUBE engine evidence is required; absent or ambiguous compiler evidence
stops qualification without math. The attempted CPU fixture path required case-specific compiler type
qualification. Publication now rejects all default make_case/factory, prepare
and probe calls before expected generation or backend work. Explicit
Adapter(compile_inspection=True).probe permits compile_only and always rejects
run. The CLI defaults to rejection before cache access and offers only explicit
--compile-inspection (plus host domain validation).

Two independent CPU models are prospectively declared in each once-generated
payload: exact FP64 full dot rounded to FP16 for compiler-proven FP32 carry,
or an exact FP64 BK32 dot plus previous FP16 carry rounded after each block
for compiler-proven FP16 carry. Selection is exclusively from compiler types,
never outputs. Integer product partial sums are bounded below2^24, making
FP32 accumulation exact for this domain. FP64 errors and rounding bounds are
retained separately. Nonzero FP16 bits must match the selected model and
paired arm; zeros must be numerically zero, with sign differences secondary.
No universal FP64 zero-sign agreement is promised.

After qualification, every GEMM output is checked once against its
typed model before any profiling. Output diagnostics use one CPU transfer and
retain first mismatch coordinates/expected/actual, count, nonfinite count and
maximum errors. Four main pairs request ten complete API profiles per arm;
guards request none. There are no warmups, autotuning, extra timings, profiled
I/O checks or diagnostic GPU forwards. Operation spans/PMUs are instrumented
component evidence, not physical traffic, occupancy or ordinary latency.


The retained qualification attempt compiled one dequantization artifact, then
failed at the materialized GEMM's DEFAULT tl.dot call with an FP16-declared
accumulator. The selected frontend asserted acc.type == ret_ty. Transactional
qualification kept the compiled cache empty; no dot/loop-carry model or
MIX/CUBE GEMM engine could be qualified. The remaining eleven arm cells were
unattempted. No accumulator promotion, precision argument or source-math fix
was made. This is a compiler/type qualification failure, not an infrastructure
or profiling-engine fault, and does not establish an incorrect AFTER output.

The [failure evidence](data/optimization-w009-awq-inline-gemm.json) includes
all43 verified member hashes, partial VEC dequantization binary/resources,
frontend stack, runtime/source identities and exact zero-execution counts.
Both prospective CPU models were exercised only on NONFROZEN CPU tests; zero
scientific expected payloads were generated. Six host tests passed locally
and in a CPU-only A3 gate with backend autoload disabled and audit guards
against real backend/compiler/profiler activity. A full-inventory coverage
audit is an expected failure, not a passing coverage claim. Publication host
hashes are separate from attempted hashes; JIT/provider/type-classifier bytes
are unchanged. No NPU/scientific replay followed the failure.
