R04 studies the prepared forward component of Triton-Ascend-Kernels rope_forward
at 7acb9c40a6ca827e885d902b7eaee5d45457b230. The original JIT is retained verbatim
in rope_layout_source.py with MIT/Huawei/LinkedIn notices. The candidate changes
only Q/K load addressing to explicit physical input strides and separate output
pointers. Token grid, padded heads/dimension, trig loads/broadcast, FP32 cast and
arithmetic/store order are preserved. No head repartition, GPU trig, backward or
autograd change is included. Both use common stage1, multibuffer=False and
FP fusion=False; these are controlled common flags.

Four main physically BHSD inputs require two real before-arm Torch copy_ calls
on every prepared forward, followed by the original in-place JIT on private
BSHD scratch. After reads original BHSD strides and writes private BSHD output.
Both return logical BHSD output views with source strides, plus original cos/sin
objects (four outputs). Input transfer, CPU validation, allocations and compiler
qualification are outside both APIs; private outputs are reused. Every actual
copy task remains charged; task names/types are observed from retained traces,
not predicted or excluded. Unsupported trace accounting stops measurement.

Physical layout determines observable source aliasing. GA has Hk=1: transposing
BHSD K is already contiguous. Before copies Q once and mutates original K;
after directly reads Q into private output and mutates original K. Both preserve
K alias and its original singleton-head stride S*D, rather than normalizing that
stride to D. GB's logical BHSD views have BSHD backing: both transposes are
contiguous, both arms use the original JIT with zero copies, and both inputs
mutate once and alias outputs. Independent per-arm backings prevent contamination.
Guards receive one correctness call per arm and no profiles.

The initial prospective contract incorrectly required GA K immutability. A
nonfrozen CPU storage-only audit identified this before source implementation,
compilation, scientific expected generation or NPU work. The retained v1 contract
was unexecuted. Authoritative v2 preserves mixed GA alias/mutation/strides; all
cases, seeds, math and numerical gates remain fixed. No postmeasurement repair.

Inputs are contiguous BHSD or exact BSHD-backed logical BHSD FP32 views, with
independent nonoverlapping Q/K/cos/sin storage and cos_B=1 or B. Q/K are q/128
in [-1,1]; trig tables are represented q/256 in [-1,1] with duplicate halves.
The separate CPU FP64 half-rotation oracle has exact dyadic products and sums:
integer products <=32768 and sums <=65536 fit FP32 exactly. Primary validation
requires exact numerical Q/K equality; paired FP32 bits and zero signs are
reported separately without imposing FP64 oracle zero signs.

Address proof: physical BHSD strides are [H*S*D,S*D,D,1]; BSHD-backed logical
strides are [S*H*D,D,H*D,1]. Source uses token row stride H*D in canonical BSHD,
or D for singleton-K alias. All active b/h/s/d coordinates stay inside their
actual storage. Source and candidate masked padded-head/dimension pointers can
extend beyond logical storage, but all pointer offsets stay below 2^24 throughout
B<=2,S<=256,Hq<=16,Hk<=4,D<=64; source token arithmetic is int64. Masks independently
exclude head>=H and half-dim>=D/2. All prepared Q/K input and private output
backings have 32 FP32 sentinels per side; full active values, original strides,
alias/mutation predicates, cos/sin identity/immutability and sentinels are checked.

Host tests use nonfrozen seeds190401/190402, source copying and mixed alias modes,
CPU/mock compilation-failure retry and no-launch probes, changed Torch default
dtypes, repeated main calls/both arm orders, actual copy_ call counts, input
admission/cache/codec and external-cwd CLI rejection. The public CLI is exploratory;
it does not claim to perform the frozen all-case collection.

The unchanged v2 source passed native BZ-A3-2 validation on observed9382
Ascend910V1, 24 Cube/48 vector cores, importedTriton3.2.0/triton-ascend3.2.2,
Torch2.10.0+cpu/Torch-NPU2.10.0 and CANN9.1. All twelve input-only JIT compile/
ELF/resource probes completed before six CPU expected payloads were generated
once. All twelve correctness calls passed before eighty profiled full API calls
in eight captures; guards had zero captures. GA and GB alias/mutation/strides
matched v2 exactly, inputs/cos/sin/canaries passed, and paired bits were equal
in all six pairs (reported separately from the exact numerical FP64 oracle).
The full original six-case coverage audit and three nonfrozen functional tests
passed. No source, case, tile, flag or numerical-gate retuning followed collection.

Observed before task names are `aclnnInplaceCopy_TransposeAiCore_Transpose`
(twenty per ten-API capture) and `_triton_rope` (ten). After has ten
`_strided_rope` tasks. All were actual `AI_VECTOR_CORE` tasks, fully retained in
accounting and PMU rows. No copy task type/name was assumed or excluded.
Raw metadata's device_kernel_calls=1 counts the explicit Triton call; actual
copy tasks and total task counts are separate. Each before API includes both
materialization copies, each after API includes its complete strided operation.
Linked host-operator names were not retained by this normalized async-flow schema
(null in receipts); original vendor traces with complete correlation evidence
remain available. No guessed per-Q/per-K task-name association is claimed.

| B,S,Hq,Hk,D | Before/after operation-span mean (us) | Before/after API-event mean (ms) |
| --- | --- | --- |
| 1,33,8,2,64 | 147.393 / 5.532 | 0.336634 / 0.328388 |
| 2,256,16,4,64 | 246.693 / 13.254 | 0.494150 / 0.356016 |
| 1,256,8,2,64 | 177.101 / 9.620 | 0.377534 / 0.336966 |
| 2,33,16,4,64 | 171.155 / 7.694 | 0.379150 / 0.430374 |

All ten samples, first-call variation, task work/union/gaps, and raw supported
PMU fields are retained. Device spans include gaps between tasks; API events
were measured with profiling active. These are diagnostic prepared-component
measurements, without ordinary latency, allocator, physical HBM traffic,
bandwidth, occupancy, roofline, default-producer or full-model performance claims.

Compact receipt391files/4,633,302B SHA256
`e0bb9ffdb654a63f55ed6e297ea70cb758a0750ea30fe41681a440d20ecbb235`
was fully hash-verified with source/provider/compiler/binary/fixture bindings.
Full vendor report trees remain remote. After terminal completion, no live
workload remained, die0 and both export locks were free, and fresh health/ECC/
process/utilization checks were clean. No scientific retry or replay occurred.
