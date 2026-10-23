# Ascend optimization library — wave 001 approval proposal

Status: source research and experiment design. No example below has been
compiled, correctness-checked or profiled for this wave. Source comments and
historical upstream measurements are hypotheses for our experiments, not our
performance results.

## Approved project boundaries

- Implement the project-owned skill, examples and runners in profiling-skill.
- Work in waves of twelve optimization families, with user approval before
  each wave. This document proposes the first wave.
- Validate Triton and Catlass C++ where applicable. Python Catlass DSL and
  TileLang material may inform research but establishes no A3 execution claim.
- Use gz-a3-2 physical die IDs 0–7. Its inventory also contains 8–15; those IDs
  are outside this task's device allowance. Device eligibility is refreshed
  before each workload; this proposal reserves no device.
- Independently generate expected outputs once per case. Check each before
  and after implementation once, then run ten profiled forwards per arm.
  No hidden warmups, repeated checks or tensor inspection in measured forwards.
- Run two held-out scenarios through actor and analyzer, with and without
  the skill: 96 no-code evaluations for this wave.

## Source acquisition and provenance

Fresh paginated GitCode organization listings contained 88 CANN repositories
and 106 Ascend repositories. Filtering names beginning with `ops-` selected
the following seventeen; every selected repository was actually cloned and
its checkout matched its freshly queried immutable revision.

| Namespace | Repositories |
| --- | --- |
| cann | ops-blas, ops-collections, ops-cv, ops-fft, ops-gnn, ops-math, ops-multimodal-fusion, ops-nn, ops-rand, ops-ras, ops-solver, ops-sparse, ops-tensor, ops-test-kit, ops-tilelang, ops-transformer |
| Ascend | ops-rec |

Four additional repositories were cloned. Historical revisions were selected
from those checkouts for the documented design and tutorial examples.

| Source | Immutable revision | Research purpose |
| --- | --- | --- |
| Ascend/triton-ascend-kernels | `7acb9c40a6ca827e885d902b7eaee5d45457b230` | Triton GEMM, attention, normalization and chunked recurrence |
| cann/catlass | `40aced927fdaaa0f3eaa9c6f5c5d859e06392f55` | Current native C++ examples and AtlasA2 design/components |
| cann/catlass v1.3.0 | `550bbffa8b36bd554aa8b7451da65c26adc0942c` | Historical design rationale and controlled example geometries |
| Ascend/triton-ascend main | `865691e2e9b656bc58008170207b4108d92e8dd1` | Migration notice and pinned GitCode provenance |
| Ascend/triton-ascend historical tutorials | `9319a71c74a630d7b2b47676557f29f529165a9c` | Executable-source attention tutorial mechanism |
| Ascend/MindSpeed-Ops | `4efe2b675c2d51517c3fdd14c721020b8d41e241` | Gated delta-rule operator contract and implementation context |

Sources are retained through the managed reference broker. Indexed snapshots
contain their declared relevant UTF-8 source/test/documentation selection,
not byte-complete Git repositories. Per-file licenses and attribution remain
part of the source evidence. Development worktrees do not contain upstream
checkouts or a copy of the broker store.

## Twelve proposed mechanisms

The before/after boundary isolates the named mechanism. Case dimensions below
are proposed inputs, not a promise of improvement. An implementation that
fails correctness or compilation is a rejected experiment; it cannot serve
as the correct tradeoff example. A tradeoff may be slower execution or a
measured resource/workspace/numerical-margin cost under the same contract.

The proposed numerical contracts are fixed before execution:

| Cases | Inputs and outputs | Independent oracle | Shared correctness tolerance |
| --- | --- | --- | --- |
| GEMM families except 02 | FP16 inputs/output; logical layout values identical between arms | FP32 CPU accumulation, followed by the declared output conversion | rtol=1e-3, atol=1e-3 |
| 02 GEMM plus add | FP16 A/B, FP32 addend Z of shape [M,N], FP32 output | FP32 CPU A@B + Z, with no FP16 intermediate conversion | rtol=1e-3, atol=1e-3 |
| 01 KKT scaling | BF16 row-normalized K, finite beta in [0,1], FP32 output | FP32 CPU dot, row beta scaling and identical triangular mask | rtol=1e-2, atol=1e-2 |
| 03 attention | FP16 Q/K/V, FP32 output | FP32 CPU score, softmax and value product | rtol=1e-3, atol=1e-3 |
| 04 GDN | FP16 inputs/output, FP32 initial/final state | Independent sequential FP32 CPU recurrence | rtol=2e-2, atol=2e-2 for both outputs |

These are contracts for the proposed gallery examples, not a claim that every
upstream operator uses these tolerances. Both arms must satisfy the same
contract, including shape/dtype, finite output, tails and input immutability.
Record actual maximum absolute/relative errors. Tolerances are never relaxed to
retain a fast candidate. The total pipeline, including prologues, epilogues,
workspace operations and reduction kernels, is the timing boundary.

| ID | Mechanism and formats | Before → after | Benefit regime to test | Correct tradeoff/counter-regime to test |
| --- | --- | --- | --- | --- |
| 01 | Invariant scaling hoist — Triton | Apply beta to each K input tile before a dot → apply beta once to accumulated rows | Chunk size BT=64, K=256; batched lower-triangular KKT blocks | BT=64, K=32; extra post-dot scaling can exceed the removed multiplication count; include cancellation/extreme-beta inputs |
| 02 | Epilogue fusion — Catlass C++ and Triton | GEMM followed by an elementwise-add kernel → fused add epilogue | M=N=2048, K=256; intermediate GM traffic and an extra launch | M=N=256, K=8192; test synchronization/live-resource cost of the mixed epilogue, retaining identical output semantics |
| 03 | Query blocking and K/V reuse — Triton | One query row per program → query tiles with shared K/V loads and online softmax | B=1, H=8, Q=K=1024, D=64 | B=1, H=8, Q=17, K=65, D=128; padded work and live accumulator pressure; explicitly support tails |
| 04 | Chunked recurrence — Triton | Tokenwise delta-rule state update → chunk preparation and chunkwise state scan | T=4096, Hk=1, Hv=4, DK=DV=128, two packed segments | T=17 with the same head/state dimensions; preparation/launch overhead and a partial chunk; nonzero initial state and final-state output |
| 05 | Tile/resource balance — Catlass C++ and Triton | Output tiles 128x256 → 256x128, holding K blocking/algorithm/dtype fixed | Target-core-derived geometry below gives an output-task count at or below C after, but above C before | Same geometry with alternate supported B layout; source demonstrates that movement/alignment can outweigh fewer task waves |
| 06 | SplitK — Catlass C++ | M/N-only task grid → split K plus complete partial-output reduction | M=N=64, K=16384; deliberately sparse M/N task grid | M=N=2048, K=256; already abundant M/N parallelism and extra partial-result traffic/reduction |
| 07 | Multibuffering — Catlass C++ | Sequential operand movement/compute → supported ping-pong buffers | M=N=2048, K=4096; multiple reusable K stages | M=N=256, K=256; fill/drain and additional buffer capacity with little work to overlap |
| 08 | Preload across output tasks — Catlass C++ | Load the next output task after the current task → preload while the current task computes | M=N=2048, K=512; enough output tasks per core | M=128, N=256, K=512; too few tasks to amortize preload/drain; respect PRELOAD_STAGES < L1_STAGES |
| 09 | L1 operand residency — Catlass C++ | Reload A across output tasks → retain a supported full-A tile | M=128, N=32768, K=256; enough N tasks for each core to reuse A across output tasks | M=128, N=256, K=256; little reuse. Both arms must fit L1; an over-capacity compile failure is only an additional rejection case |
| 10 | Padding and physical layout conversion — Catlass C++ | Direct strided ND access → explicit padding/repacking prologue | M=1024, N=1000, K=1000; nonaligned input strides | M=N=K=128 with aligned strides; count every conversion/copy and workspace cost |
| 11 | ShuffleK — Catlass C++ | Cores start at the same K tile → rotate the starting K tile by core | M=N=2048, K=4096; several concurrent K tiles | M=N=512, K equal to one K tile; no useful stagger, plus index/order cost; verify rounding under the changed sum order |
| 12 | Output-tile program ordering — Triton | Row-major M/N tile assignment → correctly guarded diagonal assignment, keeping K ascending | M=N=4096, K=2048; many output tiles and substantial B footprint | M=256, N=512, K=64; small/cache-resident tile grid and mapping overhead; rectangular/tail grids must remain bijective |

For 02, the before implementation writes an FP32 GEMM intermediate and adds
the FP32 Z tensor in a second device kernel; the after implementation adds Z
to the FP32 accumulator before the final FP32 store. Include allocation and
device traffic of that FP32 intermediate in the declared whole-call boundary.
Neither arm rounds the intermediate through FP16. This avoids changing the
mathematical operation through a different conversion order.

Numerical precision and reduction ordering are cross-cutting contracts for all
twelve entries. They are not counted as a thirteenth recipe or used to fill a
slot without an independently evidenced optimization mechanism. Output-tile
ordering is a proposed source-grounded replacement for that provisional
standalone family and is part of this approval request.

Case 05 derives its geometry from the selected die's observed Cube-core count
C, rather than assuming the upstream example's twenty cores. Choose positive
a and odd b in [3,31] such that a*b <= C < a*(b+1), maximizing a*b and then
minimizing a*(b+1); set M=256*a, N=128*b, K=6144. With the declared tiles,
before has a*(b+1) output tasks and after has a*b. For example, C=24 produces
M=2048, N=384, with 32 versus 24 tasks. This establishes a task-count hypothesis,
not a latency prediction. If no supported candidate satisfies the rule, report
the capability gap rather than substituting the historical twenty-core claim.

## Source anchors and expected observations

The following references are research evidence. They do not certify our
examples, the upstream kernel, or A3 performance.

| ID | Pinned source selector | Expected observation and interpretation limit |
| --- | --- | --- |
| 01 | [KKT beta hoist](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/chunk_scaled_dot_kkt.py:L82-L94), plus `docs/sources/best-practices/chunk_scaled_dot_kkt.md:L75-L105` | Multiplication count changes from BT*K to BT*BT. Test duration and emitted/cast work; scaling before versus after the dot changes rounding. |
| 02 | [Block epilogue assembly](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/03_matmul_add/matmul_add.cpp:L111-L120) | Fewer launches and less intermediate traffic may help. Compare whole-call time and compiler-reported resource use; a required additional output cannot be silently removed. |
| 03 | [Attention online update and UB slicing](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-historical-scope-9319a71c74a6#ascend/examples/tutorials/04-fused-attention.py:L111-L138) | Reuse Q/K/V and avoid an attention-matrix intermediate. Larger live accumulators can require slicing. Replace the historical fixed scheduling stride with a verified mapping. |
| 04 | [Chunk state update](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/chunk_delta_h.py:L157-L171) | Inspect duration across all preparation/state/output kernels. More matrix work does not by itself establish a speedup; preserve both outputs and packed boundaries. |
| 05 | [Task balance versus layout example](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-v130-design-550bbffa8b36#docs/contents/advanced/catlass_optimize_guidance.md:L114-L150) | Compute task counts using the actual target core count; compare elapsed time and resource footprint. Historical 20-core timings are not target measurements. |
| 06 | [SplitK cost model](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L217-L249) | More active tasks can reduce the long-K critical path, but partial writes and ReduceAdd are part of the result. |
| 07 | [Buffering mechanism and residency caveat](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L293-L322) | More buffers consume L1/L0 space. Pipe-active ratios alone cannot prove simultaneous overlap; require supported temporal evidence for an overlap claim. |
| 08 | [Preload mechanism](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L332-L348) | Test reduced transfer bubbles and total duration. Explicitly count fill/drain and respect dispatch-policy capacity limits. |
| 09 | [Full-A L1 capacity gate](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#examples/25_matmul_full_loadA/matmul_full_loadA.cpp:L92-L103) | Establish reuse and capacity from the source/compiled configuration. Claim fewer physical reads only with supported traffic evidence. |
| 10 | [Physical-layout rationale](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L371-L417) | Compare complete conversion-plus-compute cost and workspace. Alignment improvement alone is not an end-to-end win. |
| 11 | [Per-core K-start rotation](ref://profiling-skill/common/perf-analysis/repos/optimization-wave1-catlass-40aced927fda#docs/en/2_Design/01_kernel_design/04_matmul_summary.md:L447-L453) | Test conflicting-access hypothesis and duration; no stagger exists with one K tile. Do not confuse this with output-tile ordering. |
| 12 | [Threshold-gated output mapping](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/matmul.py:L100-L117) | Test locality/mapping cost. Source lines 136–141 contain a suspicious tuple assignment, so independently implement and prove a bijective mapping rather than copying the source unchanged. |

## Planned validation and evaluation accounting

The gallery proposes fourteen language implementations: twelve families plus
second-language examples for 02 and 05. Two regimes and two arms produce
**56 correctness checks and 56 ten-forward captures (560 profiled forwards)**,
with 24 independent oracle cases.

Fresh held-out evidence is acquired separately, before actor/analyzer tests,
using both before/after arms but only one language per held-out scenario.
This adds **48 correctness checks, 48 ten-forward captures (480 profiled
forwards), and 24 oracle cases**. The complete planned wave therefore has
**104 checks, 104 captures, 1,040 profiled forwards, 48 oracle cases and 96
no-code model evaluations**. Only before-source/profile artifacts are shown
to the evaluated models; after-source/results remain with the independent
grader. Correctness and profiler acquisition are not repeated for the two
model conditions.

Expected outputs are reused across languages where contracts agree.
Additional correctness-only
stress cases or diagnostic replays must be declared in the final case ledger;
they are never hidden in the primary 1+10 counts.

The separate held-out case recipes use the same mathematical contracts as the
gallery. The beneficial/adverse labels are hypotheses for acquisition; gold
decisions are assigned from the actual paired evidence. An inconclusive
acquisition cannot be converted into a confident apply/avoid answer key.

| ID | Held-out beneficial recipe | Held-out adverse/cost recipe | Language |
| --- | --- | --- | --- |
| 01 | BT=32, K=128 | BT=32, K=16 | Triton |
| 02 | M=1536, N=2304, K=384 | M=192, N=320, K=6144 | Triton beneficial; C++ adverse |
| 03 | B=1, H=4, Q=K=1536, D=64 | B=1, H=4, Q=23, K=79, D=128 | Triton |
| 04 | T=3073, Hk=2, Hv=4, DK=DV=64; three unequal packed segments | T=31 with nonzero initial state and the same head/state dimensions | Triton |
| 05 | The actual-core task-count rule above, K=4096 and fresh data | Same dimensions, alternate B layout and fresh data | C++ beneficial; Triton adverse |
| 06 | M=N=96, K=12288 | M=N=1536, K=384 | C++ |
| 07 | M=1536, N=2304, K=3072 | M=192, N=320, K=192 | C++ |
| 08 | M=1536, N=2304, K=768 | M=N=128, K=768 | C++ |
| 09 | M=128, N=24576, K=384 | M=N=128, K=384 | C++ |
| 10 | M=1536, N=1500, K=1004 | M=192, N=256, K=256 | C++ |
| 11 | M=1536, N=2304, K=3072 | M=384, N=640, K equal to one K tile | C++ |
| 12 | M=3072, N=4608, K=1536 | M=192, N=384, K=96 | Triton |

The first approved Triton and Catlass C++ pairs are hardware/collector pilots.
Successful pilot evidence is reused rather than rerun for reporting. The
runner must handle both arms symmetrically; the existing eager-PyTorch baseline
controller is not sufficient for this gallery.

Retain exact compiler/kernel identities, hashes, raw ten samples, mean/sample
deviation, per-kernel rows, complete-call timing, numerical errors, capture
order, workload geometry, runtime versions and die eligibility/lease evidence.
Keep sample completeness separate from export/accounting/identity success.
Effects within observed variability remain inconclusive. Unsupported metrics
remain unavailable; activity percentages cannot supply a bandwidth numerator
or an unsourced roofline denominator.

Run at most three isolated die jobs concurrently and at most two heavy exports.
Each pair holds its die lease across before/after captures. Observe retained
remote job handles after interruptions; do not duplicate live work.

For each family, a separate author supplies one beneficial and one adverse
held-out scenario. Actor and analyzer each receive the same source/evidence
under two fresh conditions: existing skills only, or existing skills plus the
new optimization library. Neither condition sees the optimized holdout answer,
after-profile results, this proposal's labeled case matrix, or grading key.
Evaluation mounts contain only the allowed skill files and anonymized
before-source/profile bundle, never the full repository. Both roles may read
artifacts and write analysis only.

Actor outputs must choose apply/avoid, describe concrete edits without writing
kernel code, preserve preconditions and identify expected benefit/cost.
Analyzer outputs must connect actual code/profile evidence to the choice,
state uncertainty and propose a falsifying observation. Each with-skill
scenario must make the correct conditional decision, preserve correctness
guards and avoid unsupported counter/roofline claims. Keep failed attempts and
initial scores; use fresh held-outs after a skill revision. Control ties are
reported as ties rather than an invented skill improvement.

## Approval and implementation sequence

This proposal requests approval of these twelve families, formats, semantic
case recipes, measurement accounting and 96-evaluation design. The case
ledger will retain any resource-compatible refinement needed within an
approved family, with every attempted source/case/result preserved. A family
replacement or additional replay is an explicit matrix amendment.

After approval, implement the project-owned skill and symmetric pair/evaluation
runners, qualify the two language pilots, then implement the twelve entries
in independent PR-sized streams. Source and test changes target approximately
500 lines per PR where a cohesive split is possible. Every runnable example
must pass actual A3 correctness and profiler acquisition before being labeled
verified. Carry out the repository's functional validation and review workflow.

At wave completion, present source-backed cards, paired data, rejected
hypotheses, model-understanding scores, review results and remaining limits.
Stop for approval before beginning another wave of twelve.
