# Running-threshold TopK (F03)

This finite FP16 example returns exactly K values in descending order and matching
unique original indices. K must be between 1 and min(V, 1024). Equal-valued
indices may occur in any order. NaN and infinity are excluded.

Both arms use the same Ascend C 8192-score tiles, FP32 comparisons, sorting
primitives and bounded 1024-element partial merges. BEFORE fully sorts every
tile without pruning. AFTER compares each later tile to the current Kth value,
compacts strict-greater candidates and skips sorting/merging an empty survivor
set. Discarding equal-threshold candidates preserves an arbitrary valid tied
set already retained from earlier tiles. Neither arm emits an unused full-row
permutation. This controlled baseline clarifies the original proposal's
"full vocabulary sort" wording: all input tiles are sorted, then TopK is merged.

The sort, compare, GatherMask and partial-merge routines adapt CANN's A2/A3
TopKTopPSample implementation; the source header and CANN_LICENSE preserve its
license. Filtered source provenance:
`ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn#index/top_k_top_p_sample/op_kernel/top_k_top_p_sample.h:L570-L637`,
librarian query `4f2ef93afa0c4ad1a650a9c4faf5cc53`.
The example does not invoke softmax, TopP or sampling.

`threshold_topk_compile.py --output DIRECTORY` builds both source-owned shared
libraries with the selected runtime's installed CANN SDK and retains exact
compiler, source and binary hashes. It performs no device launch. Set
`THRESHOLD_TOPK_DIRECTORY` to that immutable output directory and pass
`threshold_topk_adapter:factory` to `scripts/optimization_pair.py` with this
directory on PYTHONPATH. Compilation and buffer allocation happen before the
single correctness check. A forward only launches one native kernel and returns
its preallocated value/index tensors. The compiled selectors are
`wave003_topk_before` and `wave003_topk_after`.

`threshold_topk_cases.json` freezes four profile cases plus two correctness-only
guards from prospective matrix v1. Every case's CPU oracle is a full-row sort
computed once and cached. Admission checks exact value codes, descending order,
unique in-range indices and matching gathered values. The equal-score guard
checks arbitrary valid tied indices; the K1025 guard must reject before native
buffers are used. Guards are never profiled. Profiles use exactly ten complete
forwards per arm, zero warmups and no additional unprofiled timing forwards.
The reported device-operation span must include all physical tasks and gaps;
missing PMU columns are unavailable, never zero or inferred traffic.

## Retained A3 result

Candidate v6 passed all four frozen profile cases, both equal-score guard arms,
and the unchanged K1025 host rejection guard. It reuses the original cached CPU
oracles. Each main arm was checked once and then profiled for exactly ten full
forwards: eight captures and 80 profiled forwards, with no warmups or extra timing.
The measured product was Ascend910B4-1 (20 Cube/40 Vector cores), CANN SDK directory
label 9.1.0, Torch 2.10.0+cpu and Torch-NPU 2.10.0.

| Case | Prespecified role | Before span (us) | After span (us) | Before / after |
|---|---|---:|---:|---:|
| F03-01 | gallery benefit | 64.848 | 25.034 | 2.590 |
| F03-02 | gallery cost | 32.442 | 33.052 | 0.982 |
| F03-03 | heldout benefit | 95.644 | 30.120 | 3.175 |
| F03-04 | heldout cost | 48.300 | 49.314 | 0.979 |

These are means of ten **profiled device-operation spans**, not ordinary
application latency. The cost controls retain a small slowdown; results do not
establish a universal threshold-pruning advantage. Each forward has one observed
device task, so its active union equals its span and its inter-task gap is zero.
`threshold_topk_evidence.json` retains every final sample, correctness diagnostics,
exact task selectors, source/binary/trace/CSV digests, raw PMU column names and
rows, and the failed candidate lineage. Task pipe ratios do not establish
physical traffic, roofline position or whole-device utilization.

The profiler actually reports **MIX_AIC**, with Block Num equal to the logical
rows and Mix Block Num twice that count. The task-type macro alone did not
establish an AIV-only launch on this toolchain. The native entry explicitly skips
AIC and AIV block indices beyond the logical rows before any I/O. Earlier
versions lacked that guard and could overwrite adjacent index buffers with
half-score output; their failed correctness checks and partial captures remain
retained as failures. Explicit ownership fences also protect scalar index seeds,
vector merge scratch, input DMA reuse and full-array output completion. No
compiler/task-mode switch was made after observing timing.
