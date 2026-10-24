# Integer slot-mapping pipeline fusion (wave005 T06)

This study is a new operator application of integer address-pipeline fusion. It
adds no new unique optimization mechanism and does not compare against a vendor
best kernel. The four prospective benefit/cost labels are hypotheses; all outcomes
are retained. Source-backed code presence is separate from runtime qualification.

## Mapping and source

The source is `convert_index.py` from triton-ascend-kernels, immutable commit
`7acb9c40a6ca827e885d902b7eaee5d45457b230`, MIT, with Huawei/vLLM notices.
The filtered evidence is:

- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/index/convert_index.py:L13-L99` (body)
- The same file `:L102-L204` (interface)
- `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#tests/index/test_convert_index.py:L9-L64` (reference), `:L71-L120` (decode), `:L123-L184` (mixed mode)

Broker query `9fa52956ef764d31b133b2a0308d66c5` and unresolved compatibility miss
`4fd51bd67e746b2e05ebe59688ebf3f9` are retained in the orchestration record.
This is an independent bounded adaptation, not proof that the upstream package
runs unchanged with the qualified runtime. Source license text is retained in
`examples/optimization-library/triton/slot_mapping_LICENSE.txt`.

All inputs and outputs are contiguous int32. For token `tok`, block size `B`,
request `r`, and block-table width128:

```
tok < 0 or tok // B >= 128: -1
otherwise decode: block_table[r, tok // B] * B + tok % B
otherwise prefill: workspace_starts[prefill_request] + tok
```

The derived-block validity rule applies to prefill too. Unsafe block indices are
clamped before loads; request IDs, ranges, strides and output arithmetic bounds
are checked before device preparation. Negative token values all produce -1.
Integer arithmetic avoids the source's unnecessary float conversion, which loses
exact low address bits above 2^24. Both arms use this same exact contract.

## Paired implementation

BEFORE runs three kernels: decompose block/offset/validity; gather decode block
base or prefill start; emit global slots. Four real int32 intermediate arrays are
fully written and consumed each API invocation. AFTER fuses the same mapping and
gathers into one kernel. Block256, linear grid, launch options, inputs, output and
integer equations are identical. No input-dependent metadata is cached, and no
attention consumer, compaction or alternate layout is fused.

BEFORE logical workspace is `16*T*K` bytes; AFTER has zero mapping workspace.
This is not a peak allocator or physical memory traffic measurement: all arm
allocations are made during preparation, and both arms can coexist. Fewer task
launches and intermediate logical bytes are expected benefits. Random block-table
gathers, live address state and execution jitter can dominate either arm.

## Verification and collection

`experiments/optimization-wave005/T06.json` is the unchanged frozen specification.
Four measured pairs cover large/small decode and large/small mixed mode. Two
correctness-only guards cover negative/upper-bound tokens, exact addresses above
2^24, mixed nonzero prefill starts and an odd K33 tail. An independent CPU int64
oracle verifies the equation before conversion to int32. Outputs begin at INT_MIN;
correctness compares the complete output exactly and checks all inputs unchanged.

Each of the six scientific CPU payloads is cached once. Every arm compiles before
any scientific forward. Each arm then executes one correctness forward, followed
by exactly ten full API profiler forwards for each measured case. Guards have
zero captures. There are no warmups, autotuning, extra timing runs, or input/output
inspection during profiling. The public fixture CLI accepts nested frozen-style
JSON. Six behavioral CPU tests cover boundary semantics, cache reuse, rejected
layout/metadata/ranges, the CLI and ambient default dtype compatibility.

The primary comparison is instrumented complete device-operation span, including
inter-kernel gaps. Instrumented caller events, active compute union and noncompute
gap are reported separately. These are not ordinary application latency, measured
physical bandwidth, whole-device utilization or a roofline estimate. Available
original PipeUtilization task rows are retained without inventing missing counters.

## Results

All four pairs and both guards passed on gz-a3, physical6 / logical0,
**Ascend910B4-1**, CANN9.1.0, Torch2.10.0+cpu, Torch-NPU2.10.0,
Triton3.2.0. The retained gate exited0. All12 arm compilation identities were
stable across correctness/profiling. No failed or discarded attempt occurred.

| Case | T × K | Mode | BEFORE span µs ± SD | AFTER span µs ± SD | Before / after |
|---|---:|---|---:|---:|---:|
| W005-T06-01 | 1024 × 512 | decode | 1308.624 ± 0.640 | 1210.210 ± 0.409 | 1.081× |
| W005-T06-02 | 1 × 32 | decode | 374.922 ± 22.268 | 11.706 ± 0.374 | 32.028× |
| W005-T06-03 | 768 × 384 | mixed | 737.974 ± 1.555 | 679.272 ± 0.286 | 1.086× |
| W005-T06-04 | 2 × 64 | mixed | 372.422 ± 11.901 | 18.110 ± 0.233 | 20.564× |

Geometric mean of these four instrumented span ratios: **5.274×**.
All four designated benefit/cost cases improved; the prospective cost hypotheses
did not produce a measured reversal. This is not evidence that fusion always wins.
The large decode and mixed cases improved only about8%; the small cases benefited
strongly from removing two task launches and their gaps. These results do not
establish a physical traffic reduction or the cause of the modest large-case gains.

BEFORE has three compute tasks per API and AFTER one (160 tasks total across80
profiled forwards). All10 span/event samples, deviations, task ownership, original
PMU fields, source/binary identities, guard checks and oracle manifests are in
`skills/ascend-optimization-library/references/slot-mapping-evidence.json.gz`.
`slot-mapping-summary.json` gives a readable index and evidence SHA. Full raw
profiler trees remain retained remotely; the verified compact archive also contains
all six cached CPU payloads, compiled binary cache and exact source snapshots.

Gate: `remote:gz-a3:job:f4cdd5ba-1565-41a6-a8ea-9ae9b44a31f1`, exit0.
Read-only compact job `remote:gz-a3:job:fab3bfe8-8755-45dc-8278-620600f9292f`,
exit0, no new NPU forwards. Compact SHA256
`5b19f717ae7e7256555a626984b55dddfe461e9636cf556231406eab4052da74`.
