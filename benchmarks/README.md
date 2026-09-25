# Pinned benchmark baselines

These immutable starting baselines and JSON-lines case specifications were
copied from `huawei-cpl-zurich/ascendc-kernelgen-data`, branch
`br_perf_baseline`, revision
`a42c54b916189500e2f7cb47640980f230f2eb65`.

| ID | Original file | Device | Development cases |
| --- | --- | ---: | --- |
| `gdn` | `npu_benchmark/level4/30_ChunkGatedDeltaRule.{py,json}` | 0 | 40, 49, 47, 46, 45 |
| `bsa` | `npu_benchmark/level4/54_BlockSparseAttnFwd.{py,json}` | 1 | 47, 46, 49, 44, 43 |

Each case specification contains exactly 50 cases. Experiment candidates are
separate files; do not modify these baselines during a run.
