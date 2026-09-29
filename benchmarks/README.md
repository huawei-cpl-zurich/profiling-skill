# Pinned benchmark baselines

These immutable starting baselines and JSON-lines case specifications were
copied from `huawei-cpl-zurich/ascendc-kernelgen-data`, branch
`br_perf_baseline`, revision
`a42c54b916189500e2f7cb47640980f230f2eb65`.

| ID | Original file | Device | Development cases |
| --- | --- | ---: | --- |
| `gdn` | `npu_benchmark/level4/30_ChunkGatedDeltaRule.{py,json}` | 0 | 40, 49, 47, 46, 45 |
| `bsa` | `npu_benchmark/level4/54_BlockSparseAttnFwd.{py,json}` | 1 | 47, 46, 49, 44, 43 |
| `matmul` | local `benchmarks/streaming_matmul_add.py` control | 2 | 7, 8, 9 |

The GDN and BSA specifications contain exactly 50 cases. Matmul contains seven
correctness cases followed by three performance cases and uses `rtol=atol=2e-2`.
Its source identity is repository revision
`9e39c8d3ee94ebd657ad4a9ee031718665b43efa`, path
`benchmarks/streaming_matmul_add.py`, SHA-256
`c877d1db26a820bc861c756d47bd94c60a7c737a3addbaf5844d04081653e250`.
`baseline.json` is a byte-for-byte compatibility copy of `cases.jsonl`, because
the Python loaders resolve their case file from the module stem. Experiment
candidates are separate files; do not modify these baselines during a run.

Generated controller configurations include matmul and its files are part of
the frozen controller closure. The campaign scheduling change separately adds
isolated matmul agent cells to the measured wave plan.

The three-treatment, nine-cell protocol and reproducible launch procedure are
documented in [`docs/experiment-campaign.md`](../docs/experiment-campaign.md).
