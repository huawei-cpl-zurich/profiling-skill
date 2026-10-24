# W007 E05: Sinkhorn forward launch fusion

This study isolates joining the qualified eight-lane Sinkhorn pre/post and comb kernels into one kernel. The before arm always uses eight comb lanes; the parent width-four optimization is not part of this experiment. The fused JIT invokes the unchanged parent pre/post body, including stores, followed by the unchanged parent comb body with the same GROUP32 grid and common `enable_fp_fusion=False` flags.

Source provenance is MindSpeedOps `4efe2b675c2d51517c3fdd14c721020b8d41e241`, `mindspeed_ops/arch32/triton/sinkhorn.py` (repository Apache-2.0); source test `tests/unit_tests/triton/test_sinkhorn.py` has its own MIT notice retained in SINKHORN_LICENSE. Filtered library receipt: `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-mindspeed-4efe2b675c2d#`, query `30c29d0af3324ab9b48c4c685e29a32c`. The exact source file receipt is SHA256 `eb3cfd9b0a411154bffd6fb1a347c83920eaec28e55ee491c6d2399140c2e83f`. This is a new application of launch fusion, with no claim of a unique mechanism.

The parent FP64 reference is reused unchanged. Six frozen cases each generate one cached CPU payload. Each arm executes one complete API correctness call per case, checks all three outputs against fixed 3e-6 absolute/relative bounds, and checks exact paired output bits before any capture. Output backing remains `[BS,4,8]`, with comb view `[BS,4,4]` and strides `(32,8,1)`; private output buffers are allocated during prepare and reused. Positive-zero padding, canaries, input immutability, finite FP32 ownership and nonaliasing remain required.

The collector compiles and screens full binary identity, actual `.text` and relocations for every case before mathematics. Four main cases request ten prepared complete API forwards per arm; two guards request no captures. There are no warmups, extra timing forwards, or profiled output inspection. Before has two device components per API call; after has one.

Reported spans, task sums, union and gaps describe instrumented device events. They exclude allocation costs and do not establish ordinary Python latency, physical DRAM traffic, occupancy or roofline behavior. All slower, equal, unsupported and failed outcomes are retained without changing the frozen matrix or tolerance.

Standalone admission works from any cwd:

```bash
python /path/to/examples/optimization-library/triton/sinkhorn_fusion_cli.py \
  --case case.json --output output --cache cache --provenance provenance.json --validate-only
```

`--validate-only` accepts flat or nested parameters and does not create output/cache directories. Actual captures require a qualified A3 profile through the installed remote-access transport. The retained evidence is recorded in `docs/data/optimization-wave007-E05.json`.

A3 validation completed successfully on a healthy idle Ascend910B4-1. All six paired bit checks passed. Twelve correctness API calls and eighty profiled API calls produced 138 device kernel launches; guards had no captures. Seven functional host tests also passed on A3 (45.483 seconds).

The table retains all four planned main scenarios, including the prospectively designated cost cases. Values are means in microseconds across ten instrumented calls per arm. For one fused task, operation span equals active union and there is no intercomponent gap.

| Case (BS, iterations) | Span before → after | Task sum before → after | Gap before → after |
| --- | ---: | ---: | ---: |
| 128, 1 | 171.156 → 9.578 | 13.576 → 9.578 | 157.580 → 0.000 |
| 4096, 20 | 226.094 → 74.148 | 83.012 → 74.148 | 143.082 → 0.000 |
| 257, 4 | 175.978 → 11.140 | 14.678 → 11.140 | 161.300 → 0.000 |
| 1, 20 | 179.598 → 15.608 | 16.868 → 15.608 | 162.730 → 0.000 |

Fusion reduced both the profiled component sum and the intercomponent gap in this retained run. The large span changes include profiler-instrumented launch gaps; they are not ordinary latency speedups. All ten raw samples, supported PMU values, compiled binaries and resource metadata remain bound in the dataset and compact manifest.
