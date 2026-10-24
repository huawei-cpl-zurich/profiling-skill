# Direct paged KV controlled pair

`paged_kv_pairs:factory` compares gather-then-attend with direct page addressing.
The baseline launches one kernel that gathers both K and V to external contiguous
workspaces, then attention. The candidate traverses pages inside the same
attention core. Each forward computes the complete attention output; all gather
work belongs to the measured forward. Both arms use BM=16, BN=32, FP32 online
softmax statistics/accumulation, FP16 probability operands, causal suffix-query
positions, and the same GQA mapping. There is no KV split or autotuning.

Inputs are contiguous FP16 Q[B,Hq,Q,D] and page-major K/V[physical_pages,16,Hkv,D/Dv],
plus an int32 per-sequence page table. The table permutes fully allocated pages.
Output is contiguous FP32 [B,Hq,Q,Dv]. Live inputs and metadata are never outputs.
For empty KV the shared kernel writes zeros explicitly. Unsupported dimensions,
nonfinite inputs and malformed tables are rejected before launch.

The CPU provider constructs a deterministic bounded uniform [-0.25,0.25] payload
and reconstructs logical pages for an FP64 causal attention oracle. Acceptance
uses atol=rtol=1e-3, exact output shape/type and finite values; the device output
starts at NaN so incomplete writes fail. Gallery specifications are F01-01 and
F01-02 in `../cases/paged-kv/`. Guard A exercises a partial final page and shuffled
pages; guard B exercises empty KV. Guards have no profiling forwards.

For measurement cases, `scripts/optimization_pair.py` performs one correctness
check per arm followed by exactly ten full profiled forwards per arm. Cache the
provider fixture once and preserve failures. Do not warm up, retune, inspect
outputs during profiling, or infer device traffic from counters that lack
documented byte semantics. Retain stage counts and whole operation spans;
baseline has two stages and direct attention has one.

Guard usage requires the shared correctness-only runner API from
[PR #191](https://github.com/huawei-cpl-zurich/profiling-skill/pull/191)
(commit `2b7cf99`). That API also automatically selects correctness-only execution
for a case with `role=guard`. From the repository root, with the selected profile's
`provenance.json` prepared and fresh output directories, use the explicit flag:

```bash
for guard in F01-GA F01-GB; do
  PYTHONPATH=examples/optimization-library/triton python scripts/optimization_pair.py \
    --adapter paged_kv_pairs:factory \
    --case "examples/optimization-library/cases/paged-kv/${guard}.json" \
    --output "run-evidence/${guard}" --cache run-evidence/oracle-cache \
    --provenance provenance.json --correctness-only
done
```

Each guard performs one correctness check per arm and zero profiling forwards.
The flag is a dependency of these instructions; the earlier shared runner does
not support it.

## Retained A3 runtime status

This pair remains experimental and the PR remains a draft. The complete cost
pair was slower after the change; the retained A3 study does not establish a
profiling gain.

| Case | Once-per-arm correctness | Retained profiling outcome |
|---|---|---|
| F01-01 | Both passed before capture | Baseline submitted 10 forwards, zero event samples, vector-core timeout 507034; after count unknown following process abort. |
| F01-02 (cost) | Both passed | Complete 10-forward capture per arm; device spans 682.022 µs before and 1349.814 µs after, before/after ratio 0.5053. |
| F01-03 | Both passed | Baseline submitted 10 forwards, zero event samples, vector-core timeout 507034; fatal stop prevented after capture. |
| F01-04 | Unattempted | Unattempted under the stop rule following the second timeout. |
| F01-GA / F01-GB | Both arms passed for each guard | Correctness only; zero profiling forwards. |

The timeouts' cause remains unresolved. Failed cases were retained without
unchanged-payload replay. Only F01-02 provides a complete paired measurement;
its values are profiled device-operation spans, not ordinary application latency.
No four-pair success or general runtime coverage is claimed.

Mechanism provenance:
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/attention/paged_attention.py:L381-L405`.
The source repository is MIT; its header credits Huawei, DeepLink, OpenMMLab and
lmdeploy v0.6.1. This pair implements the isolated address-mapping mechanism;
it does not claim to reproduce every source feature or its split-K dispatch.
