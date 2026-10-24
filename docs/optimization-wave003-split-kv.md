# Four-way decode KV partitioning

This pair isolates partitioning a decode query's contiguous KV range into four
independent ranges and merging their nonlinear softmax statistics. The baseline
traverses one range. Both arms keep Q=1, the same 16-row padded grouped-head tile,
64-key blocks, FP16 dot operands, FP32 online maximum/denominator/weighted-value
state, scale 1/sqrt(D), and FP16 output. GQA groups map to their original KV head;
causal decode admits positions [0, KV). Inputs are finite FP16 in [-0.25,0.25].

For partial state `(m_s,l_s,a_s)`, the merge uses `M=max(m_s)`,
`L=sum(exp(m_s-M)*l_s)` and `A=sum(exp(m_s-M)*a_s)`, returning `A/L`.
Adding normalized partial outputs would produce a different result. Probability
conversion to FP16 occurs at the same weighted-value dot boundary in both arms.
Different partition reduction order can change floating rounding.

The after arm partitions blocks with `ceil(ceil(KV/64)/4)` blocks per range.
It uses the unsplit path when fewer than four blocks exist. Empty KV returns
fully initialized zero output through that same safe unsplit kernel. Measured
cases keep every range occupied. The tail case KV449 preserves all valid keys
and excludes padding. Sliding windows and softcaps are outside this contract.

Output and partial workspace allocation and input transfer occur before the
forward. Every forward overwrites all used partial statistics before the merge;
there is no required zero-fill outside the operation. The complete operation
contains every range launch and partial store plus the merge launch and reads.
Device span includes gaps between the stages; active union and idle gap are
reported separately. Profiler event durations remain instrumented diagnostics,
not ordinary uninstrumented latency or physical-memory-traffic measurements.

The four fixed cases and two correctness-only guards are under
`examples/optimization-library/cases/wave003/F07*.json`. Fixtures use an
independent dense FP64 CPU oracle, output atol=rtol=2e-3. Per source version,
each side runs correctness once; each measured side then runs exactly ten
profiled forwards, with no warmup, autotuning, extra timing, or profiled output
inspection. Guard cases request zero captures. Compile-only qualification covers
all arm/case specializations before scientific forwards.

Source provenance: `ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/attention/paged_attention.py:L132-L221`
and the nonlinear reducer at lines225-272. The MIT source retains DeepLink,
OpenMMLab, Huawei, and lmdeploy attribution. This adaptation removes paged
addressing symmetrically and fixes head batching symmetrically. It explicitly
guards the source's uninitialized empty-partial boundary.

The collected A3 evidence is in
[the compact numeric record](data/optimization-wave003-split-kv.json). All six
oracle payloads were generated once. Twelve side correctness checks passed;
eight captures contain 80 full API forwards. Both guards passed with zero
captures. All 16 arm/stage specializations compiled before scientific execution.
The observed device product was Ascend910B4-1. No source versions failed and no
scientific workload was retried.

| Case | Hypothesis | Before span µs, mean ± sample SD | After span µs, mean ± sample SD |
| --- | --- | --- | --- |
| F07-01 | benefit | 58.104 ± 1.047 | 366.474 ± 0.850 |
| F07-02 | cost | 19.956 ± 0.400 | 146.066 ± 7.446 |
| F07-03 | benefit | 85.174 ± 0.561 | 1045.468 ± 1569.358 |
| F07-04 | cost | 16.904 ± 0.860 | 149.850 ± 9.193 |

Every measured after span was larger in this capture. This result includes
the second launch and its gap, as required by the complete API contract. The
F07-03 after samples include a 5511.94 µs span, retained without filtering or
rerun; its large sample SD limits interpretation. Prespecified benefit labels
are hypotheses, and this collection provides negative evidence for them under
the measured implementation and conditions. Active time and gap samples remain
separate in the numeric record. No uninstrumented latency study was performed.
