# Certified TopP prefix (F04)

This standalone finite-FP16 example returns original-index support, count, and
an ordinary dispatch tag (`0=full_sort`, `1=prefix`). Sampling and filtered logits
are excluded. Tied score indices may occur in any order.

BEFORE is a controlled, pedagogical complete-row scalar heapsort baseline.
It does not represent vendor or Torch's best full-sort implementation, and its
performance cannot be compared with the F03 vector-sort example as an isolated
algorithm comparison. AFTER uses a bounded minimum heap to select the largest
prefix, orders that prefix, and accepts only a certified strict crossing.
Both heaps occupy global memory and use scalar global-memory accesses. The
smaller prefix has a smaller active heap footprint; cache residency and heap
operation count are parts of this implementation's tradeoff.

Both sides compute FP32 exponentials after subtracting the row maximum, scan
them in input order for the FP32 full-vocabulary denominator, divide in FP32,
and accumulate ordered probabilities from left to right in FP32. The selected
support ends at the first cumulative probability strictly greater than FP32 p.
The candidate requires both `p-previous_mass>1e-5` and
`crossing_mass-p>1e-5`. Failure to cross within the bounded prefix, or failure
of either margin, invokes the identical complete-row baseline heapsort and
crossing routine on the device. No output is inspected by the host to decide
whether to fall back. This is a specified FP32 boundary rule, without a claim
of universal FP64 support-boundary agreement.

All normalization, workspace/index initialization, heap comparisons and swaps,
selection, ordering, certification, fallback, scalar-cache publication and
returned dispatch occur inside the single measured native kernel. Buffers are
allocated before correctness. Row workspace/output strides are padded to 128
bytes so scalar cache writeback cannot overwrite another core's row metadata.
The source guards excess AIV blocks and AIC execution before any tensor I/O.

The independently implemented heap ordering is motivated by the source-backed
N04 mechanism, not copied from vendor heap code. The filtered mechanism source
is `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-nn#index/top_k_top_p_sample/op_kernel/top_k_top_p_sample.h:L108-L153`
and `:L659-L673`, librarian query `4f2ef93afa0c4ad1a650a9c4faf5cc53`.
There is no F03 source dependency or direct raw-library access.

Compile both arms without launching with
`python top_p_prefix_compile.py --output DIRECTORY` in the selected A3 runtime.
Set `TOP_P_PREFIX_DIRECTORY` to that immutable build directory and load
`top_p_prefix_adapter:factory` with the example directory on PYTHONPATH.
The selectors are `wave003_topp_before` and `wave003_topp_after`.
The compiler retains exact command, source, compiler and binary hashes.

The case manifest freezes the four approved matrix cases and two correctness-only
guards. CPU oracle payloads are cached once. Each native side receives one
correctness forward per case and source version, followed by exactly ten full
profile forwards for each main case. Guards have zero captures. Capture forwards
perform no validation, input fingerprinting, host output inspection or oracle
work. Whole-device operation span includes all work and gaps; active duration
and gaps remain separate diagnostic quantities. PMU busy ratios do not establish
physical traffic or roofline performance.

## Retained A3 result

Candidate v2 passed the four fixed profile pairs and both guard pairs on
Ascend910B4-1 (20 Cube/40 Vector cores), CANN SDK directory label 9.1.0,
Torch 2.10.0+cpu and Torch-NPU 2.10.0. Each main arm was checked once and then
captured for ten forwards: eight captures and 80 profile forwards total.
The two guard pairs produced four correctness forwards and zero captures.
The near-p guard returned `full_sort` and matched the baseline output exactly.

| Case | Before span (us) | After span (us) | Before / after | Candidate dispatch |
|---|---:|---:|---:|---|
| F04-01, gallery benefit | 3939.768 | 1880.952 | 2.095 | prefix |
| F04-02, gallery cost | 1044.964 | 1178.018 | 0.887 | full_sort |
| F04-03, heldout benefit | 6041.658 | 2842.130 | 2.126 | prefix |
| F04-04, heldout cost | 2082.166 | 2325.068 | 0.896 | full_sort |

These means describe profiled device-operation spans for this controlled
scalar-heap implementation. Diffuse controls retain the cost of speculative
selection followed by full fallback. The full-sort baseline is deliberately
pedagogical; these ratios do not compare against a production vector sorter.

V1's first baseline correctness forward returned an invalid count and stopped
with zero captures. V2 explicitly publishes scalar-cache stores and separates
row outputs into 128-byte regions. Both source versions, binaries, compiler
manifests, the failed result, original profiler reports and compact PMU rows
remain retained. All six CPU oracles were created once in v1 and reused by v2.
The machine-neutral evidence JSON beside this document records source/binary
hashes, all ten spans per side, separate active/gap values and original PMU
columns. Measured sources were frozen before v2 timing and were not changed.
