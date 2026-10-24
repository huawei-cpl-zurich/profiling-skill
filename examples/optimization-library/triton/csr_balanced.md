# Balance CSR nonzeros with segmented scans

This project-owned vector adaptation isolates work assignment from the CANN
`ops-sparse` SpMV implementation. The source assigns contiguous nonzero blocks,
searches CSR offsets for block boundaries, scans products and merges row
fragments. The source uses Cube-assisted scans; this example uses Triton vector
segmented associative scans and makes no Cube performance claim.

Sources, retrieved through the reference librarian query
`4f2ef93afa0c4ad1a650a9c4faf5cc53`:

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/spmv/arch22/kernels/spmv_cube_kernel.cpp`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-sparse#sparse/spmv/arch22/kernels/csr_gather_kernel.h`

Before owns one CSR row per Triton program and reduces its gathered products.
After first overwrites output with zeros, then assigns 256 consecutive nonzeros
per program. Each lane binary searches the original CSR offsets to identify its
row; a segmented scan accumulates products and atomically adds each row fragment.
The second kernel's dependency on zeroing is provided by launch order on the same
stream. Every forward includes both kernels, searches and fragment merging. No
CPU row map, cached product, output reset or device preprocessing is omitted.
Preparation preallocates output and transfers immutable inputs, as in both arms.
Neither arm allocates during forward. Allocation and input transfer latency are
outside this experiment's measurement.

The fixed contract is contiguous int32 CSR indices, finite FP32 values and X in
[-0.25,0.25], alpha=1, beta=0 and FP32 output. Repeated columns contribute
repeatedly. Initial output is NaN; beta=0 must overwrite it, including empty rows.
The independent FP64 CPU oracle uses atol=rtol=0.002. Output accumulation order
may differ, including nondeterministic atomic fragment addition. The optimized
path requires a certified actual maximum row length at most 1024; longer rows up
to 2048 use the same vector row baseline. The certificate is validated against the
original host CSR offsets before transfer. It is dispatch metadata, never a
per-nonzero row map. Empty matrices launch the zero kernel in both arms.

Uneven long rows can make per-row ownership finish late while other workers are
idle. Nnz partitioning can reduce this tail. Uniform rows already balance well;
additional output zeroing, binary search, scans and atomics may outweigh that
benefit. Examine the complete operation span and its inter-kernel gap alongside
per-task pipe composition. More kernels or shorter individual kernels alone do
not establish a speedup. Busy ratios do not measure physical traffic, bandwidth,
whole-device utilization or distance to a hardware roofline.

`csr_balanced-cases.json` contains the four prespecified gallery/held-out profile
cases and two correctness-only guards. With this directory on PYTHONPATH, use
`scripts/optimization_pair.py --adapter csr_balanced:factory` with one extracted
case specification and required output/cache/provenance arguments. Each ordinary
case checks each arm once, then captures exactly 10 full forwards per arm without
checking input or output during capture. Guard specs carry `role: guard` and
therefore run correctness only; `--correctness-only` also selects that mode
explicitly. Use a fresh output directory and preserve the oracle cache. Changing
source creates a new candidate version; it does not authorize repeating an
unchanged partially captured candidate. Retain failures and cost cases.
