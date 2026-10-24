# Fixed numerical decomposition for a row mean

The candidate assigns each row to one output owner and accumulates consecutive
1024-element block sums in a fixed order. Unrelated batch rows do not change
that row's arithmetic decomposition. It is an independent adaptation of
`deterministic/mean_batch_invariant.py` in triton-ascend-kernels at
`7acb9c40a6ca827e885d902b7eaee5d45457b230`, Apache-2.0, with vLLM contributor
and Huawei notices preserved. Exact source:
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/deterministic/mean_batch_invariant.py:L9-L59`.
The retrieved source test checks closeness; this study independently checks
cross-batch bit patterns. Source presence does not establish target support.

The controlled educational baseline splits each row into 512-element chunks
when the batch has at most two rows and 1024-element chunks otherwise, writes
partial sums, and launches a final reduction. It is a project contrast, without
a vendor-best claim. The candidate uses this format:

```python
acc = 0.0
for start in range(0, N, 1024):
    j = start + tl.arange(0, 1024)
    acc += tl.sum(tl.load(X + row * N + j, j < N, 0), 0)
tl.store(Y + row, acc / N)
```

Use this pattern when a row's reduction tree changes with unrelated batch
shape and the application requires batch-context reproducibility. The competing
cost is reduced parallelism along a long row. Removing partial storage and a
second launch may help short rows. Neither accuracy nor timing improvement is
promised. The baseline can also happen to produce identical context results.

The public pair accepts contiguous FP32 matrices, `dim=-1` or `1`, and boolean
`keepdim`, with positive sizes and bounded finite inputs. It returns a tuple
containing one output tensor. The cancellation guard explicitly returns three
full context outputs for batches of 1, 3, and 129 rows; each guard API computes
three means. Their identical designated input row appears at positions 0, 1,
and 128. Only the candidate is required to return bitwise-identical designated
means. All output rows still meet the independent CPU FP64 reference gate
`atol=2e-5, rtol=2e-4`. This is a last-axis property; reduction over the batch
axis is outside the contract. Inputs remain immutable and outputs do not alias.

All required partial sums and final reductions execute inside each API call.
Buffers are preallocated for both arms: logical partial-workspace savings do
not establish peak allocation savings. Correctness runs once per arm and case,
then each measured arm executes exactly ten profiled API forwards without
input/output inspection, warmups, autotuning, or extra timing. Guards have zero
captures. The six prospective cases are frozen in
`experiments/optimization-wave005/T02.json`.

Inspect the exact task count, supported vector/MTE fields and all samples when
comparing the two formats. The primary boundary is the first device operation
through the last, including interkernel gaps, while profiling is active. Active
compute union and the remaining gap are separate; the gap is not all-device
idle. These spans are diagnostic rather than ordinary application latency.
No physical traffic or roofline estimate follows from busy percentages.

Results and target/compiler/binary bindings are retained in the accompanying
`batch-mean-summary.json` and compressed `batch-mean-evidence.json.gz` reference
files. This is an implementation study, with no automatic new-technique claim.
