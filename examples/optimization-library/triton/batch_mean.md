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

## Recorded results

Qualified on gz-a3, actual Ascend 910B4-1 physical die 4 / logical device 0,
CANN 9.1, Torch 2.10 / Torch-NPU 2.10 and imported Triton 3.2.
All six CPU payloads were cached once. All twelve API correctness checks passed,
then eighty profiled forwards produced eight complete captures and 120 physical
device tasks. The two guards made zero captures. GA performs three component
means per correctness API; the six component means are included in its two
API calls, rather than hidden repetitions of measured cases.

| R × N | Before span µs (sample SD) | After span µs (sample SD) | Span ratio | Active union before → after µs |
| --- | ---: | ---: | ---: | ---: |
| 128 × 512 | 152.034 (8.741) | 7.922 (0.092) | 19.191× | 14.734 → 7.922 |
| 1 × 16384 | 131.038 (9.709) | 5.880 (0.089) | 22.285× | 3.980 → 5.880 |
| 96 × 768 | 159.286 (18.523) | 5.530 (0.149) | 28.804× | 10.856 → 5.530 |
| 2 × 12288 | 153.678 (16.220) | 4.938 (0.180) | 31.122× | 5.012 → 4.938 |

The geometric mean of these four diagnostic span ratios is 24.883×.
The baseline gaps dominate these large ratios. Active compute reductions are
smaller; no ordinary latency or physical-traffic speedup is established. All
ten samples and their sample deviations are retained without outlier removal.
Both prospective cost cases also favored the candidate in this measurement.

The cancellation guard returned the same designated-row FP32 bits (`847249408`)
in all three contexts for **both** implementations. This does not demonstrate
baseline batch instability or a unique numerical advantage. It does verify the
candidate’s prospective bitwise property for these fixed contexts, and all
context rows passed the independent FP64 error gate. No seeds, shapes or
tolerances were changed to create a favorable counterexample.
