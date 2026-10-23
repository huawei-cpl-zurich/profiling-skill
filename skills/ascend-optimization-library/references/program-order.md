# Change output-tile assignment without changing reduction order

Status: source-grounded hypothesis; paired A3 evidence pending. Family `12` is
implemented in `examples/optimization-library/triton/gemm.py` (`gemm:Adapter`).

The [pinned threshold-gated source mapping](ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/gemm/matmul.py:L100-L117)
motivates testing different output-tile assignments. The source has a suspicious
tuple assignment elsewhere; our mapping is independently implemented and
bijective rather than copying it as a verified operator.

Before, assign row-major tile coordinates. After, rotate the column within each
row; keep the K loop ascending and identical:

```python
row, column = divmod(program_id, columns)
# After only, when rows >= 2 and columns >= 2:
column = (column + row) % columns
# Both: for k_tile in range(ceildiv(K, BK)): accumulate(...)
```

For each row, addition modulo columns is a permutation; rows are unchanged.
This covers rectangular grids exactly once. Masks still handle matrix elements
outside the final M/N/K tiles. The CPU tests additionally enumerate real matrix
elements on a tail grid to detect duplicates and missing output coverage.

Recognize many output tiles reusing the same operand regions, with a plausible
cache or conflicting-access hypothesis. A changed assignment can alter which
operand tiles concurrent programs request, at the cost of extra mapping work
and possibly worse reuse of the other operand. Small grids and cache-resident
inputs can gain nothing. The guard prevents meaningless rotation of one-row
or one-column grids; it is a correctness/applicability guard, not a measured
crossover threshold.

The gallery proposes 4096×4096×2048 and 256×512×64; held-outs use fresh shapes.
Both arms use FP16 inputs/output, FP32 accumulation, and the same CPU oracle
converted to FP16 at rtol=atol=1e-3. Preserve all input strides and dispatch only
within the adapter's declared matrix/layout contract.

Compare ten owned complete-call timings and kernel identities. Attribute a cache
or traffic improvement only with supported evidence, not elapsed time alone.
Falsifiers include equal times within variability, slower rotation, or unchanged
measured traffic. This transformation is distinct from ShuffleK: it reorders
M/N program assignment, never the sequence of K accumulation.

No beneficial or adverse timing has yet been measured for this example.
