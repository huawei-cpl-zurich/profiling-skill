# F02: hierarchical triangular inverse

This narrow Triton example compares full-chunk forward substitution with a
hierarchy of independent 16-row diagonal solves. Each merge computes
`X21 = -X22 @ A21 @ X11`, retains the two diagonal inverses, and writes zero
to the upper-right block. BT32/64/128 use one/two/three merge levels.

The mechanism follows the filtered source reference
`ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/solve_tril.py:L1-L240`.
That reference attributes Huawei Technologies, vLLM contributors, Songlin Yang
and Yu Zhang / flash-linear-attention. This example independently implements the
block equation using direct pointer loads/stores. It avoids the source's
`insert_slice` extension, which is absent from the measured runtime. It retains
FP32 IEEE matrix products rather than converting inverse state to FP16.

Inputs have shape `[B,T,H,BT]`, FP16 dtype, strictly lower-triangular chunk
blocks, and each row's absolute sum at most 0.25. Optional shared sequence
boundaries partition each batch's T dimension; chunks never cross a boundary.
Output has the same shape with FP32 dtype, exact unit diagonal, zero upper
triangle, and zero columns beyond a tail chunk's extent. Unsupported BT is
rejected before device allocations. Supported BT is 16, 32, 64 or 128.

The independent oracle solves `(I+A) X = I` in CPU FP64. Acceptance requires
`atol=rtol=1e-3` and infinity-norm residual at most `2e-3`, plus exact structure.
The frozen gallery/heldout cases and two guards live under
`examples/optimization-library/cases/wave003/F02*.json`. Their initial matrix
has SHA256 `66a1aa6d957727edd8172939a4decd74cabfe8591b3cba83f7c6e3eb9dee6890`.

The before arm launches a full-BT column solve and unpack. The after arm
launches a 16-row leaf solve, two product/materialization kernels per merge,
and unpack. It includes all merge products and output preparation in the
forward. Workspace allocation and immutable input/table transfers happen once
in adapter preparation, which launches no kernel. All read workspace footprints
are initialized within each forward. The after arm reserves three FP32 dense
chunk buffers; before reserves one. Preparation also allocates the common
output and chunk descriptor tables. No autotuning or warmup launches occur.

A3 compile qualification uses `JIT.warmup` with mock tensor arguments, never a
kernel launch. Compilation alone establishes no device correctness. The study
uses exactly one correctness forward per arm/case followed by ten full
profiled forwards; guard checks have no captures. Profiled event latency is
reported as such, alongside the full operation span, active union and device
gaps. Exported task PMU values are diagnostic; missing counters remain
unavailable, and no physical traffic or roofline claim follows from them.

Run the CPU functional checks from a task worktree:

```bash
python -m unittest discover -s tests -p test_optimization_triangular_inverse.py -v
```

## Retained A3 study

The fixed study completed on Ascend910B4-1 (20 Cube / 40 Vector cores), using
Triton import 3.2.0, Torch2.10.0+cpu and Torch-NPU2.10.0. Six compile-only
FP32 dot probes passed; all eight case correctness checks and four logical
guard checks passed. There were exactly eight captures / 80 profiled forwards,
two tail-guard device forwards and two unsupported-BT rejections. The largest
observed residual was below `1.7e-8`; every output retained exact structure.

| Case | Mean full span before, us | Mean full span after, us | Before/after |
|---|---:|---:|---:|
| Gallery benefit, BT64 | 3786.652 ± 33.605 | 1111.918 ± 44.715 | 3.405x |
| Gallery cost, BT32 | 384.694 ± 11.149 | 571.160 ± 39.248 | 0.674x |
| Heldout benefit, BT128 | 11880.610 ± 73.116 | 1701.150 ± 24.861 | 6.984x |
| Heldout cost, BT64 tail | 1301.038 ± 38.557 | 1200.300 ± 657.514 | 1.084x |

The ± values are sample standard deviations over ten observations. The tail
case retains an after sample of 3048.26us. Its profiled event means were
1.346260ms before and 1.575250ms after, so the two measures disagree on the
direction of its mean change. No observation was discarded or replayed.

The BT32 after arm reduced active task union from 384.672us to 233.092us but
added 338.068us of mean gaps between tasks. The tail after arm likewise had
363.240us active union and 837.060us gaps. Thus a shorter compute chain alone
does not establish a faster complete operation. The larger cases retain clear
full-span reductions. PMU rows include Cube activity for merge kernels and
Vector activity for leaf solves; raw task metrics remain separate from
whole-device utilization or any physical throughput claim.

[The portable result file](optimization-wave003-triangular-inverse-results.json)
retains individual event/span/active/gap samples, kernel task accounting,
source/binary identities, numerical diagnostics and evidence handles. The
compact evidence archive has SHA256
`95e80f9c753162c4e4888ff79725675a49bde27f3e8caa38d6176f40cd7052b8`.
It retains 30 actual compiled binary variants, CSV rows and accounting;
complete vendor reports and the immutable frozen source remain on the target.
