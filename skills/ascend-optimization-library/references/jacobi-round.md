# Schedule one Jacobi round

Use this pattern when disjoint Jacobi pivot pairs are submitted serially.
The [source scheduler](ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-solver#src/cheevj/kernel/cheevj_pair_schedule.hpp:L34-L116)
rotates a ring with a dummy slot for odd block counts. Its
[serial stage loop](ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-solver#src/cheevj/kernel/cheevj_block_jacobi.hpp:L146-L165)
motivates grouping independent pairs. This example adapts source block sizes
16/32 to scalar blocks. It implements exactly one prescribed round, without
an eigensolver, convergence claim, or eigenvector interface.

[jacobi_round.py](../../../examples/optimization-library/triton/jacobi_round.py)
exposes `jacobi_round:factory` and a reusable `Stage` primitive. The
[case matrix](../../../experiments/optimization-wave004/W09.json) fixes four
measured cases and two guards. Pass an individual nested case object to
`scripts/optimization_pair.py --case`; place the Triton example directory on
`PYTHONPATH`. The independent contract builds the dense FP64 transform
`J.T @ A @ J`, with each disjoint block `[[c,s],[-s,c]]`.

The serial arm computes angles, copies the input, then updates each pair's
rows and columns in separate launches. The parallel arm computes identical
angles from the original input, applies all left updates into scratch, then
all right updates into the output. The launch boundary prevents intersecting
left/right updates from racing. Other disjoint rotations leave a pair's pivot
block unchanged, so computing its angle from the input preserves the serial
transformation. Copies, angles and all updates execute inside every API call.
Only the shape-derived pairing metadata and storage allocation are prepared.

The domain is bounded symmetric contiguous FP32 matrices with 2 ≤ N ≤ 128,
scalar blocks, and a valid round. Missing, overlapping or duplicate prescribed
pairs fail on the host before device imports. An odd-size singleton uses the
identity rotation. Both arms overwrite a NaN-initialized output and preserve
the input. Comparisons use atol 2e-5 and rtol 2e-4, with independent Frobenius
norm, symmetry and trace checks. This domain does not establish unrestricted
symmetric-matrix or full-solver accuracy.

The main hypothesis is fewer launches and better parallel ownership for the
large cases. Scratch traffic and staging can cost more for small cases. Use
all ten device-operation spans, including inter-kernel gaps; active time is a
separate quantity. Profiled spans do not establish application latency,
physical memory traffic or a roofline result. Serial pair indices remain
runtime arguments: compile-only probes at indices zero and one must share
binary identities. Both arms use `num_stages=1`, `multibuffer=False` and
FP32 arithmetic with fusion disabled; these settings document this experiment
rather than a universal compiler limitation.

The [retained evidence](jacobi-round-evidence.json) records four successful A3
pairs, all ten samples per arm, unchanged compilation identities, PMU samples,
and both guards. Mean profiled device-operation spans in microseconds were:

| N | Serial pairs | Staged round |
| --- | ---: | ---: |
| 128 | 18068.078 | 423.518 |
| 4 | 772.494 | 363.804 |
| 96 | 14121.524 | 397.516 |
| 6 | 1059.952 | 368.526 |

All four measured cases favored the staged arm at this instrumented boundary;
the proposed small-case cost did not appear here. The comparison includes
Python-dispatch gaps between kernels and does not predict a fused native
implementation. No sample was dropped. The scalar-block launch schedule is
especially expensive in the serial arm. Maximum observed absolute numerical
error was below 3e-7. The odd N=5, round=1 guard passed on both arms without
captures; the duplicate-map guard rejected both arms with zero NPU calls.
An initial compile-only attempt used an unsupported option name and executed
zero scientific forwards. Its source and failure remain in the retained archive.
