# Reuse tridiagonal factors across right-hand sides

Recognize a tridiagonal matrix shared by many independent RHS columns. Factoring
A once per API and sharing the read-only factors removes repeated factor work;
an additional launch and global factor reads can outweigh that saving.

The independent [Thomas pair](../../../examples/optimization-library/triton/shared_tridiagonal.py)
uses finite FP32 coefficients and column-major B represented by contiguous
`[R,ldb]` storage. Both arms preserve A/B and padding, return separate storage,
and reset output inside every API. The before arm factors per RHS in its fused
solve; the after arm rebuilds shared factors per full API. RHS ownership,
division order and FP32 fusion settings match. No factors are cached across
forwards. Strict row diagonal dominance is the sufficient no-pivot guard;
otherwise a real CPU dense solve handles correctness-only fallback.

Choose **avoid** for this implementation on the four measured 910B4-1 cases.
Full profiled mean operation spans increased from 806.704 to 1124.860µs
(m512/R64), 225.128 to 397.582µs (m128/R1), 703.068 to 858.306µs
(m768/R48), and 208.096 to 375.178µs (m64/R2). These are conditional negative
results, not a universal dispatch threshold. Evidence is **insufficient** for
other shapes, targets, or a tuned/native solver.

The [frozen matrix](../../../examples/optimization-library/cases/wave004/W11.json)
fixes four pairs and two correctness-only guards. The
[implementation notes](../../../docs/optimization-wave004-shared-tridiagonal.md)
qualify the host-source ownership adaptation and unresolved device-helper miss.
The [compact evidence](../../../docs/data/optimization-w004-shared-tridiagonal.json)
retains all ten samples per arm, residuals, padding checks, hashes, PMU fields
and the operational collector negative. Compare complete operation spans,
including copy/factor/solve and gaps, with active duration separately. Do not
infer unprofiled application latency, physical traffic or roofline behavior.
