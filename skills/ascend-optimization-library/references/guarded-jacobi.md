# Guarded forty-step parallel Jacobi

Recognize a single-RHS tridiagonal solve whose current coefficients admit a
strong contraction certificate. Parallel row updates remove serial dependence,
but repeated launch boundaries and global ping-pong storage may cost more.

The independent [adapter](../../../examples/optimization-library/triton/guarded_jacobi.py)
checks actual GPU coefficients/RHS on every API, requiring strict twofold row
dominance, bounded finite magnitudes and 128..6144 rows. It initializes zero,
runs exactly forty simultaneous Jacobi updates, and otherwise uses unchanged
serial Thomas arithmetic through GPU conditionals. Resets, copies, certificate,
all passes and fallback dispatch are measured; no host predicate sync or
cross-call factor caching occurs. Inputs and padding remain unchanged.

Choose **avoid** for this implementation on the four measured 910B4-1 cases.
Mean full operation spans increased from 1071.284 to 6764.956µs (m2048),
259.518 to 7148.896µs (m128), 776.850 to 7380.748µs (m1536), and 313.076 to
7208.280µs (m256). Evidence is **insufficient** for other shapes, targets or
fused/native implementations. Do not turn these negatives into a dispatch rule.

The [frozen cases](../../../examples/optimization-library/cases/wave004/W12.json),
[notes](../../../docs/optimization-wave004-guarded-jacobi.md) and
[compact evidence](../../../docs/data/optimization-w004-guarded-jacobi.json)
retain all forward/residual metrics, component guard tests, samples and hashes.
Exact-arithmetic `2^-40` truncation is not an FP32 proof; the fixed CPU FP64
forward gate decides accuracy. The impulse guard's componentwise backward error
reaches 1.0 in tiny/underflow rows even though its forward gate passes.

Provenance is SOURCE-DOCUMENTED: upstream README algorithm, architecture layout
and numerical test interfaces, with an independently written device body. The
native body was unseen; this makes no native wide-range robustness claim.
Compare full device-operation span including all stages/gaps, and keep active
duration separate. Profiled spans/PMU do not establish application latency,
physical traffic or whole-device roofline behavior.
