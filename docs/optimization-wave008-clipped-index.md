# Clipped SwiGLU forward addressing

This source-backed component study compares a conventional flattened divmod
address calculation with the exact MindSpeed 2D forward kernel at revision
`4efe2b675c2d51517c3fdd14c721020b8d41e241`. Both use the pinned FP32 block picker
and the same tile element count, clamp/sigmoid expression, alpha 1.702, limit 7,
and compiler flags. It introduces no saturation or other pointwise math change.
MindSpeed's Apache 2.0 license and Huawei attribution are retained.

The baseline enumerates output `j`, then derives `row=j//K` and `col=j%K`.
The candidate uses independent row and column program axes. Input storage is
contiguous `GUP[M,2K]`, with gate at `row*2K+col` and up at gate+K. Output
storage is contiguous `OUT[M,K]`. The source 2D grid and flat grid can have
different program counts and padding. Timing differences therefore include
layout and scheduling effects; source syntax alone establishes no instruction
count reduction.

The independent CPU provider uses finite FP32 inputs sampled in dyadic 1/64
steps within [-12,12], plus explicitly specified clamp-boundary neighbors and
signed-zero guards. It computes the unchanged expression in FP64 once per
frozen case. Every output must satisfy fixed `atol=3e-5, rtol=2e-5`, and the two
arms must also produce identical FP32 output bits. Preparation owns separate
reusable input/output storage; canaries, immutability and nonalias checks run
outside profiling. Allocation latency is excluded.

Integer admission considers every padded row/column/flat intermediate and byte
offset, including masked lanes. Both arms choose the same int64 address route
when padded byte bounds exceed signed int32. Unsupported integer intermediates
are rejected without allocation. Symbolic nonfrozen tests cover wide layouts
without large allocations.

The frozen matrix in `experiments/optimization-wave008/T05.json` retains four
main cases and two guards. Its v2 clarification corrects an unexecuted textual
BM4 reference: the actual source picker selects BM2/BK512 for M3/K65. All
input-only compile/resource/codegen probes precede the six CPU fixture payloads,
and all twelve correctness API calls precede the eight ten-call captures.
The input-only probe creates no fake expected fixture and executes no kernel.
Public first launch also qualifies compilation before execution, then reuses the
compiled artifact.

The standalone `clipped_index_cli.py` is explicitly exploratory. Its metadata
admission mode supports flat or nested case parameters and preserves role
metadata without creating cache/output directories. It does not claim the
complete frozen collector qualification.

The frozen native A3 collection passed all six FP64 and paired-bit checks and
all compile/resource/codegen screens. All six candidate binaries have distinct
selected code. The inventory audit passed the original coverage receipt.

| Shape M×K | Flat grid | Source 2D grid | BM×BK | Flat span (µs) | Source span (µs) |
| --- | --- | --- | --- | ---: | ---: |
| 128×3072 | 96 | 32×3 | 4×1024 | 299.820 | 6.414 |
| 33×1024 | 9 | 9×1 | 4×1024 | 145.097 | 3.710 |
| 256×4000 | 250 | 64×4 | 4×1024 | 905.426 | 13.202 |
| 129×65 | 3 | 17×1 | 8×512 | 156.867 | 4.378 |

These are all-ten-sample profiled means, including first samples. Each API has
one device kernel, so its task work and operation span coincide; event samples
remain separately retained. The candidate uses more programs on the last two
cases and still has lower profiled duration in this collection. This evidence
supports the measured addressing/layout comparison on the observed runtime;
it does not identify instruction counts or establish ordinary latency.

`T05-results.json` retains every event/task/span sample, grids, padded integer
bounds, fixed-gate errors, paired-bit results, fixture/source/binary/trace hashes,
and selected-code receipts. The verified compact archive also retains all raw
supported PMU fields, compiler resources/cache artifacts and CPU payloads. Full
vendor trees remain remotely retained. The original queued GZ source transfers
were never dispatched for science; this was the first acquisition of this
study on a separately verified capable native A3 target.
