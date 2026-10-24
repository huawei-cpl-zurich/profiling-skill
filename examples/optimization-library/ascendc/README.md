# Native row DMA command aggregation

`row_dma.py:factory` implements the paired-runner interface for the immutable
`experiments/optimization-wave006/L04.json` matrix. Compile every numerical
case before correctness or profiling with `row_dma.compile_all(specs, output)`;
set `ROW_DMA_DIRECTORY` to that output directory. Run the public paired CLI
with a single nested case object from the matrix. Guard GB is a host-interface
check: call `prepare` for each arm and retain its pitch rejection separately,
before fixture creation. It has no numerical payload and no capture.

The kernel computes immutable contiguous FP32 `x[rows, columns] + 1` with one
AIV block. Both arms initialize all UB lanes to zero, wait for V→MTE2, copy
input, wait for MTE2→V, add one to all lanes, wait for V→MTE3, then copy each
valid row to GM with the same output commands and terminal barrier. The sole
change is one multi-block input command replacing the per-row input commands.
`DataCopyExtParams.blockLen` and GM source stride use bytes; the UB destination
stride uses 32-byte blocks. With contiguous GM rows and eight-FP32-aligned UB
pitch both strides are zero. The selected SDK declaration is retained alongside
the build logs, compiler identity, source hashes and binaries. Runtime cases
qualify this encoding for the installed A3 SDK; integer field widths alone do
not establish hardware limits.

Positive rows, columns 1..128 and at most 65536 UB bytes define the conservative
public study scope. The after arm aggregates up to 128 rows and uses the same
per-row fallback above 128. Neither bound claims a hardware maximum. Explicit
pitch overrides must equal `ceil(columns/8)*8`; invalid pitch is rejected before
device access or fixture caching. Output storage has 32 FP32 prefix and suffix
canaries, checked alongside exact bitwise output, input immutability and
nonaliasing once per arm against an independent CPU FP64 oracle rounded FP32.

Five numerical payloads cover four pairs plus row129 fallback. GB adds two
host checks. Main pairs capture exactly ten full API calls per arm, with no
warmup or extra timing, and guards capture none. The source command counts
(rows versus one) describe source expectations, not measured PMU instructions.
Retained profiler device spans and original pipe fields are diagnostics;
pipe percentages do not measure physical bytes, bandwidth or roofline.

The qualified build marks the entry `__attribute__((aiv))`. Before its first
forward, the retained audit checks every compiled entry exists in its vector
object and has no Cube entry. Device ELF metadata identifies `K_TYPE_AIV=3`;
the profiler reports one task per forward with no mixed block or Cube time.
The earlier mixed-placement run is retained as nonconforming evidence. The
corrected execution reuses its five cached CPU payloads without regeneration.

[Collected evidence](../../../experiments/optimization-wave006/L04-evidence.json)
retains all ten diagnostic samples per arm. Mean device-operation spans in
microseconds were 5.792→5.670 (128×65), 2.102→2.130 (1×65), 4.406→4.276
(96×97), and 1.460→1.450 (2×9). These small changes overlap sample variation
and establish no general performance benefit. Ordinary latency was not measured.

`twoprod.cpp` is an independent dense compensated-dot extraction of the guarded
and fast product bodies from `cann/ops-sparse` commit
`bfc7c19d83ce246b03c65dc916ed4753e5467cf3`,
`sparse/spmv/arch22/kernels/spmv_kernel.h` (CANN Open Software License 2.0;
retained in `CANN_LICENSE`). The helper retains its copyright and arithmetic.
Its vector/scalar ownership events are explicit in this adaptation. Dense
staging, row iteration and the diagnostic flag publication are local code,
not an upstream dense-dot implementation. Both arms stage each row and publish
the result plus a dispatch flag. The after arm freshly computes two maxima per
row on every call; CPU expected flags are used only by the validator.

The bounded interface is contiguous immutable FP32 rows (1..64), width 1..4096,
main data in [-4,4], plus the two declared finite-product guard distributions.
Sequential dot compensation, DS merge and final addition match between arms.
The independent represented-input FP64 dot uses atol=1e-6 and rtol=1e-6.
Correctness diagnostics retain output hashes for read-only paired agreement.
This does not establish accuracy across the full FP32 range.

`twoprod.compile_all` retains compiler intermediates and uses identical `-O2`
and `-ffp-contract=off` flags. Call `audit_placement` before any forward; it
requires a generated vector function, absence of a Cube function, and metadata
bound to the actual selected SDK's AIV enum. A generic AI_CORE profiler label
alone does not qualify this execution engine. Width and scratch bounds come
from the pinned reduction helper (256 floats, count <=4096).

N02 remains **unqualified** on the selected CANN 9.1/910B4-1 gate. The first
pair passed FP64 tolerance, dispatch, immutable-input and canary checks, but
the once-only FP32 output hashes differed. The v1 controller checked this
paired requirement after two captures; all 20 profiled forwards are retained
as nonconforming, and the remaining cases/guards are unattempted. No speedup
or comparative measurement is admitted. The cause is unestablished because
the selected tools could not expose decoded instructions. See N02-evidence.json
for the exact original source hashes and counters.

The published adapter now fails closed before profiling with an explicit
unqualified-preflight error. `--correctness-only` remains available. This
publication-only gate changes no measured kernel/helper/fixture arithmetic;
no new NPU runs or frozen CPU-oracle evaluations were made for that gate.
