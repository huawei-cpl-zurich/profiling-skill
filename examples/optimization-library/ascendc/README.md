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
