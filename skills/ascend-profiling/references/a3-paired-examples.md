# A3 paired controls: a runnable source-to-evidence workflow

`examples/a3_paired_probes.py` provides three small controls, not universal
optimization recipes. It requires Torch, Torch-NPU and compatible Triton-Ascend
for execution; its manifest uses only Python's standard library. Run commands
from the skill folder, using the device index valid in the configured runtime.

```bash
python3 examples/a3_paired_probes.py --manifest
python3 examples/a3_paired_probes.py --case gather --variant baseline \
  --size 1048576 --device 0 --artifacts /absolute/evidence/gather-direct-ir
python3 examples/a3_paired_probes.py --case gather --variant candidate \
  --size 1048576 --device 0 --artifacts /absolute/evidence/gather-indexed-ir
```

The script initializes and computes references on CPU, copies inputs to the
configured device, launches the selected kernel, checks its output, and prints
JSON. A correctness failure returns nonzero. It reports useful traffic,
allocation/span, requested options, package versions, source hash and retained
compiler-artifact hashes. `--iterations` repeats launches; it does not provide
wall-clock timing or manufacture a profiler selector. Discover actual exported
names using [two-pass acquisition](acquisition-workflow.md).

| Pair | Exactly what changes | Kept fixed / hypothesis |
| --- | --- | --- |
| `--case copy`, `baseline` → `candidate` | Affine input stride 1 → 2 | Same N fp32 reads/writes, output count, `2N` input allocation and block256. Accessed span changes; output values are validated against each access pattern. This is an access-pattern control, not a semantics-preserving rewrite. |
| `--case gather`, `baseline` → `candidate` | Direct affine read → runtime-loaded identity-index read | Same mathematical output, input values, N, allocation, and block256. Candidate adds `4N` index bytes and may scalarize; identity values do not promise DMA lowering. |
| `--case matmul`, `baseline` → `candidate` | Requested `num_stages=1` → `2` | Same fp16 inputs, fp32 output/accumulator, dimensions, `32×32×64` tile and grid. Confirm actual buffers/flags and resource fallback; no guaranteed speedup or overlap. |

`--size N` selects copy/gather element count; `--matrix-size N` selects square
GEMM dimensions. The default matrix size512 has eight K tiles. Tiny matrices
and one-program probes do not establish device-wide compute capacity.

## Compare in an order that answers the question

1. Run both variants normally and retain correctness and manifests. Verify that
   source/options/input contracts match the intended pair.
2. Discover exported names separately for both specializations. Collect at
   least three interleaved exact-selector BasicInfo captures per variant with
   the same warmup, launch count, replay mode and device conditions. Retain all
   declared samples and dispersion, including failures separately.
3. Capture PipeUtilization for each exact selector with launch count1. Inspect
   every block/sub-block/core type. Retain absolute active time and all available
   wait/stall fields alongside activity fractions.
4. Add a separate ResourceConflictRatio capture if supported and relevant.
   Missing fields or unsupported groups stay unavailable; do not turn them
   into zero. Instrumented durations are diagnostic, not speedup samples.
5. Inspect actual retained compiler stages. For gather, find index load and
   address/scalar loop versus contiguous DMA. For matrix, find buffer bases,
   rotation, flags and matrix/FIXPIPE output. Record missing late stages rather
   than calling early IR final code.

For timing after discovery:

```bash
python3 scripts/profile_a3.py --output /absolute/evidence/gather-direct-timing-1 \
  --kernel-name '<discovered exact direct kernel>' --warm-up 3 --launch-count 1 -- \
  python3 /absolute/skill/examples/a3_paired_probes.py \
  --case gather --variant baseline --size 1048576 --device 0
```

Repeat with a new output path and the discovered candidate selector; alternate
variant order. Use [the field glossary](a2-a3-msprof-op.md#field-glossary-and-calculations)
and [late synchronization inspection](a3-synchronization-inspection.md) to
connect observations without turning correlation into critical-path proof.

## Read an outcome without forcing the expected story

For the stride pair, equal useful bytes with larger span can reduce transfer
efficiency; high MTE2 fraction alone cannot identify physical byte volume.
For the identity gather pair, lower MTE2 and higher scalar time can mean a
less efficient memory instruction path, not less memory work. For the matrix
pair, changed buffers/flags with equal Cube-active time can reduce other waits,
startup/drain, or movement scheduling; do not require scalar_wait_time to move.

If the candidate is slower, keep that result. If timings overlap their
dispersion, report the mechanism observation without a precise speedup. Sweep
sizes and compare [empirical envelopes](a3-performance-envelope.md) before
extrapolating a small-shape result.
