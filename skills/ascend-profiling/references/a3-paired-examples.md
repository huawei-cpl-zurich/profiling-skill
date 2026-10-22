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
3. Capture PipeUtilization for each exact selector with launch count 1. Inspect
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

The separately collected [A2 worked series](a2-paired-examples.md) uses these
same source controls on Ascend910B4-1. Its timing/counter evidence remains
explicitly A2; it does not complete or replace this A3 series.

## Retained observations: an explicitly incomplete measurement series

[Measured-pair evidence](a3-measured-pairs.json) retains all seven accepted
exact-selector BasicInfo timing samples, successful correctness for all five
specializations, discovery records, compiler/source hashes and capture order.
The capture date was 2026-10-22 and the product was Ascend910_9382. Timing payloads
record Torch 2.10.0+cpu, Torch-NPU 2.10.0, Triton 3.5.0 and Triton-Ascend 3.2.2.
This is **not a completed three-repeat study**: the device ceased to satisfy
idle requirements before the remaining repetitions, pipe/conflict groups and
calibration sweep. Those missing results are `null`, not zero or passing.

| Control | Accepted ordinary task-duration samples, μs | What these samples establish |
| --- | --- | --- |
| Direct contiguous copy, N 1048576 | 20.760416 | One observation of the direct baseline; also the shared identity-gather baseline because its specialization/input contract is identical. |
| Stride 2 copy, same N and allocation | 873.217468,881.017639 | Two observed durations for the changed access pattern. Useful read/write bytes remain8 MiB; physical traffic and instruction-path cause were not measured. |
| Runtime identity-index gather, same N | 476.489532,478.649567 | Two observations of the semantically equivalent indexed path, with 4 MiB additional logical index reads. No late memory IR or pipe capture establishes scalarization in this run. |
| Matrix 512³, requested stages 1 | 36.680733 | One timing observation bound to the single-buffer binary. |
| Matrix 512³, requested stages 2 | 34.240685 | One timing observation bound to the double-buffer binary; insufficient repetition for a stable speedup claim. |

Every accepted timing capture used kernel replay, warmup 3, launch count 1 and one
application launch. BasicInfo discovery was a separate wider-window operation.
The common exported names (`paired_memory` or `paired_matmul`) do not prove
binary identity: specialization and compiled hashes distinguish variants.

The copy/gather observations motivate a transfer/addressing-path investigation,
but latency alone does not identify scalar instructions, DRAM bandwidth or
saturation. The correct next capture is each variant's late memory IR and
PipeUtilization, followed by conflict groups only for a specific hypothesis.
The direct rate from one sample is descriptive, not a calibrated envelope for
the strided or arbitrary-index contract.

The matrix pair supports a stronger **compiler** conclusion. Its diagnostic
late-IR captures produced byte-identical ordinary/debug binaries for each
variant. [Complete late dumps and annotations](a3-synchronization-inspection.md#real-matched-matrix-example-flags-and-buffer-slots)
show input cbuf slots growing 8192 → 16384 bytes, modulo 2 slot/event selection,
MTE2↔MTE1 dependency flags, macro-internal synchronization operands and FIXPIPE
output. Those facts establish the buffering transformation, not its dynamic
critical-path benefit. Without matched pipe/wait evidence and completed timing
repetitions, do not attribute the observed duration difference to eliminated
waits or claim measured overlap.

No empirical size curve was collected. The
[envelope reference](a3-performance-envelope.md) provides explicit formulas and
a runnable sweep procedure; it does not supply a measured device ceiling or
invent missing calibration points.
