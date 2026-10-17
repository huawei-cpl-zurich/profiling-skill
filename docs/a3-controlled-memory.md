# Controlled MTE2 activity and useful throughput

This experiment separates two controls that the historical stride sweep
confounded. It adds new entrypoints; historical memory cases are unchanged.

At the default size 1,048,576, `count/<stride>/<size>` always reads that many
unique FP32 elements from one 64×size allocation. `span/<stride>/<size>` uses
one size-element allocation and reads `ceil(size/stride)` elements. Strides
are 1, 2, 4, 8, 16, 32, 64. The last accessed address is explicitly reported;
the fixed span means a fixed bound, not identical last addresses for every
stride. Grid, useful bytes and output bytes change in the span sweep.

`indexed/{contiguous,local32,global}/<size>` reads a runtime-loaded int32 index
tensor; it shares the count sweep's 64×size allocation. Compare it with
`count/1/<size>`, not the smaller allocation in `span/1/<size>`. All indexed
patterns are permutations, so distinct input bytes and output bytes match.
Index reads are extra useful traffic and must remain explicit. CPU-generated
bounded FP32 values provide an exact CPU gather reference before profiling.

## Run

Use `$remote-access` to stage a self-contained repository bundle and run the
commands inside the selected named Triton runtime, never directly over SSH.
The script selects an eligible idle physical device using the existing lease
helper and exposes logical device zero to the application. Keep the returned
remote handle and observe that same handle after transport interruptions.

```bash
# CPU-only manifest: no Triton import or device needed.
python benchmarks/a3_controlled_memory.py --size 4096 --manifest

# Fail-fast single-case gate, then the three-case correctness/profiler pilot.
python scripts/a3_controlled_memory_batch.py --output <fresh-single> \
  --size 4096 --case count/1/4096
python scripts/a3_controlled_memory_batch.py --output <fresh-pilot> --size 4096 --pilot

# Example matched pair: keep both cases together on the same device.
python scripts/a3_controlled_memory_batch.py --output <fresh-comparison> \
  --case count/1/1048576 --case indexed/contiguous/1048576
```

Without `--case`, all 17 cases run. Independent comparison groups may run
concurrently using the existing device lease, but include the relevant direct
baseline within each group rather than comparing across devices.

The runner verifies all selected cases and discovers their exact BasicInfo
kernel names before measured captures. It interleaves cases round by round,
rotating the first case each round: five BasicInfo, three PipeUtilization, then
three Memory captures. Every capture retains its command, stdout, stderr,
exit status, raw report tree and compact rows. BasicInfo samples are the
declared timing source; diagnostic durations are not substituted. Source
hashes, footprint, selector, environment returned by correctness, capture
order, physical device and capture timestamp bind the results.

## Failure, resume and interpretation

Any failed correctness, ambiguous selector, partial collection or missing
required export stops the batch. A named rejection of the Memory group is an
explicit unsupported result, not zero bandwidth. Other Memory failures remain
failures. Unknown profiler versions are not an execution failure: capture the
available version/help and product inventory in the retained job separately.

`--resume` reuses only completed capture boundaries with unchanged source,
case configuration, commands and evidence file hashes. Incomplete capture
folders stop with an actionable path; inspect them before an explicit retry.
Do not launch a second remote job while the original handle may still run.
A deliberate retry must preserve the incomplete folder under a new name and
be recorded by the coordinator; it is not automatic checkpoint recovery.

`useful_input_GB_per_s` is useful input bytes divided by whole-kernel median
microseconds, in decimal GB/s. It excludes index reads and output writes,
which remain separate footprint fields. It is not physical memory bandwidth.
Memory CSV columns are retained verbatim with unvalidated units/denominators;
validate product/version-specific semantics before converting counters to
bytes. Do not divide device-wide bytes by one core's active time or treat a
high MTE2 activity ratio as proof of efficient bandwidth. Separate replays do
not establish overlap. Retain noisy samples rather than silently discarding
them; the existing summary marks range/median above 5% as low confidence.

Compiler capture through `$mlir-triton-dump` is a separate run: forced compile
and debug capture must not change timing conditions. Bind that capture to the
same case, specialization and source before interpreting scalar loops versus
copy operations.
