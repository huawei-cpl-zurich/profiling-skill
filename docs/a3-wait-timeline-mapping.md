# A3 wait-placement qualification

The six paired groups in `scripts/a3_wait_timeline.py --manifest` reuse the
existing native synchronization control: MTE2→Vector and Vector→MTE3,
plus balanced/imbalanced inter-core modes 0 and 2. Default repetitions are
32; select `--rounds 1 --rounds 32` for a two-point repetition sweep using
the existing native API. Both placements use the same repetition count.
Each pair compares `overlap` with `early-wait`; useful work, flags and outputs
are unchanged within that pair. Do not compare different balance groups as
equivalent-work candidates.

Native controls allocate intra-core event IDs through their owning `TPipe`,
and release every allocated ID after its matching wait. The two independently
pending MTE2→Vector signals use distinct allocated IDs, as do the intra-core
output roles. Cross-core flag IDs 8/9 are unchanged: they are a separate
namespace. This follows the [TPipe synchronization contract](https://asc.gitcode.com/api/SIMD-API/basic_api/sync_control/intra_core_sync/TQueSync.html).
Earlier literal-ID captures remain historical observations of that exact binary;
they do not validate the corrected allocator-backed source. A new native canary
must verify the updated binary before accepting its hardware measurements.

## Native application canary

Stage the repository through `$remote-access`, select its named runtime, and
run from the registry-derived workspace. The default is hardware-only and
does not require a simulator model:

```bash
python scripts/a3_wait_timeline.py --family mte2-vector --canary \
  --rounds 1 --rounds 32 --output "$NEW_OUTPUT_DIRECTORY"
```

The worker compiles the native executable before acquiring one idle physical
device lease. Every application uses logical zero. It performs host numerical
verification of **both** variants before any capture, then requires one exact-
kernel BasicInfo hardware acquisition per variant. Correctness alone does not
pass the canary. A failing variant prevents profiling either member of that pair.

Omit `--canary` for both variants and five interleaved BasicInfo, three separate
PipeUtilization and three separate Memory replays per variant. Each ordinary
subprocess and each acquisition stage has a 300-second limit. The outer
optional simulator worker has a 620-second bound because it performs two sequential
bounded stages: native correctness and profiling. Timeout kills its process
group before releasing the lease. Rotate pair order between rounds, retain all
samples, and preserve low-confidence flags rather than selecting a convenient
repeat. Durations are **hardware `msprof op` diagnostic measurements**, not
uninstrumented event timings or phase timelines. An explicit unsupported Memory metric stops only that metric group;
compiler, correctness and runtime errors remain failures.

Select several `--family` values, or omit them to run all six pairs. A failed
pair is retained while independent pairs continue. Output directories must be
new: there is no implicit retry, overwrite or duplicate scheduling. Every
completed hardware phase retains its request, command, device mask, timestamp,
raw artifact location and hash-checked `completed.json` checkpoint. Transport
recovery and terminal infrastructure retries belong to the supervising
`$remote-access` invocation, not this measurement worker.

## Optional simulator evidence

Only add `--simulation --soc-version "$OBSERVED_SIMULATOR_SOC"` after observing
an installed compatible model. There is no guessed SoC. Optional simulation
runs **after** hardware acquisition; its errors, timeouts and clean-acquisition
status are retained separately and never invalidate or prevent completed
hardware results. No trace is fabricated when simulation fails. Hardware-only
results explicitly record simulation as `not_requested`.

The optional worker uses the native executable route rather than the historical
Python/Torch teardown path. No hidden Triton arguments or binary-config ABI are
copied or guessed. A simulator failure is not a passed timeline acquisition
even if partial traces exist; a successful hardware canary is not a simulator
capability claim.

## What the evidence does and does not establish

The build record binds source, executable and compiler artifacts. Each selected
timeline requires the profiler's clean terminal marker and exact kernel binding.
`instruction_evidence` verifies the parser's original trace hashes and retains
**all** normalized control events, including controls beyond the normalizer's
256-event display sample. PC, core, raw lane, lane provenance, instruction
detail and timestamp provenance remain attached. Operand text mentioning MTE2
does not relabel a scalar/barrier instruction as transfer activity.

Source-controlled early/late placement is not an independently verified PC→source
map. Inspect decoded native debug/disassembly evidence before attributing a
specific wait PC to a source flag; an unavailable decoder leaves that attribution
unavailable. Different control PCs or event durations alone are not a source
mapping. Likewise, emitted waits do not establish how long another engine was
idle or whether the pipeline was saturated.

The normalizer's within-core pipeline interval unions and overlaps come from one
capture and its simulator clock. Cross-core overlap remains unavailable without
independent proof of a common clock. Never align different replays or treat
simulator timestamps as hardware latency, hardware utilization or a performance
gain. Report paired device BasicInfo latency separately. A negative or
inconclusive mapping is a useful result, but not a passed source-mapping gate.

Related acquisition constraints and retained clean/partial distinctions are in
[A3 simulator timelines](a3-simulator-timeline.md); the underlying control
semantics are in [synchronization controls](a3-synchronization-controls.md).
