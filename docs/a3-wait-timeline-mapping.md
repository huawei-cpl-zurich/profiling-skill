# A3 wait-placement qualification

The six paired groups in `scripts/a3_wait_timeline.py --manifest` reuse the
existing native synchronization control: MTE2→Vector and Vector→MTE3 with
32 useful vector rounds, plus balanced/imbalanced inter-core modes 0 and 2.
Each pair compares `overlap` with `early-wait`; useful work, flags and outputs
are unchanged within that pair. Do not compare different balance groups as
equivalent-work candidates.

## Native application canary

Stage the repository through `$remote-access`, select its named runtime, and
run from the registry-derived workspace. Supply the **observed supported**
simulator SoC; there is intentionally no guessed default:

```bash
python scripts/a3_wait_timeline.py --family mte2-vector --canary \
  --soc-version "$OBSERVED_SIMULATOR_SOC" --output "$NEW_OUTPUT_DIRECTORY"
```

The worker compiles the native executable before acquiring one idle physical
device lease. Every application uses logical zero. It performs host numerical
verification, then passes an executable shell wrapper to the existing explicit
simulator worker. This exercises the native application route rather than the
historical Python/Torch simulator teardown path. No hidden Triton arguments or
binary-config ABI are copied or guessed. A failed canary is an acquisition gap,
not a successful timeline just because partial trace files exist.

Omit `--canary` for both variants and five interleaved BasicInfo, three separate
PipeUtilization and three separate Memory replays per variant. Each ordinary
subprocess and each acquisition stage has a 300-second limit. The outer
simulator worker has a 620-second bound because it performs two sequential
bounded stages: native correctness and profiling. Timeout kills its process
group before releasing the lease. Rotate pair order between rounds, retain all
samples, and preserve low-confidence flags rather than selecting a convenient
repeat. An explicit unsupported Memory metric stops only that metric group;
compiler, correctness and runtime errors remain failures.

Select several `--family` values, or omit them to run all six pairs. A failed
pair is retained while independent pairs continue. Output directories must be
new: there is no implicit retry, overwrite or duplicate scheduling. Transport
recovery and terminal infrastructure retries belong to the supervising
`$remote-access` invocation, not this measurement worker.

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
