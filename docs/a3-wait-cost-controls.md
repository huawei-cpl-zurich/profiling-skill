# Hardware timing controls for wait-placement interpretation

Run through the approved A3 remote-access runtime, not through the simulator:

```bash
python scripts/a3_wait_cost_batch.py --canary --output <fresh-evidence-directory>
python scripts/a3_wait_cost_batch.py --group 8192/32/16 --output <another-fresh-directory>
```

First run the four-mode canary. Expand only after all four numerical checks,
normal event timing and profiler groups are usable. The full sweep contains
eight groups: 1024/8192 float elements, 1/32 vector additions per repetition,
and 1/16 repetitions. Each group contains isolated transfer, isolated compute,
serialized combined work and delayed-wait combined work. Run whole groups,
not independently selected modes. The existing idle-device lease is retained
through correctness, event timing and profiler collection.

The paired kernels have identical input, outputs, memory, operations and one-block
launch geometry. Only the MTE2-to-vector wait position changes. Compute operates
on an independent UB buffer. Both the copied input and accumulated compute
result are returned and checked against a CPU formula, so compute is not dead
work. Each native application normally launches exactly one kernel; its separate
timing mode performs three warmups followed by seven stream-event measurements
without msprof, with positive microsecond samples retained. Event timing uses ACL
events, not whole-process wall time or device-intrusive timestamp instructions.
Every profiler application disables this timing loop. Five BasicInfo, three
PipeUtilization and three Memory captures rotate case order between rounds.
BasicInfo durations are diagnostic profiler durations, not the uninstrumented
canonical latency. No noisy samples are discarded or retried to manufacture
low variance. Missing metrics are unavailable, never zero.

Interpret serialized versus delayed-wait event latency as an intervention on
ordering. Isolated transfer and compute medians provide qualitative sum/max
references, not exact bounds: both independently include initialization, output
stores, launch/prologue/epilogue costs. All variants use a bounded working set;
repeated transfers read the same GM range, so cache warming is intentional and
this is not a sustained HBM-bandwidth benchmark. Each source iteration requests
a transfer, but repetitions overwrite its output. Confirm instruction retention
in generated code before claiming repeated transfers; source-level logical bytes
are not measured physical traffic. Vector work accumulates and is fully
observable. Tiny groups may be overhead dominated. A null or negative
performance change is useful evidence, not a failed experiment.

Pipe/Memory counters can support a bottleneck interpretation, but separate
captures do not prove simultaneous overlap, identify a stalled instruction or
reconstruct cross-core clocks. This control has one core; balanced/imbalanced
cross-core flag placement remains in the separate synchronization family.

`configuration.json` binds cases and all runner/build sources; `build/` retains
compiler stdout/stderr, binary identity and available artifacts. `summary.json`
records normal event samples separately from profile timing and exact raw
counters. Commands, process exits and capture checkpoints remain on failure.
`--resume` reuses only verified completed boundaries, retaining each original
device identity. Incomplete phases stop with retained logs and require an explicit
new attempt directory after diagnosis. Do not delete evidence or dispatch a
duplicate remote job while its handle is nonterminal.
