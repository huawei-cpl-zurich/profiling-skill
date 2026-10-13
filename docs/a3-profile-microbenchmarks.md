# A3 causal profiling microbenchmarks

`benchmarks/a3_profile_microbench.py` provides correctness-checked before/after
probes for the hypotheses in `experiments/a3-profile-hypotheses.json`. The JSON
manifest is the machine-readable source for sweep values, predicted evidence,
and falsifiers. Contract discovery does not require Triton or an NPU:

```bash
python benchmarks/a3_profile_microbench.py --contract
```

Run both variants with identical seeded inputs and check their outputs:

```bash
python benchmarks/a3_profile_microbench.py \
  --experiment loop-bound --sweep short-local --pair --launches 3
```

Use `--variant before` or `--variant after` for an individual profiler launch.
The single-variant form omits tensor values from stdout so profiler logs remain
small. Compilation limitations are returned as `status: unsupported`; a pair
containing unsupported or failed work is `inconclusive`, never a fabricated
comparison.

## Causal probe inventory

The ten probes isolate local sequence bounds, direct head rank, key tile size,
empty-tile branching, scalar block decisions, final-predicate load masking,
warp/stage schedules, row reduction versus tiled dot, query-row reuse, and
generic versus specialized sparse-mode control. These are deliberately small
proxies, not performance models of BSA. A result supports a kernel hypothesis
only when the sweep changes the predicted latency and A3 activity evidence in
the declared direction. The manifest's falsifier is equally important: for
example, a compiler may eliminate a scan, scalarize a vector decision, or
reject a resource-heavy tile.

The six timeline controls cover an ordered stability replicate, reversed
source order, repeated load/compute regions, compute-duration scaling,
serialized versus independent dataflow, and fused versus split launches.
Compiler scheduling is part of the observation: source order does not prove
executed order. The fused/split control supplies known task boundaries; the
other probes test whether any exported timeline field tracks the intended
change.

## Profiling protocol

On A3, first run a selected variant without filtering to learn the exact
exported kernel name. Then repeat the same variant with `msprof op`, one launch,
kernel replay, and the exact selector. Capture `BasicInfo` and
`PipeUtilization` separately. Compare matching block/sub-block observations;
do not align rows as temporal intervals.

```bash
msprof op \
  --application='python benchmarks/a3_profile_microbench.py --experiment row-reduction-vs-dot --sweep square --variant after --launches 1' \
  --output=PROFILE_DIR --aic-metrics=PipeUtilization \
  --warm-up=1 --launch-count=1 --replay-mode=kernel \
  --kernel-name=a3_dot_probe
```

The runner reports expected selectors, but discovery remains mandatory because
compiler versions can rename exports. Device timing comes from `OpBasicInfo`,
not `host_observed_us`; host values are diagnostic only. Interpret A3 pipe
ratios as activity composition, not capacity saturation.

## Timeline decision rule

`PipeUtilization` is aggregate block/sub-block activity and supplies no common
clock. A `TimelineDetail` relationship is not temporal pipe utilization. A
timeline hypothesis is supported only if the installed A3 exporter provides a
timestamped field with documented or empirically controlled correspondence to
the reversed/repeated/scaled probes. If `msprof op` emits no TimelineDetail, or
the fields cannot distinguish the fused controls, record the result as
`unsupported` or `inconclusive`. Never infer a load/vector/store phase diagram
from aggregate ratios.

For reproducibility retain the manifest revision, source hash, exact command,
target and selected physical device, durable remote handle, kernel selector,
raw one-launch CSV hashes, and compact normalized evidence. Infrastructure
failures are retried against the same logical experiment and are not converted
into compiler or correctness failures.
