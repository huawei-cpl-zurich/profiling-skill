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

The control differences are runtime-live. The fixed loop reads its active
extent at runtime while retaining the inactive predicates; head rank is either
scanned from runtime flags or loaded directly and then feeds the source
address; key tile size changes the real reduction loop. Scalar decisions feed
non-identity arithmetic, generic sparse mode reads a runtime extent, and the
load-mask variants issue mutually exclusive unmasked or final-predicate loads.
The manifest declares a minimum pipe-signal sanity change so a compiler-erased
contrast is reported as inconclusive profiling evidence, not as causal proof.

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

## Resumable campaign

The campaign coordinator expands the manifest into 52 variant/sweep cells. It
uses `scripts/acquire_profile.py` for a discovery pass followed by an exact
PipeUtilization pass for every expected kernel. Three samples whose relative
range is at most 5% are required by default; acquisition is capped at seven
attempts. The retained dispatch receipt is authoritative after observer loss,
so rerunning a cell resumes its handle instead of creating another job.
Terminal infrastructure or malformed-evidence attempts retire that receipt
and advance to a numbered retry identity. A nonterminal transport interruption
does not consume the attempt identity. Each remote acquisition holds a
nonblocking physical-device lease from probe through evidence emission.

Run two deterministic shards concurrently on the two configured A3 targets:

```bash
python scripts/a3_profile_microbench_campaign.py run \
  --target bz-a3-1 --shard-index 0 --shard-count 2 --output-dir artifacts/a3-shard-0
python scripts/a3_profile_microbench_campaign.py run \
  --target bz-a3-2 --shard-index 1 --shard-count 2 --output-dir artifacts/a3-shard-1
python scripts/a3_profile_microbench_campaign.py merge \
  --input artifacts/a3-shard-0/shard-result.json \
          artifacts/a3-shard-1/shard-result.json \
  --output artifacts/a3-hypothesis-results.json
```

The merged `schema_version: 1` document retains all per-variant evidence under
`raw_cells` and exposes report-ready paired hypothesis records under `results`.
Each result contains structured expected direction/threshold, one before/after
measurement per sweep, and signal relationships using the report vocabulary
`tracks_outcome`, `context_only`, `contradicts_outcome`, or `not_exported`.
Raw cells retain samples, pipe signals, discarded infrastructure attempts, and
handles/devices/source hashes. Compilation, runtime, and correctness failures
are `candidate_failure`; profiler or malformed-evidence failures are
`profiling_failure`; exhausted device/transport problems remain
`infrastructure_exhausted`. For a split variant, the reported duration proxy is
the sum of each exact kernel's maximum AIC/AIV reported time. It is explicitly
a PipeUtilization core-time proxy, not BasicInfo task latency.

Before/after cells are assigned to the same shard and are paired only when
their retained target and physical-device placement match. Shard merge also
verifies the canonical manifest digest, while cached terminal cells are bound
to the manifest, benchmark source, target, sampling policy, and cell contract.

Timeline cells make a real `--aic-metrics=TimelineDetail` acquisition after
BasicInfo discovery. Compact identities for any timeline files are retained;
otherwise the attempted command's log digest and explicit absence are recorded
as `unavailable`. On the validated A3 runtime, handle
`remote:bz-a3-1:job:20261013T225304Z-fdae1c669479` accepted the attempt but
exported no inspectable TimelineDetail files. This is not a candidate failure
and does not authorize phase-order claims from aggregate pipe activity.
