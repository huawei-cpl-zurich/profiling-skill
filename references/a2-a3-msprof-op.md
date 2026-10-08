# A2/A3 `msprof op` evidence

This guide applies only to Ascend A2/A3. Do not import A5/V6 event IDs,
ceilings, bandwidth multipliers, `PipeTimeline`, or per-pipe `InstrTimeline`
semantics. The source-backed definitions below come from
`Ascend/msprof@922ce56938fe48063bbffb5019b65645b2bb77cb` through
`ref://profiling-skill/common/profiling/repos/ascend-msprof-pipe-memory-timeline-semantics`.

Use `scripts/profile_a3.py` inside the managed `$gz-a3` native Python 3.11
runtime. The profile wrapper selects the physical device; the workload and
profiler see logical device 0. Do not pass a host physical ID to Triton code.

The helper runs one bounded `BasicInfo` capture with kernel replay and writes:

- `raw/`: retained vendor output;
- `msprof.log`: the complete profiler transcript;
- `evidence.json`: compact, deterministic latency evidence also printed as one
  JSON line.

Example inside the managed job:

```bash
python scripts/profile_a3.py \
  --output artifacts/profile/round-1-case-47 \
  --kernel-name '<exact exported kernel>' \
  --warm-up 3 --launch-count 1 -- \
  python benchmark_case.py --case 47
```

For performance measurements, `--kernel-name` is required: an unfiltered
capture may select the first framework setup operator rather than the Triton
kernel. An unfiltered capture is suitable only for discovering exported names.

The workload must execute one deterministic case, validate its result, and
return nonzero on compilation, runtime, or correctness failure. Keep input
construction and host-to-device transfer outside the profiled launch whenever
the benchmark interface permits it. Use separate captures for repetitions;
the experiment controller should take the median of successful captures.

## Evidence contract

Successful schema version 1 evidence has `status: success`, target family
`Ascend-A2-A3`, metric `BasicInfo`, and one `kernels` entry per matching
exported kernel. Each entry contains the raw duration samples and min, median,
p90, and max in microseconds. `sources` records each contributing CSV hash.

A failed command still writes JSON with `status: failure`, the msprof return
code, and a concise reason; the full compiler/runtime/profiler transcript stays
in `msprof.log`. Absence of the profiler terminal-success marker, absence of a
matching numeric row, timeout, and nonzero msprof exit are failures. Do not
turn them into timing samples.

`BasicInfo` task duration is device-side task latency, not end-to-end Python
latency. Multiple exported kernels are reported separately unless an exact
selector is supplied. Do not sum them unless the benchmark specification
defines that aggregate. Do not apply the A5 PMU event IDs, bandwidth
multipliers, core ceilings, or utilization formulas to A2/A3 captures.

## Pipe and data-path inventory

`PipeUtilization` ratios are `active_cycles / taskCyc`. They locate work and
show its composition over the task, but they are not a validated fraction of
the pipe's maximum throughput. Even a ratio near 100% does not by itself prove
capacity saturation.

| Exported field | A2/A3 pipe | Code meaning | Evidence kind |
| --- | --- | --- | --- |
| `aic_cube_ratio` | AIC Cube | matrix multiply/accumulate work such as lowered dot operations | activity only |
| `aiv_vec_ratio` | AIV Vector | elementwise/vector arithmetic and conversions | activity only |
| `aic_scalar_ratio`, `aiv_scalar_ratio` | AIC/AIV Scalar | scalar address, loop, predicate, and control work | activity only |
| `aic_mte1_ratio` | AIC MTE1 | L1 to L0A/L0B operand movement | activity only |
| `aic_mte2_ratio`, `aiv_mte2_ratio` | AIC/AIV MTE2 | DDR/GM into the AI Core memory hierarchy | activity only |
| `aic_mte3_ratio`, `aiv_mte3_ratio` | AIC/AIV MTE3 | AI Core memory hierarchy back to DDR/GM | activity only |
| `aic_fixpipe_ratio` | AIC FixPipe | deployed exporter activity; source attribution is unresolved | activity only |

The source maps the default event pairs as vector `0x8`, cube `0xa`, scalar
`0x9`, MTE1 `0xb`, MTE2 `0xc`, and MTE3 `0xd`. These identifiers are recorded
for A2/A3 attribution only; they are not transferable to A5/V6. The deployed
A3 exporter also emits FixPipe, but the retrieved default-event definition did
not establish its A2/A3 event mapping, so the model preserves it as an
unvalidated activity field.

`MemoryAccess` can expose GM→L1, L0C→L1, L0C→GM, GM→UB, and UB→GM paths. The
retrieved source did not establish their export scaling, so retain values and
mark the unit unknown. Do not convert them to bytes or bandwidth without an
additional product-valid definition.

`memory_bound = mte2_ratio / max(mac_ratio, vec_ratio)` is a bottleneck hint,
not a saturation metric. Values below 1 mean the documented heuristic found no
memory bottleneck; values above 1 mean memory activity dominates; exactly 1 is
unclassified. Cube utilization may approach the theoretical 100%, but no
lower universal saturation cutoff is documented.

## Normalize compact pipe evidence

Capture `PipeUtilization` and, when needed, `Memory` in separate comparable
single-launch replays (`--launch-count=1`) with the same exact kernel selector.
Each replay's `OpBasicInfo` must contain exactly one total row, and that row
must exactly match the selector. Repeated launches and multi-operator exports
are rejected rather than ambiguously associated. Keep the vendor report
remotely and pass only its compact exported CSVs to:

```bash
python3 scripts/normalize_a3_pipe_evidence.py \
  --product a3 \
  --activity-model references/a3-pipe-activity-model.json \
  --basic-info PipeCapture/OpBasicInfo.csv \
  --pipe-utilization PipeCapture/PipeUtilization.csv \
  --capture-id '<pipe replay durable handle>' \
  --memory-access MemoryCapture/Memory.csv \
  --memory-basic-info MemoryCapture/OpBasicInfo.csv \
  --memory-capture-id '<distinct memory replay durable handle>' \
  --kernel-name '<exact exported kernel>' \
  --launch-count 1 \
  --output activity-evidence.json
```

Each exported block/sub-block row becomes an `observation` with a reported
duration, not a temporal phase or interval. Rows have no common clock and must
never be aligned or used to infer overlap or temporal phases across blocks.
Duplicate `(block_id, sub_block_id)` keys are rejected.

Pipe and Memory reports are distinct captures and retain distinct durable
capture IDs. Automatic joining is permitted only when both single-launch
reports have exactly the same `(block_id, sub_block_id)` key set; missing,
extra, disjoint, or duplicate keys fail closed. Source hashes carry stable
roles (`pipe_basic_info`, `pipe_utilization`, `memory_basic_info`, and
`memory_access`) and the combined digest binds those roles. Memory columns
retain their source labels, but their normalized unit is `null` and scaling is
`unvalidated`, even when a vendor column name contains `(KB)`.

The activity model marks saturation `unknown` because no capacity denominator
is available. Do not pass this output to `analyze_pipe_saturation.py`; that
interface accepts temporal phase/capacity evidence, which A3 block exports are
not. Activity ratios can identify a compute-heavy or movement-heavy
block/sub-block observation, but cannot prove a saturated bottleneck or locate
a temporal phase.

`TimelineDetail` flow events describe mapping relationships, not capacity.
Sampled usage is based on task cycles divided by frequency times elapsed
interval, so use it to locate active and inactive intervals only when the
capture provides those inputs. It does not change the fail-closed saturation
result.
