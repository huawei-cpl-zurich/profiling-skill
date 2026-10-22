# A2/A3 `msprof op` evidence

This guide applies only to Ascend A2/A3. Do not import A5/V6 event IDs,
ceilings, bandwidth multipliers, `PipeTimeline`, or per-pipe `InstrTimeline`
semantics. The source-backed definitions below come from
`Ascend/msprof@922ce56938fe48063bbffb5019b65645b2bb77cb`.

Read [Two-pass acquisition](acquisition-workflow.md) for discovery and exact
selection. `scripts/profile_a3.py` runs directly in the configured workload
environment; profiler and application must agree on device addressing. The helper
preserves an inherited `ASCEND_RT_VISIBLE_DEVICES` value and defaults it to `0`
when unset. Set the workload's intended visibility explicitly before running it.

The helper runs one bounded `BasicInfo` capture and writes:

- `raw/`: retained vendor output;
- `msprof.log`: the complete profiler transcript;
- `evidence.json`: compact, deterministic latency evidence also printed as one
  JSON line.

Example from the skill folder:

```bash
python scripts/profile_a3.py \
  --output /absolute/evidence/timing \
  --kernel-name '<exact exported kernel>' \
  --warm-up 3 --launch-count 1 -- \
  python /absolute/workload.py
```

`--replay-mode` accepts exactly `kernel` or `application` and defaults to
`kernel`. Use `application` when kernel replay cannot create its required memory snapshot:

```bash
python scripts/profile_a3.py \
  --output /absolute/evidence/application-timing \
  --kernel-name '<exact exported kernel>' \
  --replay-mode application --warm-up 3 --launch-count 1 -- \
  python /absolute/workload.py
```

The compact evidence records the actual mode in `protocol.replay_mode`.
Do not combine samples captured with different replay modes.

For performance measurements, `--kernel-name` is required: an unfiltered
capture may select the first framework setup operator rather than the Triton
kernel. An unfiltered capture is suitable only for discovering exported names.

The workload must execute one deterministic case, validate its result, and
return nonzero on compilation, runtime, or correctness failure. Keep input
construction and host-to-device transfer outside the profiled launch whenever
the benchmark interface permits it. Use separate captures for repetitions;
the experiment controller should report the median and dispersion of all
declared valid captures, retaining failed captures separately. Do not select a
favorable stable subset or repeatedly sample until variability disappears.

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

The pinned source expresses `PipeUtilization` ratios as
`active_cycles / taskCyc`, but the exported `taskCyc` normalization must be
verified for the installed exporter. Do not equate it automatically with a
row's `aic_total_cycles`/`aiv_total_cycles`, row duration, or BasicInfo task
duration. Ratios locate activity; they are not a validated fraction of the
pipe's maximum throughput. Even a ratio near 100% does not by itself prove
capacity saturation.

| Exported field | A2/A3 pipe | Code meaning | Evidence kind |
| --- | --- | --- | --- |
| `aic_cube_ratio` | AIC Cube | matrix multiply/accumulate work such as lowered dot operations | activity only |
| `aiv_vec_ratio` | AIV Vector | vector arithmetic, conversions, masks, reductions and applicable local copies | activity only |
| `aic_scalar_ratio`, `aiv_scalar_ratio` | AIC/AIV Scalar | scalar address, loop, predicate/control work; scalarized indirect loads can also contribute | activity only |
| `aic_mte1_ratio` | AIC MTE1 | L1 to L0A/L0B operand movement | activity only |
| `aic_mte2_ratio`, `aiv_mte2_ratio` | AIC/AIV MTE2 | DDR/GM into the AI Core memory hierarchy | activity only |
| `aic_mte3_ratio`, `aiv_mte3_ratio` | AIC/AIV MTE3 | applicable outbound DMA paths such as UB to GM; not every store | activity only |
| `aic_fixpipe_ratio` | AIC FixPipe | applicable matrix-result movement/conversion; exact event mapping remains unresolved | activity only |

The source maps the default event pairs as vector `0x8`, cube `0xa`, scalar
`0x9`, MTE1 `0xb`, MTE2 `0xc`, and MTE3 `0xd`. These identifiers are recorded
for A2/A3 attribution only; they are not transferable to A5/V6. The deployed
A3 exporter also emits FixPipe, but the retrieved default-event definition did
not establish its A2/A3 event mapping, so the model preserves it as an
unvalidated activity field. Observed A3 compiler lowering includes CC/L0C to
GM through FIXPIPE; this is not a validation of every exporter field or a
portable mapping of all matrix outputs.

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
at its capture location and pass its exported CSVs to:

```bash
python3 scripts/normalize_a3_pipe_evidence.py \
  --product a3 \
  --activity-model references/a3-pipe-activity-model.json \
  --basic-info PipeCapture/OpBasicInfo.csv \
  --pipe-utilization PipeCapture/PipeUtilization.csv \
  --capture-id '<unique pipe capture ID>' \
  --memory-access MemoryCapture/Memory.csv \
  --memory-basic-info MemoryCapture/OpBasicInfo.csv \
  --memory-capture-id '<distinct memory capture ID>' \
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
Automatic instruction capture and explicit Triton simulation are distinct;
read [A3 timelines and correctness diagnostics](a3-timeline-and-correctness.md)
before interpreting instruction traces. Neither simulator timestamps nor pipe
instrumentation replace the benchmark's separate BasicInfo timing captures.
Sampled usage is based on task cycles divided by frequency times elapsed
interval, so use it to locate active and inactive intervals only when the
capture provides those inputs. It does not change the fail-closed saturation
result.

## Field glossary and calculations

The following names are the A2/A3 section of the
[official profiler field reference](https://github.com/Ascend/msopprof/blob/a52c8694979016be703006ec242901519bef31d8/docs/en/user_guide/msopprof_performance_data.md#pipeutilization-execution-time-and-ratios-of-compute-units-and-mtes).
The source revision is `a52c8694979016be703006ec242901519bef31d8`. Installed
versions can omit fields, especially event-ID waits and instrumented bandwidth.
`ai*` below expands to `aic` or `aiv`; `id*` spans `id0` through `id15`.
The wildcard is explanatory, never a literal CSV column or a physical-device ID.

| Field family | Unit and scope | Use |
| --- | --- | --- |
| `block_id`, `sub_block_id` | Exported task-block and core/sub-core identifiers | Join only exact keys within validated matching reports; do not interpret an identifier as a timestamp. |
| `aic_time(us)`, `aiv_time(us)` | Per-Cube/per-Vector unit's allocated-task window, μs | A row-local duration; not automatically the exported-ratio denominator, separately captured latency, or sum over cores. |
| `aic_total_cycles`, `aiv_total_cycles` | Cycles in that unit's task window | Time/cycle consistency check; not automatically the denominator used by the exported activity ratios. |
| `aic_cube_time(us)`, `aic_cube_ratio` | Cube instruction time and active-cycle fraction | Arithmetic activity, not achieved FLOP/s or percentage of theoretical FLOP capacity. |
| `aiv_vec_time(us)`, `aiv_vec_ratio` | Vector instruction time and active-cycle fraction | Arithmetic, conversion and applicable local-copy activity. |
| `ai*_scalar_time(us)`, `ai*_scalar_ratio` | Scalar instruction time and active-cycle fraction | Address/control/scalarized work; use the subcategories below to investigate. |
| `ai*_scalar_single_time(us)`, `ai*_scalar_dual_time(us)` | Single-issue and dual-issue time | Compare instruction issuance modes; neither alone is an instruction count. |
| `aic_mte1_time(us)`, `aic_mte1_ratio` | L1→L0A/B transfer time/fraction | The documented time excludes transfer waiting; inspect MTE1 waits separately. |
| `ai*_mte2_time(us)`, `ai*_mte2_ratio` | Input-transfer time/fraction | Confirm path and transfer shape; fractions are not bandwidth. |
| `ai*_mte3_time(us)`, `ai*_mte3_ratio` | Outbound-transfer time/fraction | Includes applicable output instructions, not every source-level store. |
| `aic_fixpipe_time(us)`, `aic_fixpipe_ratio` | L0C→GM/L1 movement/conversion time/fraction | Diagnose matrix-result output independently from MTE3. |
| `ai*_scalar_wait_time(us)` | Intra-core scalar wait blocking, μs | Event wait category, not all synchronization or all idle time. |
| `ai*_scalar_wait_id0_time(us)` … `ai*_scalar_wait_id15_time(us)` | Inter-core wait-ID blocking, μs | Only emitted when available; correlate IDs with bound generated sync, not guessed source phases. |
| `aic_scalar_mte1_stall_time(us)`, `ai*_scalar_mte2_stall_time(us)`, `ai*_scalar_mte3_stall_time(us)` | Scalar blocked by the named full instruction queue, μs | Backpressure while issuing movement work; distinct from an explicit wait instruction. |
| `aic_scalar_cube_stall_time(us)`, `aiv_scalar_vector_stall_time(us)` | Scalar blocked by Cube/Vector instruction queues, μs | Producer issue rate versus consumer progress; high values can accompany useful busy compute. |
| `ai*_scalar_wait_ib_time(us)` | Scalar instruction-buffer/iCache wait, μs | Investigate code size, fetch path and control flow, rather than automatically adding data buffers. |
| `aiv_scalar_stall_by_ub_time(us)` | Scalar blockage associated with UB, μs | A symptom requiring UB conflict/dependency evidence; not an automatic proof of a bank conflict. |
| `ai*_icache_miss_rate` | Exported iCache rate; retain installation's scale | Compare with IB waiting; do not assume an unlabeled rate is a percentage. |
| `aic_mte3_active_bw(GB/s)`, `aiv_mte3_active_bw(GB/s)` | Bytes per active outbound-transfer time, decimal GB/s | Different denominator from whole-task useful-byte bandwidth. |
| `aic_fixpipe_active_bw(GB/s)`, `aiv_mte2_active_bw(GB/s)` | Active FIXPIPE/input-transfer bandwidth, decimal GB/s | Preserve path/core scope and the profiler's traffic definition. |
| `aic_mte1_active_bw(GB/s)`, `aic_mte2_active_bw(GB/s)` | Instrumented active bandwidth for matrix-input channels | A2/A3 availability requires `MemoryDetail` dynamic instrumentation; do not assume ordinary PipeUtilization contains them. |

The normalizer preserves available row-local fields in optional
`observations[].reported_metrics`, keyed by the **original CSV column name**.
Each entry has numeric `value`, exact trimmed `raw_value`, and `unit` (`us`,
`cycles`, `fraction`, `GB/s`, or `null` for an unlabeled unknown scale).
A present empty/NA field has `value: null`; an absent column has no entry.
Counterpart AIC/AIV columns are not copied into the wrong engine's observation.
Existing `activity`, `metrics`, and unknown-saturation outputs are unchanged.
There is no conversion of active bandwidth into capacity classification.

### Verify the denominator before doing arithmetic

A real A3 mixed-kernel capture exposes the danger of substituting row duration.
Representative Cube fields were:

```text
aic_time(us) = 12.689445       aic_total_cycles = 22841
aic_cube_ratio = 0.04         aic_cube_time(us) = 0.68
aic_mte2_ratio = 0.663235     aic_mte2_time(us) = 11.275
```

The row's time/cycle pair implies approximately `1800 cycles/μs`, but
`0.68 / 0.04 = 17 μs` and `11.275 / 0.663235 ≈ 17 μs`, not `12.689445 μs`.
A Vector row similarly had `0.117778 / 0.006928 ≈ 17 μs` for Vector and
`0.980556 / 0.05768 ≈ 17 μs` for MTE2, despite a row time of `12.597222 μs`.
The same report's BasicInfo task duration was `17.399652 μs`, **also different**.
Neither row time nor BasicInfo can simply replace the normalization window.
Preserve all three quantities separately; the implied window is a consistency
observation, not a validated exporter formula or a reconstructed timeline.

The [compact denominator example](a3-ratio-denominator-example.json) preserves
the selected raw fields. Source SHA-256 bindings: PipeUtilization
`e1cee99c65323908f14e1d8e1ecda60d8da847044a69fa6d24d0ebf3eb024b1d`;
same-report OpBasicInfo
`3b0f4bfabaa9847b28b12433d9f17650df682cff74109baea6338eb3f907a0da`.
The selector was `streaming_matmul_add_kernel_mix_aic`; the report described
24 Cube blocks and 48 Vector sub-blocks at a reported frequency of1800 MHz.
These are diagnostic report values, not an independent latency comparison.

For a ratio `r`, multiplying by cycles is valid only after establishing that
those exact cycles are its denominator. Similarly, `r × row_time` is not a
substitute for the exported active time without a validated matching scope.
The normalizer preserves reported fractions and active times separately and
does not reconstruct active cycles or classify capacity from them.

### Worked arithmetic with explicit scopes

This calculation is illustrative, not a measured result. Suppose a row reports
`10 μs` duration and `2 μs` intra-core scalar wait. Their derived ratio is
`2/10 = 0.2` **of that chosen row-duration window**. It is neither an exported
pipe ratio nor 20% of every pipe's idle time, and it does not prove a removable
2 μs critical-path cost.

If a validated transfer moved `600000` bytes during its **reported** `6 μs`
active time, active bandwidth would be `600000/(6 × 1000) = 100 GB/s`.
Using a separately defined whole-task window of `10 μs` would give `60 GB/s`.
State the bytes, window and unit rather than treating both as one bandwidth.
Cycles convert to microseconds via `cycles/frequency_Hz × 10^6` only with the
corresponding frequency and event scope. Rounding, frequency changes and
exporter normalization differences must remain visible, not silently corrected.

A row can have Cube 70%, MTE2 60% and scalar 40% because resources overlap.
The sum is not 170% device utilization, and `100% − sum` is not idle time.
Scalar subcategory times are not promised to partition scalar or task time;
do not add them into a critical-path budget.

## Diagnose waits and stalls

First retain all available categories with their core/row and capture. Compare
absolute microseconds alongside explicitly labeled derived fractions of a
chosen window; exported pipe ratios can use a different normalization. A smaller task can raise a wait fraction while reducing absolute wait time.
An absent counter means unavailable, not zero. A reported zero describes only
that event and coverage. A nonzero stall can overlap useful work elsewhere.

| Observation | Competing explanation to test | Next measurement / candidate |
| --- | --- | --- |
| Intra-core `scalar_wait_time` is large | True producer→consumer dependency, or reuse of one buffer delaying the next iteration | Pair generated set/wait with the actual addresses and lifetime; inspect whether an independent buffer/iteration can overlap safely. |
| One or more `scalar_wait_id*` fields are large | Cube/Vector handoff imbalance, flag reuse, or a slow producer | Identify the matching cross-core flag and both core types; compare producer movement/compute and consumer epilogue. Do not map `id7` to “core7” or a source phase without compiler evidence. |
| MTE1/2/3 queue stalls are large | Movement pipe cannot accept issue as fast as scalar produces it | Compare named pipe active time, transfer sizes, strides, and active bandwidth where valid. Test coalescing/reuse/batching before blindly increasing unrolling. |
| Cube/Vector queue stalls are large | Useful compute consumes the queue, or dependency/resource pressure prevents progress | Check compute workload and conflict metrics; reduce redundant operations or improve independent scheduling only after identifying which. |
| IB wait / iCache miss is large | Large unrolled body, branch/loop path, instruction fetch pressure | Compare late code size and a smaller-unroll variant; extra buffers do not directly repair instruction fetch. |
| UB stall is large | UB access conflict, dependency, or transfer interaction | Capture `ResourceConflictRatio`, inspect layout/stride and actual concurrent operations; change one layout/padding choice. |
| Explicit waits are zero but latency falls after scheduling change | Queue stalls, macro-internal synchronization, critical startup/drain or another counter changed | Inspect full stall categories and late IR/macro operands; zero explicit wait is not evidence that buffering was irrelevant. |

Capture `ResourceConflictRatio` separately with the exact selected kernel,
launch count 1 and matching inputs/replay mode. Check support with installed
`msprof op --help`:

```bash
msprof op --output=/absolute/evidence/conflicts \
  --aic-metrics=ResourceConflictRatio --kernel-name='<exact exported kernel>' \
  --warm-up=3 --launch-count=1 --application='python3 /absolute/workload.py'
```

The [official conflict field definitions](https://github.com/Ascend/msopprof/blob/a52c8694979016be703006ec242901519bef31d8/docs/en/user_guide/msopprof_performance_data.md#resourceconflictratio-resource-conflict-ratio)
separate `aiv_vec_bankgroup_cflt_ratio` (stride/bank-group interaction),
`aiv_vec_bank_cflt_ratio` (operand-address/bank interaction),
`aiv_vec_resc_cflt_ratio` (execution-resource contention), and
`aiv_vec_mte_cflt_ratio` (Vector/MTE interaction), with
`aiv_vec_total_cflt_ratio` as broader conflict evidence. The remaining
`aic_cube_wait_ratio`, `aiv_vec_wait_ratio`, `aic_mte1_wait_ratio`,
`ai*_mte2_wait_ratio`, and `ai*_mte3_wait_ratio` describe blocked-instruction
cycle ratios, not the scalar explicit-wait categories above. Preserve the
source definition/denominator: do not equate all ratios to task-cycle fractions
or add overlapping conflict ratios. A second replay cannot prove temporal
alignment with the PipeUtilization replay.

Changing padding/layout, transfer granularity, tile shape or buffering is a
hypothesis. Recheck correctness, inspect whether the intended lowering changed,
and compare independent BasicInfo samples; conflict reductions without timing
improvement are diagnostic observations.
