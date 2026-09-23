# Ascend 950/A5 metric semantics

These rules apply to the validated Ascend 950/A5 target (28 AIC and 56 AIV).
Do not transfer event IDs, byte multipliers, or ceilings to another product.

## Keep measurement layers separate

| Quantity | Meaning |
| --- | --- |
| Per-row pipe ratio | Fraction of one exported core/sub-core row occupied by a pipeline |
| Whole-device utilization | Useful cycles divided by elapsed time and the target's available cores |
| Useful throughput | Algorithmic operations or bytes divided by elapsed time |
| Average-path bandwidth | Bytes divided by the complete row/task duration |
| Active bandwidth | Bytes divided by cycles in which that path was active |
| Occupancy | Relative work distribution reported by the profiler, not GPU warp occupancy |
| PC samples | Sample counts at PCs, not elapsed cycles or instruction issue counts |

Rows are identified by `(block_id, sub_block_id)`. With cycles `C` and MHz
frequency `f`, row time is `C / f` microseconds. Most exported ratios are
fractions in `[0, 1]`, despite their names; instruction-cache miss is also a
fraction. `NA` means unavailable, not zero. Reported KB/MB/GB bandwidth uses
binary scaling.

## Aggregate from events

Prefer raw numerator and denominator events. For rows `i`, aggregate a ratio
as `sum(numerator_i) / sum(denominator_i)`, not as an unweighted mean of row
ratios. Keep AIC and AIV domains separate unless a formula explicitly defines
a whole-device denominator.

The reviewed profiler formulas are:

- Operator Profiler A5 `aic_cube_ratio = event[810] / row_cycles`.
- General-profiler V6 `mac_ratio = sum(event[0x301]) / sum(task_cycles)`.
- General V6 `cube_fops = event[0x323] * 8192 + event[0x324] * 16384`, subject
  to the event's dtype semantics.

The first two events are not aliases. Arithmetic instruction counts are event
counts, not FLOPs; A5 Roofline useful operations instead come from operand
recording.

Whole-device cube utilization is
`100 * accumulated_Cube_block_cycles / (frequency * physical_Cube_cores * Task_Duration)`.
It does not use operation counts or multiply by Cube/MAC ratio. On 28 AICs, a
single balanced 24-block wave has a coverage ceiling of `24/28 = 85.714%`
even if each participating row reports about 95% Cube composition.

For a memory path, average bandwidth divides event-derived bytes by complete
duration; active bandwidth divides by path-active cycles. State which one is
reported. A5 request-byte multipliers include:

| Path/event family | Bytes per counted request |
| --- | ---: |
| Main memory, GM to L1, GM to UB | 128 |
| L1 to L0A/B, Vector to/from UB | 256 |
| L0A to Cube | 64 |
| Cube to/from L0C | 1024 |

Vector GM-to-UB is an estimate that subtracts two data-cache event classes.
Current-master UB-to-GM fields use dynamic instrumentation and are `NA` when
that instrumentation is absent.

For `Ascend950PR_9579`, the embedded per-core reference ceilings used by the
vendor's binary-scaled GB/s usage calculations are GM-to-L1 `162.77`,
L0C-to-L1 `377.80`, L0C-to-GM `130.21`, GM-to-UB `163.13`, and UB-to-GM
`131.76`. Usage is capped at 100%, so retain the uncapped bandwidth value.

L2 event rates must retain their documented request/lookup denominator.
Resource-conflict categories may overlap and must not be summed into a total.
Occupancy is useful for comparing distribution between equivalent launches,
not for claiming a GPU-style residency limit.

L2 read/write hit denominators include close hits, far hits, misses, and victim
events. Inspect denominator and victim totals: a low rate with negligible
traffic is rarely the first bottleneck, while a high rate does not exclude
serialization.

ResourceConflictRatio includes Cube/MTE wait ratios and Vector STU/LDU/SFU,
Vector-wait, and MTE-wait ratios against row cycles. These overlap and are not
a complete stall partition. PCSampling categories include `IBuf_Empty`,
`Nop_Cycles`, `Scoreboard_Not_Ready`, `Register_bank_conflict`,
`Resource_conflict`, `Warp_Level_Sync`, `Divergence_Stack_Spill`, `Others`,
and `Active`; all are samples rather than exact cycles.

## Comparable campaigns

A5 exposes ten programmable PMU slots per pass, so a complete diagnosis may
require multiple replays. Keep each metric group in its own pass and record:

- application inputs and correctness result;
- exact runtime kernel selector;
- binary or source revision/hash;
- device and observed frequency;
- warmup and launch counts;
- replay/pass identity and capture order.

Use a separate minimally instrumented run for canonical latency. Reject
unweighted ratio aggregation, missing-as-zero conversion, mixed-frequency or
busy-device comparisons, and cross-product formulas.

Installed public 26.0 output lacks current-master A5 UB-to-GM average/data/
usage, UB-read-toward-GM, and Vector MTE3 active-bandwidth fields. Do not
project those fields onto an older capture. Exact event conditions, counter
width/overflow/reset behavior, and simultaneity require a version-matched PMU
manual or controlled capture; preserve that uncertainty in conclusions.
