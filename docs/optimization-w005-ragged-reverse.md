# Single-pass ragged reverse

This study applies the existing redundant-store-elimination technique to a new
operator. It adds no new unique-technique claim. The standalone implementation
supports contiguous finite FP32 `x[B,T,D]`, contiguous int32 `lengths[B]`, and a
separate FP32 output. Every operation clamps each length into `[0,T]`. Inputs
remain unchanged, and the valid prefix reverses without changing its channels;
the suffix is copied unchanged.

The source contract, clamping policy and involution property come from Ascend
ops-rec commit `e489c0bf6a2d5f2d05b3205fcacc01b62a908b10` (Apache-2.0):

- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/sequence/reverse_sequence/docs/torch_reverse_sequence.md:L11-L55`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/sequence/reverse_sequence/torch_extension/reverse_sequence_plugin.cpp:L44-L67`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#ascendc/sequence/reverse_sequence/examples/torch/test_torch_reverse_sequence.py:L28-L47`
- `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-ascend-ops-rec#cmake/soc_ops_config.cmake:L20-L36`

The retrieved plugin clones the input because its native prefix kernel leaves
the suffix untouched. The independent Triton AFTER body implements the documented
mapping directly. The upstream native device body was not retrieved; this is no
claim about its performance. Attribution and the Apache license accompany the
runnable example.

BEFORE always launches a full clone, then overwrites the reversed prefix from
the original immutable input. AFTER writes each output element once:

```python
length = minimum(maximum(lengths[batch], 0), T)
source_time = where(time < length, length - 1 - time, time)
out[batch, time, channel] = x[batch, source_time, channel]
```

The block size, flat output mapping and dtype are the same. Both arms compute
length clamping on device each operation; no data-dependent work moves into
setup. Allocation and input movement happen before the measured API in both
arms. The clone and all required output stores execute inside it. There is no
zero-length fast dispatch: BEFORE still launches both stages.

Look for a clone/copy whose prefix is immediately overwritten by a permutation.
Replacing it can remove the extra launch and prefix rereads/stores. With all
lengths zero, the net logical data movement does not decrease: both sides still
copy the complete input. For general lengths, the net logical read-plus-write
reduction is `8 * D * sum(clamp(lengths,0,T))` bytes for this FP32 format. The
`logical_full_copy_bytes` metadata counts the eliminated clone stage's output
bytes, not net savings or measured physical traffic. Predicates, integer indexing
and less regular gathers can compete with a simple contiguous copy. No dispatch
threshold or universal speed benefit follows from the fixed examples.

The CPU oracle uses independent integer gather indices. Correctness compares
FP32 bit views exactly, checks immutable inputs and nonaliasing output, and
observes any remaining NaN sentinel. Guard GA performs two reversals per guard
API, checks its nontrivial intermediate against one reversal and its final result
against the original. Guard GB checks partial prefixes and untouched suffixes.
These are correctness-only guards with zero captures; measured cases perform
one reversal per API.

The protocol freezes four main cases and two guards before execution. Each of
six CPU payloads is cached once, each side is checked once per case, then each
main arm executes exactly ten profiled API forwards. There are no warmups,
autotuning, separate timing calls, or profiled tensor inspections.

The primary boundary is the complete device-operation span from first to last
attributed compute task, including interkernel gaps. Event samples, compute union
and noncompute gap remain separate. Profiler-active spans are diagnostic and do
not establish ordinary application latency. Noncompute gaps can contain caller
dispatch and do not prove all-device idle. PMU percentages do not establish
physical traffic, bandwidth or roofline distance.

All four main pairs and both guards passed on gz-a3 physical6 / logical0,
**Ascend910B4-1**, CANN9.1.0, Torch2.10.0+cpu, Torch-NPU2.10.0 and
imported Triton3.2.0. Eight CPU functional tests and all twelve logical arm compile
probes passed before scientific execution. Source/binary identities stayed stable.

| Case | B × T × D | Length pattern | BEFORE span µs ± SD | AFTER span µs ± SD | Before / after |
|---|---:|---|---:|---:|---:|
| W005-B04-01 | 64 × 256 × 32 | mostly-full | 916.826 ± 69.081 | 768.250 ± 1.044 | 1.193× |
| W005-B04-02 | 1 × 32 × 8 | all-zero | 193.186 ± 6.562 | 16.886 ± 0.452 | 11.441× |
| W005-B04-03 | 48 × 192 × 48 | mixed-long | 812.442 ± 65.160 | 714.180 ± 0.132 | 1.138× |
| W005-B04-04 | 2 × 64 × 16 | short-zero | 188.028 ± 18.916 | 16.688 ± 0.605 | 11.267× |

Geometric mean of these four instrumented span ratios: **3.637×**.
All four prospective benefit/cost cases favored the candidate; a timing regression
was not observed in these frozen cases. The zero-length case has no net logical
traffic saving. Changes in interkernel/caller gaps can dominate these diagnostic
spans, so the ratios cannot be interpreted as ordinary application speedups or
as measured bandwidth improvements.

| Case | BEFORE compute union µs | AFTER compute union µs | BEFORE noncompute gap µs | AFTER noncompute gap µs |
|---|---:|---:|---:|---:|
| W005-B04-01 | 884.994 | 768.250 | 31.832 | 0.000 |
| W005-B04-02 | 20.514 | 16.886 | 172.672 | 0.000 |
| W005-B04-03 | 783.568 | 714.180 | 28.874 | 0.000 |
| W005-B04-04 | 20.586 | 16.688 | 167.442 | 0.000 |

The legacy JSON field `device_idle_gap_us` means the noncompute gap in this table;
it does not prove all-device idle.

BEFORE has two device tasks and AFTER one per measured API:120 tasks total across
80 forwards. GA has two component reversals per correctness-only API (four
BEFORE tasks or two AFTER tasks), with both intermediate and final outputs checked.
Together the twelve correctness API calls contain fourteen component reversals.
The guards made zero captures. Every entered-API capture counter is exactly ten;
there were no failed, discarded or repeated measurement attempts.

All ten span/event samples, deviations, task ownership, original PMU fields,
source/binary identities, guard checks and six oracle manifests are retained in
`skills/ascend-optimization-library/references/ragged-reverse-evidence.json.gz`;
`ragged-reverse-summary.json` provides its digest and readable case index. Full
vendor trees remain remote. The verified compact archive retains all six cached
CPU payloads, source snapshots and compiled binary cache.

Gate `remote:gz-a3:job:5f07bb68-3727-437a-9cdf-4a382a822451` exited0.
Compact-only job `remote:gz-a3:job:490ff2c5-0da8-4862-9533-8016b22b91f4`
exited0 with zero new device work. Compact SHA256
`c2b091a36f2bd2947a3a0daf8dee78cf843f25c152e9f957b9b877748d95473b`.


To generate one cached CPU payload using the public nested case format:

```bash
python examples/optimization-library/triton/ragged_reverse_contract.py \
  --case case.json --output fixture.pt
```

The paired runner is `scripts/optimization_pair.py` with adapter
`ragged_reverse:Adapter`, the same case JSON and a shared oracle cache. Use the
configured remote-access and ascend-profiling workflow for actual NPU work.
