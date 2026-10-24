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

Collection is in progress; no performance result is claimed by this text yet.

To generate one cached CPU payload using the public nested case format:

```bash
python examples/optimization-library/triton/ragged_reverse_contract.py \
  --case case.json --output fixture.pt
```

The paired runner is `scripts/optimization_pair.py` with adapter
`ragged_reverse:Adapter`, the same case JSON and a shared oracle cache. Use the
configured remote-access and ascend-profiling workflow for actual NPU work.
