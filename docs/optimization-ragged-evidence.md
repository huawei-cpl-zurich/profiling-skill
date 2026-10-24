# Ragged packing: measured prefix reuse and padding stores

These examples isolate two mechanisms in packed-to-padded sequence conversion.
They are evidence for evaluating a proposed change, including a negative result;
they do not establish that every mechanism improves complete-operation runtime.
The active optimization skill and demonstration inputs are unchanged.

The implementation is [ragged_pairs.py](../examples/optimization-library/triton/ragged_pairs.py),
with its independent CPU contract in
[ragged_contract.py](../examples/optimization-library/triton/ragged_contract.py).
The mechanism controls change independently: E02 changes only offset reuse;
E03 changes only the output-store pattern. Both use the same 32-by-64 tiles
and grid within a pair.

## Source and numerical contract

The source-backed starting point is the Triton-Ascend packed/unpacked sequence
implementation. Its preceding-length sum and padding-store/valid-overwrite
patterns are cited separately:

- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/pack_unpack.py:L24-L30`
- `ref://profiling-skill/triton-ascend/perf-analysis/repos/optimization-wave1-triton-kernels-scope-7acb9c40a6ca#src/triton_ascend_kernels/index/pack_unpack.py:L44-L51`

The examples adapt those patterns; the source citation is not device correctness
evidence. Inputs are packed `x[sum(lengths), D]` and integer `lengths[B]`.
Output is padded `[B, Lmax, D]`, preserving input values exactly and filling
remaining positions with the declared padding value. The CPU oracle uses
sequential slices, independently of the device scan/address calculation.
Empty sequences, all-empty packed input, feature tails and sequence tails are
included. FP16/FP32 equality uses zero tolerance, and correctness checks input
immutability. Nonfinite padding and unsupported shapes are rejected before
kernel preparation.

## E02: reusing sequence-prefix offsets

Before, each packing program sums preceding sequence lengths. After, an
exclusive device prefix scan writes `B` int32 offsets and the packing kernel
loads its sequence's offset:

```python
# Before: repeated within each packing program.
start = tl.full((), 0, tl.int32)
for previous in range(batch):
    start += tl.load(Lengths + previous)

# After: separate scan kernel, then the packing consumer.
exclusive = tl.cumsum(count, 0) - count
# Consumer:
start = tl.load(Starts + batch)
```

Both arms use the single output-store pattern. The after forward includes the
entire scan and consumer on every call. Its `4 * B` bytes of offset workspace
and output are preallocated; allocation is outside timing, but the scan is not.
Preparation launches no kernels.

Code with repeated prefix work is a reason to investigate reuse. The profiler
must then show that the saved consumer work outweighs scan execution and its
additional launch gap. In these measurements it does **not**: every complete
operation becomes slower, including both predeclared benefit hypotheses.
For the large gallery case the after operation's gap is approximately
144–181 microseconds, despite lower summed active kernel duration. Treat this
as measured negative guidance; it is not a demonstrated runtime optimization.
No replacement benefit case was selected after seeing these results.

## E03: removing a redundant padding store

Before, each packing program stores padding over its output tile and overwrites
valid elements. After, it selects the final value before one output store:

```python
# Before:
tl.store(destination, tl.full((BT, BD), PAD, tl.float32), bounds)
tl.store(destination, value, valid)

# After:
tl.store(destination, tl.where(time[:, None] < count, value, PAD), bounds)
```

Both arms calculate the prefix inside the same packing kernel. No scan or
additional launch is introduced. Repeated output stores plus high AIV MTE3
activity are a reason to investigate this change. The tradeoff is the selection
work and its temporary values; small tiles may have no measurable benefit.

For the substantial gallery case, exported AIV MTE3 time falls from
16.935 to 10.258 microseconds, while AIV vector time rises from 0.510 to
2.999 microseconds. This supports the store-traffic explanation without
pretending that selection is free. Retained adapter IR contains two output
destination materializations before and one after, and actual binary hashes
differ. Backend dead-store elimination was not independently disassembled.
If another compiler already removes the redundant store, these source patterns
alone cannot establish a gain.

The two substantial cases improve by 1.498× and 1.346× in operation span.
The small gallery case is inconclusive; the empty held-out case has only a
small point improvement. Neither predeclared cost case demonstrates a
regression. This evidence therefore does not provide a measured E03 slowdown
example or justify claiming a universal improvement.

## Frozen cases and results

The cases were declared before device execution. Labels `benefit` and `cost`
record hypotheses, not observed outcomes. All unspecified patterns are mixed;
all unspecified dtypes are FP32. Seeds and cases were not changed after results.

| Case | Mechanism | B | Lmax | D | Pattern | Dtype | Seed |
|---|---|---:|---:|---:|---|---|---:|
| e02-triton-gallery-benefit | Prefix reuse | 512 | 129 | 257 | mixed | FP32 | 83201 |
| e02-triton-gallery-cost | Prefix reuse | 1 | 17 | 33 | full | FP32 | 83202 |
| e02-triton-heldout-benefit | Prefix reuse | 257 | 193 | 129 | mixed | FP32 | 83203 |
| e02-triton-heldout-cost | Prefix reuse | 3 | 5 | 19 | empty | FP32 | 83204 |
| e03-triton-gallery-benefit | Single store | 8 | 1024 | 512 | mixed | FP32 | 83301 |
| e03-triton-gallery-cost | Single store | 1 | 1 | 17 | full | FP32 | 83302 |
| e03-triton-heldout-benefit | Single store | 5 | 777 | 383 | mixed | FP16 | 83303 |
| e03-triton-heldout-cost | Single store | 3 | 5 | 19 | empty | FP16 | 83304 |

The primary metric is the **profiled device operation span**: last owned task
end minus first owned task start, including gaps between the scan and consumer.
The table reports mean ± sample standard deviation of all ten forwards in
microseconds. This metric excludes the first enqueue and is not unprofiled
end-to-end latency. Caller-event timings are retained separately; neither their
samples nor first samples are filtered.

| Case | Before µs | After µs | Before/after |
|---|---:|---:|---:|
| e02-triton-gallery-benefit | 260.778 ± 3.511 | 345.462 ± 18.776 | 0.7549× |
| e02-triton-gallery-cost | 3.594 ± 0.897 | 137.662 ± 68.928 | 0.0261× |
| e02-triton-heldout-benefit | 76.752 ± 0.898 | 238.684 ± 16.771 | 0.3216× |
| e02-triton-heldout-cost | 2.126 ± 0.105 | 162.104 ± 40.531 | 0.0131× |
| e03-triton-gallery-benefit | 21.216 ± 0.350 | 14.168 ± 0.291 | 1.4975× |
| e03-triton-gallery-cost | 1.518 ± 0.079 | 1.508 ± 0.027 | 1.0066× |
| e03-triton-heldout-benefit | 16.206 ± 0.468 | 12.042 ± 0.329 | 1.3458× |
| e03-triton-heldout-cost | 2.194 ± 0.030 | 2.040 ± 0.127 | 1.0755× |

## Validation and limits

Kernel/test revision: `c75871dc5c91f9bd3d41d084b10fa099f0a036f9`.
Observed product: Ascend910B4-1, 20 Cube cores and 40 Vector cores, on a healthy
idle isolated device. Evidence applies to that observed product/runtime rather
than an inferred machine generation.

All eight frozen pairs passed: 16 exact correctness checks, 16 captures,
160 profiled forwards and eight CPU oracle generations. Each implementation
runs correctness once and then ten forwards without input/output inspection.
There are no warmups, discovery launches, unprofiled timing calls or recaptures.
CPU functional validation passed eight tests plus 15 subtests, covering oracle
partitioning, input guards, immutability, independent controls and the adapter
launch boundary. CPU fake kernels are not used as device correctness evidence.

The strict collector validates the same captured task/context identities,
source hashes, compiled binary hashes, ten owned iterations and unchanged
compilation state. All 16 arms have available PMU. Twenty unique actual compiled
binaries, their TTIR, adapter IR and compiler metadata were retained, with no
missing binary snapshots. Original result reports remain unchanged; explicitly
digest-bound analysis views only add the `physical_die` provenance alias from
`physical_device` for collector compatibility.

The retained compact evidence contains original reports, strict evidence,
trace JSON, PMU CSV, normalized/accounted tasks, and compiled artifacts.
Archive SHA256:
`01769f9ba980ba8398a31611b055b563256d295be06ddac586f9dc3dfde94e6d`.
Machine routing, job handles and evidence locations remain in the orchestration
record, outside source control.

Exported per-task pipeline ratios are not additive whole-device utilization.
No calibrated bandwidth or compute ceiling was measured, so these captures do
not justify a distance-to-roofline estimate. The fixed capture order and small
sample count also limit generalization; the complete samples and uncertainty
are retained instead of promoting tiny point differences into established gains.
