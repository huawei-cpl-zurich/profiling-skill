# MTE2 activity, reported bandwidth, useful throughput and compiler lowering

The controlled A3 memory matrix completed successfully: **13 retained jobs,
17 unique cases, 140 BasicInfo timing samples, 84 PipeUtilization and 84 Memory
captures**. No failed captures or retries were discarded. Each comparison uses
an interleaved baseline on the same selected device; baselines from different
devices are not pooled. All raw timing samples, footprints, per-replay counter
summaries and source hashes are in the
[numeric results](data/mte2-controlled-memory-results.json).

## Fixed count: activity is not useful bandwidth

Every case below reads 4 MiB useful FP32 input and writes 4 MiB output using a
common 256 MiB allocation. Strides change accessed span, not useful count.

| Stride | Same-job stride-1 baseline µs | Candidate µs | Useful input GB/s | Baseline/candidate latency | Five paired ratios, min–max | Candidate range/median |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 27.279 | 882.742 | 4.751 | 0.031 | 0.029–0.032 | 1.4% |
| 4 | 28.319 | 800.864 | 5.237 | 0.035 | 0.033–0.040 | 2.2% |
| 8 | 27.579 | 516.330 | 8.123 | 0.053 | 0.051–0.059 | 1.7% |
| 16 | 27.499 | 359.113 | 11.680 | 0.077 | 0.076–0.085 | 0.8% |
| 32 | 27.939 | 270.695 | 15.495 | 0.103 | 0.096–0.110 | 1.5% |
| 64 | 27.639 | 264.455 | 15.860 | 0.105 | 0.099–0.115 | 3.0% |

Useful input GB/s means useful input bytes divided by whole-kernel BasicInfo
median time, in decimal units; it excludes index and output traffic. Stride 2
is about **32.4× slower** than its paired contiguous baseline despite a median
row MTE2 activity ratio near 0.999. Every nonunit stride reports 131,072 KB
GM→UB versus 4,096 KB useful input: a **32× reported amount amplification** if
KB is interpreted as KiB. The measured nonunit-stride activity stays about
0.995–0.999 while useful throughput varies more than threefold.

The reported amount is identical across strides 2–64, but latency improves
toward stride 64. Exported per-row median MTE2 active bandwidth rises from
about 3.09 to 11.19 over that sweep. Thus activity can identify a busy pipeline
without explaining its address-dependent efficiency. Neither the 32× amount
nor these active-bandwidth numbers independently validates physical HBM
traffic, transfer granularity or a saturation ceiling.

TTadapter retains `memref.copy` with the respective stride in all direct
cases, rather than a scalar-load loop. Final transfer instructions and the
cause of the nonmonotonic stride behavior require later compiler evidence;
do not infer them from TTadapter or counters alone. See the companion
[compiler findings](mte2-compiler-findings.md) from PR #129.

## Fixed span: faster can mean less useful work

These cases share a 4 MiB allocation and address bound. Useful input, output
and logical grid shrink with stride; the last touched address is recorded
explicitly rather than assumed identical.

| Stride | Same-job stride-1 baseline µs | Candidate µs | Useful input GB/s | Baseline/candidate latency | Five paired ratios, min–max | Candidate range/median |
|---|---:|---:|---:|---:|---:|---:|
| 2 | 25.539 | 461.911 | 4.540 | 0.055 | 0.054–0.064 | 3.0% |
| 4 | 27.899 | 208.196 | 5.036 | 0.134 | 0.119–0.138 | 5.0% |
| 8 | 29.119 | 76.378 | 6.864 | 0.381 | 0.356–0.398 | 10.6% |
| 16 | 29.279 | 33.339 | 7.863 | 0.878 | 0.797–0.897 | 13.3% |
| 32 | 28.219 | 17.700 | 7.405 | 1.594 | 1.332–1.761 | 28.2% |
| 64 | 27.959 | 13.360 | 4.905 | 2.093 | 1.460–2.172 | 31.4% |

Nonunit strides again report 32× GM→UB amount relative to useful input.
Stride 64 has lower latency but performs 64× less useful input work; it is not
a useful-bandwidth improvement. Do not merge fixed-count and fixed-span
results into one speedup conclusion.

## Runtime indices: low MTE2 can hide costly scalarized memory access

This group shares one device/job and the same 256 MiB allocation. Every case
has 4 MiB useful input and 4 MiB output; indexed cases add 4 MiB indices.

| Runtime access pattern | Direct baseline µs | Candidate µs | Useful input GB/s | Baseline/candidate latency | Five paired ratios, min–max | Candidate range/median |
|---|---:|---:|---:|---:|---:|---:|
| contiguous | 30.099 | 481.470 | 8.711 | 0.063 | 0.055–0.064 | 1.1% |
| local32 permutation | 30.099 | 483.090 | 8.682 | 0.062 | 0.055–0.064 | 0.2% |
| global permutation | 30.099 | 2,378.992 | 1.763 | 0.013 | 0.011–0.013 | 0.1% |

Runtime contiguous indices designate exactly the direct case's addresses,
yet are about **16× slower**. Their TTadapter copies the indices and then
loads actual data with an `scf.for`/per-element `memref.load` loop. Triton's
tensor `tt.load` concealed that split. The three indexed TTadapter artifacts
are byte-identical (SHA256
`4d12d93035f7b95bfa3407532b7f94af71b0a1620a025c93188f0f0153dd8b79`),
so the global-versus-contiguous gap is not a different compiler path.

Contiguous/local32 have median row scalar ratios around 0.955 and MTE2
ratios around 0.046. Global indexing takes another **4.94× longer**, scalar
ratio rises to about 0.991 and MTE2 ratio falls to about 0.010. Exported MTE2
active bandwidth remains about 3.77 in all three and reported GM→UB stays
4,096 KB: that can describe the index copies, not all scalar data loads.
Reported read-main-memory amount rises from 12,294 to 134,702.125 KB with
global permutation. This exposes a locality-dependent observation outside
MTE2 composition, without independently proving physical HBM byte units.

The useful next action is to examine the actual lowered access path and
consider specialization only when contiguity is proven. High scalar activity
does not by itself justify unrolling loop control; low MTE2 activity does not
exclude expensive memory accesses in the scalarized path.

## Counter scopes and uncertainty

Across all 4,032 Memory rows, 16,128 comparisons fit this empirical formula
within 1.384e-5 relative rounding:

`reported_bw = reported_amount_KB × 0.95367431640625 / aiv_time_us`.

The factor is compatible with KiB/GiB units despite the KB/GB/s labels. Its
denominator is full AIV row time, **not MTE2-active time**. PipeUtilization
separately exports MTE2/MTE3 active bandwidth and time. Multiplying active
bandwidth by active time from the **same PipeUtilization row** recovers the
corresponding stable Memory reported amounts across replays within 4.002e-6
relative error (8,064 checks). This validates arithmetic consistency, not the
vendor byte multiplier or simultaneous overlap between separate replays.
All output amounts match logical stores under the KiB-compatible conversion.

There are 48 exported block/vector rows per full capture, distinct from the
logical user grid of up to 4,096 programs. Row medians are descriptive, not
device-wide utilization; per-row bandwidths are not summed into achieved
whole-device bandwidth. All contiguous baselines have 11.9–22.3% range/median.
Short span cases are also noisy. Preserve those flags and all five samples;
paired min/max ranges are not statistical confidence intervals. Nevertheless,
every observed paired ratio is below one for all fixed-count and indexed
regressions: these large qualitative effects do not depend on one selected
baseline sample.

## Reproduce and inspect

Use the [controlled-memory guide](a3-controlled-memory.md) and existing
`$remote-access` runtime/transfer workflow. Source freeze is `ebd2816`, with
all six runtime source SHA256 values in the numeric results. Do not change the
source of an ongoing capture. Run independent groups on eligible devices,
retaining each group's matched baseline and all reports:

```bash
for stride in 2 4 8 16 32 64; do
  python scripts/a3_controlled_memory_batch.py --output "count-${stride}" \
    --case count/1/1048576 --case "count/${stride}/1048576"
  python scripts/a3_controlled_memory_batch.py --output "span-${stride}" \
    --case span/1/1048576 --case "span/${stride}/1048576"
done
python scripts/a3_controlled_memory_batch.py --output indexed \
  --case count/1/1048576 --case indexed/contiguous/1048576 \
  --case indexed/local32/1048576 --case indexed/global/1048576
```

Observed case identity: Torch `2.10.0+cpu`, Triton module `3.2.0`,
`triton-ascend` distribution `3.2.2`, installed `triton` distribution `3.5.0`,
compiler target `GPUTarget(backend='npu', arch='Ascend910_9382', warp_size=0)`.
Distribution and imported module versions are deliberately not conflated.
Exact profiler/CANN build was not established (`msprof --version` unsupported),
so version-specific units and physical-byte formulas remain unvalidated.
All captures used `msprof op`, exact discovered selector `a3_memory_probe`,
warm-up 0, launch-count 1 and kill off. Timing used separate BasicInfo replays,
never substituted diagnostic replay durations or forced-compilation timing.

Compact raw CSV/summary/first-correctness text-IR archive, retained centrally:
`mte2-full-memory-analysis-20261017.tar.gz`, 1,231,306 bytes, SHA256
`906f91a99215756f17ebf2d6b784255c0ccff897e950b83fbc7c971a59e154b6`.
All 586 manifest-bound files were verified. Raw profile trees, binary artifacts,
machine paths and routing details are not committed. No general skill benefit,
physical peak bandwidth or exact hardware transaction-size claim follows from
this matrix alone.
