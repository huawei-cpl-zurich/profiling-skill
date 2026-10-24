W04 compares a serial loop-carried inclusive recurrence with `tl.cumsum` along
the same chunk axis. Both use contiguous FP32 `[B,T,H]`, the same
`(ceil(T/chunk), B)` grid, independent heads, and scale one. Output allocation
and input transfer occur before measurement; the measured call performs every
prefix. There is no scratch, input-dependent preprocessing, warmup, or autotune.
The nondivisible T=70 guard uses the same serial masked fallback in both arms,
including all six tokens in its final chunk.

The exact domain is integers in [-8,8] divided by 1024, chunk64/128. All partial
sums have exactly representable integer numerators, so both implementations
must equal the independent CPU FP64 per-chunk oracle cast to FP32 (signed zero
compares equal). No arbitrary noninteger FP32 bitwise-equivalence claim follows.
The reset guard has +1/1024 then -1/1024 chunks and two independent heads.
Four frozen profile cases and two correctness-only guards live in
`../cases/wave004/W004-W04-*.json`. The paired runner checks each arm once and
profiles exactly ten full forwards per arm for each non-guard case.

This computes prefixes differently from the earlier E02 optimization, which
reused prefix offsets across consumers. It does not combine reverse-prefix
identities, layout changes, scaling, chunk sizes, or grid tuning with the scan.

The source-backed mechanism is adapted from `triton-ascend-kernels`, revision
`7acb9c40a6ca827e885d902b7eaee5d45457b230`,
`src/triton_ascend_kernels/attention/fla/cumsum.py` lines62–93 and wrapper97–177;
related tests are `tests/attention/test_chunk_local_cumsum_fla.py` lines1–70.
Stable managed-library references:
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#src/triton_ascend_kernels/attention/fla/cumsum.py:L62-L93`
and
`ref://profiling-skill/triton/perf-analysis/repos/optimization-wave1-triton-kernels-7acb9c40a6ca#tests/attention/test_chunk_local_cumsum_fla.py:L1-L70`.
The file-level Apache-2.0, Huawei, sgl-project, vLLM and FLA/MIT attribution is
preserved in the kernel file. The original source's arbitrary-tail boundary
checks are not relied on for the partial-chunk fallback.

Validation: `python -m unittest discover -s tests -p test_optimization_chunk_scan.py -v`.
Device evidence must identify the exact source version, frozen case contract,
compiler artifacts, kernel names, runtime, healthy physical device and profiled
reports. Profiled operation span includes gaps; active duration and gaps should
be reported separately. These diagnostic captures do not establish application
latency, physical memory traffic, or a roofline claim.

The retained A3 run uses identical supported compiler scheduling options
`num_stages=1, multibuffer=False` in both arms. With the backend defaults, the
first serial correctness check failed (maximum error0.015625), before any
capture. A static-unroll attempt exceeded the bounded compile budget and was
cancelled with zero forwards; it is not performance evidence. The scheduling
configuration then passed every case exactly. A DMA/storage-lifetime cause for
the original mismatch remains an inference. All prior failures are retained.

The successful A3 gate compiled both arms for all six shapes before any
forward, passed four functional CPU tests and all twelve NPU correctness
checks, and captured exactly ten full forwards per arm for each of four cases.
Both guards had zero captures. Same-capture kernel identity and PMU joins
passed. [Digest-bound samples and compiler identities](chunk_scan_evidence.json)
retain every sample, exported PMU schema, source/binary/report hashes, fixture
identities, correctness diagnostics and negative attempts.

| Frozen case | Serial span µs | Parallel span µs | Before/after |
| --- | ---: | ---: | ---: |
| 01 gallery-benefit | 41.308 | 6.652 | 6.21 |
| 02 gallery-cost | 10.586 | 2.602 | 4.07 |
| 03 heldout-benefit | 44.032 | 22.382 | 1.97 |
| 04 heldout-cost | 10.618 | 2.314 | 4.59 |

Each value is the mean of ten profiled device-operation spans. Each operation
has one kernel, so active union equals span and inter-kernel gaps are zero.
The cost designations are frozen hypotheses: neither designated cost case
regressed in this run. These numbers describe these instrumented dyadic cases
under the recorded scheduling options, not arbitrary inputs or application
latency. No extra evaluation or unprofiled timing was performed.
