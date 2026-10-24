# Sinkhorn live width with padded outputs

E04 applies a width reduction to the MindSpeedOps arch32 Sinkhorn forward at
`4efe2b675c2d51517c3fdd14c721020b8d41e241` (Apache-2.0). This is a new
source-backed operator application; it makes no unique mechanism claim.
The upstream test file has a per-file MIT header. Our independent FP64 oracle
uses its mathematical contract. The exact source test's forward reference governs the math: row softmax,
entrywise epsilon, initial column normalization, then `iters-1` row/column
normalizations. Older documentation pseudocode omits steps and is not the oracle.

Both arms use GROUP32; the exposed source default GROUP48 is unsupported by
power-of-two Triton aranges. Both inline the common sigmoid formula `1/(1+exp(-x))`. The selected namespace
does expose sigmoid; a device-free source probe confirmed that its definition
is exactly `return 1 / (1 + math.exp(-x))`. This helper inlining was applied
before math and retained without a rerun; it is not a missing-API claim. These bounded adaptations mean the
baseline is source-style rather than a claim about the vendor default binary.

Every prepared API call launches the same pre/post kernel and one comb kernel.
The before arm computes 4x8 masked columns; after computes 4x4. Both retain
32 FP32 output floats per batch entry and a 4x4 comb view with strides
`(32,8,1)`. After explicitly writes all four padding columns to positive zero.
There is no fusion, loop binding change, removed padding traffic, or group tuning.

Admission requires owned contiguous CPU FP32 fixtures: mixes `[BS,24]`, scale
`[3]`, base `[24]`, positive integer BS, integer iterations 1..20, finite epsilon
1e-6..1e-3. Mixes lie in [-3,3], scale/base in [-1,1]. Empty BS is rejected
before cache creation. Flat/nested parameters accept role/expected metadata.
The adapter copies admitted inputs once to device, allocates private outputs with
canaries once, and reuses outputs across full forwards. Allocator cost is excluded.
All three results use the fixed `atol=rtol=3e-6` FP64 CPU source-test oracle gate;
validation checks input immutability, canaries, padding bits and comb layout.
Common pre/post output bits are compared using the two existing correctness calls.

For standalone host admission from any directory:

```sh
python /path/to/examples/optimization-library/triton/sinkhorn_width_cli.py \
  --case case.json --output out --cache cache --provenance provenance.json \
  --validate-only
```

Remove `--validate-only` for the generic pair runner on a supported NPU runtime.
The frozen collection contract is `experiments/optimization-wave007/E04.json`.
Its six fixture/oracle payloads are cached once; correctness runs once per arm
and case. Four main pairs have ten profiled full forwards per arm; guards have
zero captures. Compilation/resource and actual code identity screens precede
numerical execution. Profiled operation spans are diagnostics; they do not
establish ordinary application latency, physical DRAM traffic, occupancy, or a
roofline. All samples, unsupported metrics and failures must remain visible.

E05 launch fusion and E06 loop binding are separate dependent studies and must
retain this qualified eight-lane baseline unchanged.

## A3 result

The original retained job `remote:gz-a3:job:cf7339a0-c480-420a-a8bf-26beb23ce357`
completed with exit 0 on healthy idle physical0 (logical0), gz-a3 py311-torch.
All six cases passed both arms, including visible epsilon1e-3 equal logits
and the partial GROUP32 guard. Twelve correctness forwards and eighty profiled
full API forwards launched 184 component kernels; guards had zero captures.
All six comb binaries were code-distinct; pre/post binaries matched exactly.
Five functional tests passed locally and on the A3 host.

| Case | Before mean span (µs) | After mean span (µs) | Before / after |
| --- | ---: | ---: | ---: |
| gallery BS1024, iters20 | 166.062 | 262.864 | 0.632 |
| gallery BS1, iters1 | 178.390 | 182.216 | 0.979 |
| heldout BS4097, iters20 | 222.316 | 553.304 | 0.402 |
| heldout BS33, iters4 | 179.012 | 200.066 | 0.895 |

The live-width reduction regressed this profiled span in every main case.
These compound spans include the inter-launch gap and profiler instrumentation;
they do not isolate comb instruction throughput or establish ordinary latency.
No cases, tolerances, compiler flags, or timing repetitions were retuned.

[Compact evidence](data/optimization-wave007-E04.json) retains every sample,
raw supported PMU field, source/fixture/binary identity and compiler resource.
The 332-file retained archive is 1,512,174 bytes, SHA256
`70a4fef0a61b3cc5a39df9afcdb1468d047d46e1577769c164c33c46293ca7eb`.
All manifest file hashes verified after transfer. Vendor reports remain remote.

## Public runner lifecycle correction

The publication host adapter now scopes common pre/post bit snapshots to each
output parent and case specification. The first arm reports a pending comparison;
the second compares in either arm order and drops the completed pair. Reusing an
adapter across seeds and shapes no longer compares unrelated outputs. This fixes
the review finding without changing the original kernel, CPU provider, six cached
scientific payloads, compiled binaries, correctness calls or profiled samples.
The evidence records the original scientific adapter hash and the publication
host adapter hash separately. Six CPU functional tests cover public runner reuse
and reversed order, including one-ULP pre/post mismatches that pass the numerical
tolerance and fail the bit gate. No scientific execution was repeated.

The CPU-only A3 follow-up gate completed with exit 0 at
`remote:gz-a3:job:426cba5a-cc39-4d33-b943-a9b6b815bd06`; all six tests passed
in 45.248 seconds. Device movement and kernels were CPU boundary doubles.
