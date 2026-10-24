# Q/K normalization placement in a bounded gated recurrence

All four frozen main pairs passed the independent CPU FP64 output/final-state
gate (`atol=1e-4`, `rtol=2e-4`) and completed ten profiled full-API calls per arm.
The largest value-width case has more summed device work after fusion even
though its complete operation span is shorter. This is an application of an
established fusion mechanism; it makes no novelty claim.

| Case | Before span μs | After span μs | Before summed tasks μs | After summed tasks μs |
| --- | ---: | ---: | ---: | ---: |
| W007-T04-01 | 204.364 | 12.850 | 17.180 | 12.850 |
| W007-T04-02 | 359.590 | 15.866 | 16.014 | 15.866 |
| W007-T04-03 | 215.766 | 13.250 | 15.232 | 13.250 |
| W007-T04-04 | 234.280 | 23.022 | 22.098 | 23.022 |

These means are diagnostic profiler measurements with no warmups. The primary
span includes the gap between normalization and recurrence: before-arm mean
gaps were 187.184, 343.576, 200.534, and 212.182 μs. They are not ordinary
allocation/end-to-end latency, physical traffic, occupancy, or roofline evidence.
Each before iteration has exactly two source-selected device tasks; after has
one. Fusion repeats normalization across value tiles and grouped value heads;
the last case's larger task sum is compatible with that cost hypothesis,
without proving a specific hardware cause.

The fixture provider generated six CPU payloads once. Each arm ran correctness
once per case (12 calls); the four main pairs made eight captures/80 profiled
API calls. Both guards passed their actual numerical checks with zero captures.
**GA has partial qualification:** its all-zero row and signed unequal near-zero
row `[+1e-4,-2e-4,+3e-4,-4e-4,+5e-4]` omit the frozen request's exact `-1e-4`
component. This gap is preserved explicitly; fixtures were not changed or
rerun after measurement. A separate nonfrozen CPU unit test covers exact
±1e-4, which establishes no hardware coverage for the missing component.

The scoped source comes from `moe/fused_recurrent.py` at
`7acb9c40a6ca827e885d902b7eaee5d45457b230` (filtered source SHA256
`cfd98e9e446b3535a6d12cb65c9f1aa9024757631c9b7f0c344a5bd8542cc43f`).
The Huawei, vLLM, Songlin Yang and Yu Zhang attributions and MIT license are
retained. Optional KDA, varlen, speculative/continuous batching, headwise beta,
missing initial state and inplace output modes are outside this interface.
The study adapter accepts B=1..4, T=1..8, 1≤H≤HV≤4 with H dividing HV,
K=1..64 and V=1..512; it does not claim a broader vendor API. Both arms use
the same recurrence, grouping, tile grid and compiler flags.
Before normalizes both Q and K on the device on every call using the exact
source `x / sqrt(sum(x*x) + 1e-6)` expression, with FP32 intermediates; after
normalizes inside the recurrence. Inputs, initial state and output canaries
are checked. `prepare` compiles and allocates; its returned `run` includes all
required normalization work on every invocation with prepared buffers.

The A3 run passed fresh health/isolation checks on physical device 0, mapped
to application logical 0. All 12 arm/case compilations and actual ELF `.text`
plus relocation screens preceded correctness. The imported Triton module
reports 3.2.0, selected CANN is 9.1.0, and package metadata reports `triton` 3.5.0 and
`triton-ascend` 3.2.2; both observations are retained. The compiler artifacts,
source/cache identities, all raw samples and numeric supported PMU fields are in
[T04 results](optimization-wave007-T04-results.json). The retained 366-file
compact archive has SHA256
`7b85e3d85c5db135b1838d6d6d66b1e29d35fd1201c4a2f11b540143b7fa7de1`;
all members were verified after transfer. Vendor reports remain retained.

The standalone public CLI works from an external current directory:

```sh
python /path/to/checkout/examples/optimization-library/triton/recurrent_norm.py \
  --case /path/to/case.json --output /path/to/new-output \
  --cache /path/to/oracles --provenance /path/to/provenance.json
```

Case JSON accepts either the frozen `parameters` object or flat dimensions
`B,T,H,HV,K,V`, plus a nonnegative `seed`. Conflicting forms and unsupported
bounds/modes reject before the generic cache through this standalone CLI.
Explicit supported False/None defaults are accepted in either form. The generic
`optimization_pair.py --adapter` CLI may inspect an existing cache before the
adapter rejects an unsupported case in `prepare`; it is not the documented
pre-cache validation entry point. Guard-role cases automatically
perform correctness only. Use the orchestration controller's device lease,
watchdog and export locking for actual collection.

Validation: four functional unit tests passed on the capable A3 runtime, then
all six case/arm numerical gates and the four capture pairs completed in one
retained science job. Five final nonfrozen unit tests pass locally and in a separate CPU-only A3
job, including
the explicitly CPU-only exact-signed-epsilon test. No extra science was run
for that test addition. The frozen source, payload and binary bindings remain
unchanged.

The publication host-parser fix changes no scientific inputs or kernel bodies.
The measured source SHA remains preserved in the results; a separate interface
revision receipt records the current module SHA and unchanged kernel-function
text SHA. The frozen FP64 CPU provider is byte-identical.
