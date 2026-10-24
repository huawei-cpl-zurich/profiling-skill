# F12 native weight-cache-hint capability screen

F12 is **runtime-contract-unqualified**. This directory provides a device-less
compiler reproduction, not a runnable optimization or a passed performance
family. All four fixed-tuple cases remain unattempted: zero CPU oracles, zero
scientific forwards, zero captures. No benefit or cost is established.

The original native MSD operator calls
`weight.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE)` when its M partition count
is one, then passes that tensor to `Matmul.SetTensorB`. That is a real source
mechanism. It does not establish how a standalone CANN 9.1 wrapper enables the
runtime cache configuration.

The installed SDK's non-3510 `L2CacheAlter` changes the address only with
`L2_CACHE_HINT` and architecture 2002/2201. It reads a runtime cache offset.
The observed standalone device predefines select architecture 2201 and the
external `g_opSystemRunCfg` branch, rather than the `__NPU_DEVICE__` feature-3
configuration branch. Neither the bounded installed SDK/build-example search
nor the filtered source research established a supported initialization and
launch contract. Defining the macro alone is a diagnostic, not that contract.
Do not supply a guessed offset, initialize internal globals, or encode GM
pointers manually to turn this screen into a workload.

## Reproduce the compile screen

Use the installed `remote-access` skill to discover a capable A3 target and its
named runtime. Upload this directory and the behavioral test through that
interface. Run inside the registry workspace with a new output directory:

```bash
python -m unittest discover -s tests -p test_optimization_cache_hint_screen.py -v
python examples/optimization-library/cache-hint/compile_screen.py --output evidence
```

`ASCEND_HOME_PATH` and `bisheng` come from the chosen runtime; `--sdk` and
`--compiler` may select an explicitly provisioned installation. The driver
retains exact compiler argv, compiler/source hashes, device predefines, logs,
artifacts, and `screen.json`. Output directories cannot be reused. Each compiler
process has a timeout. Failed builds, timeouts, missing device sections, and
malformed artifacts remain distinct from identical or different payloads.
A zero exit status means all four builds produced inspectable device payloads;
it does not qualify the runtime or certify numerical correctness.

The probe instantiates an int8/int8-to-int32 native Matmul consumer for stacked
component rows. Only its weight GlobalTensor hint changes between arms. It does
not implement FP16 row decomposition, set a concrete tiling, reconstruct FP16
outputs, allocate tensors, register a host launch, or run kernels. Therefore,
compilation is not validation of the complete A16W8 MSD path. The macro-enabled
pair additionally diagnoses the SDK conditional; it never mutates cache globals.

Comparison hashes cover nonempty `.aicore_binary` and
`__aicore_rel_binary` sections, excluding unrelated host code. Different device
payloads still leave the runtime contract unqualified. Identical payloads do
not establish blanket hardware or public-API non-support. Keep the four
performance cases unattempted until an evidenced supported enable contract and
full numerical implementation are available. Profiling identical compiled arms
would not isolate the proposed physical intervention.

## Source and evidence provenance

The source mechanism is from ops-transformer revision
`4b92adc974cade0ae510e3b6265320e5f6fbc380`, under CANN Open Software License
Agreement 2.0. This small probe uses the public SDK API; no vendor headers are
vendored here. Exact filtered contexts:

- Hint call: `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-transformer#gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a16w8_msd.h:L480-L491`
- Matmul consumer: `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-transformer#gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a16w8_msd.h:L765-L807`
- Local build options: `ref://profiling-skill/ascendc/perf-analysis/repos/gitcode-cann-ops-transformer#gmm/grouped_matmul/op_host/CMakeLists.txt:L11-L70`

Runtime-contract research query `6b3db3ba5a6a4693b4a9fff339734ef3` retained
bounded miss `58c446d63c6a796ae5f435fb458b8668`. A research miss is not a claim
that the contract cannot exist elsewhere. `evidence.json` records the selected
A3 compiler gate and its limits; complete logs and artifacts remain with its
retained remote job handle.
