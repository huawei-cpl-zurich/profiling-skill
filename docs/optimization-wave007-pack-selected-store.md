# Wave007 T01: select packed input before one store

The selected-store arm passed exact IEEE-bit checks on all six frozen cases and
had shorter mean profiled operation spans on all four main cases. This is a new
operator application of an existing store-selection mechanism, with no claim of
mechanism novelty.

| Frozen case | Before span µs, mean ± sample SD | After span µs, mean ± sample SD | Before/after |
|---|---:|---:|---:|
| W007-T01-01 | 20.566 ± 1.061 | 12.744 ± 0.428 | 1.614 |
| W007-T01-02 | 9.428 ± 0.273 | 8.034 ± 0.383 | 1.174 |
| W007-T01-03 | 9.758 ± 0.259 | 6.720 ± 0.345 | 1.452 |
| W007-T01-04 | 1.504 ± 0.041 | 1.366 ± 0.059 | 1.101 |

Each row contains ten samples per arm. These are diagnostic spans measured while
Torch-NPU profiling was active; they do not establish ordinary application
latency, physical memory traffic, roofline position, or occupancy.

The source is derived from Huawei/vLLM's MIT-licensed `index/pack_unpack.py`,
pinned to `7acb9c40a6ca827e885d902b7eaee5d45457b230`. The baseline stores padding
then overwrites valid rows. The candidate selects input or padding before one
store. Both retain the masked input loads (`other=0`), cumulative-length scan,
grid, FP32 data, 64×64 tiles, and common compiler flags. Full selected `.npubin`
identities differed for all five nonempty cases. A subsequent read-only ELF audit
confirmed distinct executable `.text` bytes for all five pairs, with no relocation
sections in either arm; the empty compound guard emitted no kernels. The audit
made zero new device forwards or oracle calls. Local objdump could not decode the
architecture, so no instruction/store count is asserted. Compiler artifacts and
actual resource metadata were retained before any mathematical launches. No
optimizer settings were changed to force a result.

The gz-a3 run used healthy idle physical device 0 and logical device 0 on
Ascend910B4-1, CANN 9.1.0, imported Triton 3.2.0, Triton-Ascend 3.2.2, and
Torch/Torch-NPU 2.10.0. Distribution metadata reports Triton 3.5.0 alongside the
imported version; both observations are retained. Common flags were
`num_warps=4, num_stages=1, multibuffer=False, enable_fp_fusion=False`.

Six CPU oracle payloads were generated once and cached. Each arm made one
correctness API call per case, then ten profiled API calls per main case. The two
guards made zero captures. Totals were 12 correctness API calls, 14 pack
components (the empty guard also returns an empty batch), 10 correctness kernel
launches, 80 profiled API/kernel calls, and 8 captures. There were no warmups,
autotuning trials, or extra timing forwards. Inputs and outputs were prepared
outside the kernel forward; validation inspected exact output bits, padding,
input immutability, output separation, and output-storage canaries only outside
captures. Signed zero and ±Infinity passed, and empty outputs launched no kernel.

Capture and synchronous export used the shared two-slot locks and a 600-second C
watchdog with a sibling mmap API-entry counter. Correctness API entry also had a
watchdog/counter, but its subsequent validator CPU synchronization was not
watchdog-bounded. All validations completed; no additional frozen forwards were
submitted to change this completed collection. The first compact-only collector
failed on an invalid glob; its corrected successor read unchanged evidence and
made zero new device forwards or captures.

[Digest-bound samples, raw supported PMU rows, compiler identities, fixtures, and
compact manifest](data/optimization-wave007-T01.json) retain all samples and
unavailable fields. Full vendor reports remain on the target. Collection handle:
`remote:gz-a3:job:0f3a3cf2-ffc5-4407-b775-65e3ad637221` (completed, exit 0).

Host tests use nonfrozen seeds/shapes, exercise the independent CPU oracle, reject
invalid specs and tensor layouts, and invoke the public CLI from an external
working directory for both flat and nested specifications. Public usage:

```sh
python examples/optimization-library/triton/pack_selected_store.py \
  --case CASE.json --output NEW_OUTPUT --cache CACHE --provenance PROVENANCE.json
```

Use `--validate-only` to validate host case metadata without accessing the cache
or importing the device runtime. The operator envelope is contiguous FP32 input,
contiguous int32 nonnegative lengths with matching token count, positive feature
width, and products within the signed address range.
