# PyTorch ACLNN BSA/GDN smoke benchmark

`scripts/benchmark_aclnn_bsa_gdn.py` checks deterministic FP16 synthetic
inputs against CPU FP32 golden results before writing a successful JSON report.
It targets A2/A3. This is small operator smoke coverage, not the full 50 cases
or a comparison against frozen BSA/GDN baselines.

Use a matched CANN, PyTorch and torch-npu installation. BSA additionally needs
MindIE-SD (`import mindiesd`) with `torch.ops.mindiesd.rainfusionattention`
registered. GDN needs a built `fla_npu` extension exporting
`fla_npu.ops.ascendc.chunk_gated_delta_rule_fwd_h`. An installed Python module
alone does not establish that its custom operator binaries match CANN. Version
strings are retained in the result; missing operators or incompatible binaries
raise an error rather than switching implementations.

FLA-NPU v26.6.0 exposes this same stage through a raw-op wrapper using
`transpose_state_layout=False`, `save_new_value=True`, and `use_exp2=False`.
The benchmark supports this legacy schema and head-major h output
`[B,H_v,N_c,K,V]`, verified on A3. The release tag resolves to commit
`39bbde7c55f2d9c4777a50552ae42a5e67bdc295`.
It rejects the newer `state_v_first` schema: the public contract at commit
[`a7c608f2e3f828779ceeec1fcf155a6e37ede83c`](https://github.com/flashserve/flash-linear-attention-npu/blob/a7c608f2e3f828779ceeec1fcf155a6e37ede83c/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md)
specifies chunk-major h `[B,N_c,H_v,K,V]`, which this benchmark has not
validated on device. Schema inspection fails before dispatch for that interface.
The package distribution is `flash-linear-attention-npu`; its import is
`fla_npu`. See the [pinned v26.6.0 public schema](https://github.com/flashserve/flash-linear-attention-npu/blob/39bbde7c55f2d9c4777a50552ae42a5e67bdc295/torch_custom/fla_npu/npu_custom.yaml).

Run in the runtime selected through the user-wide remote-access skill, after
selecting a healthy idle physical device. `--device` is the logical device
visible to the process (normally 0 after masking):

```bash
python scripts/benchmark_aclnn_bsa_gdn.py --operator bsa --selected-blocks one --output bsa-one.json
python scripts/benchmark_aclnn_bsa_gdn.py --operator bsa --selected-blocks all --output bsa-all.json
python scripts/benchmark_aclnn_bsa_gdn.py --operator gdn --output gdn.json
```

Defaults are 256 tokens, 2 heads, dimension 128, 20 warmups and 10 samples per
timing method. BSA uses 128x128 blocks and selects the first block or all blocks
for every query block/head. Thus default runs cover two query/KV blocks.
`--tokens`, `--heads`, and `--dim` configure inputs; the installed BSA binary
must support the selected dimensions. GDN accepts dimension 128/256 and full
chunks with `--chunk-size 64` (default) or 128. Batch is one, K=V, no grouped
heads, no variable sequence lengths, and `transpose_state_layout=False`.

GDN measures **ChunkGatedDeltaRuleFwdH only**, with nonzero initial state and
small random k/w. It does not call the full fused GDN operator. The golden
stores each chunk's starting state, computes residual `u - w @ state`, and
updates state with `k.T @ residual`. The scalar cumulative natural-log gate
is zero, so all gate exponentials are one. This deliberately does not test
nonzero gate preprocessing or decay. All three outputs (h, v_new, final_state)
are checked; BSA checks the attention output. Tolerances are atol=rtol=0.02.

Timing excludes CPU golden construction and correctness transfers.
Before any timing, normal mode launches and validates one correctness call.
Separate sample loops collect synchronized host wall time and NPU Event elapsed time;
the latter is the elapsed stream interval for the operator call, not an
individual kernel's profiler duration. JSON retains every sample in microseconds,
arithmetic mean and sample standard deviation (null for a single sample).

For an external profiler, replay runs exactly warmups plus one operator call,
with no host timing or NPU events:

```bash
msprof op --application="python scripts/benchmark_aclnn_bsa_gdn.py --operator bsa --warmup 20 --replay --output bsa-replay.json" --output=profile-bsa
msprof op --application="python scripts/benchmark_aclnn_bsa_gdn.py --operator gdn --warmup 20 --replay --output gdn-replay.json" --output=profile-gdn
```

Check the installed `msprof op --help` and use the ascend-profiling skill for
capture options, kernel selection and interpretation. Replay output has empty
timing and still checks correctness after the final synchronization. A profiler
capture may include warmups; select the intended final operator invocation
explicitly. Dispatch and observe remote work through `cpl-remote`, preserving
the returned handle after interruptions.

Public contracts:

- [MindIE-SD RainFusion tests](https://gitcode.com/Ascend/MindIE-SD/blob/master/tests/plugin/test_rainfusionattention.py)
- [CANN RainFusionAttention API](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/API/aolapi/context/ops-transformer/aclnnRainFusionAttention.md)
- [FLA-NPU ChunkGatedDeltaRuleFwdH contract](https://github.com/flashserve/flash-linear-attention-npu/blob/main/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md)

These evolving public interfaces require runtime verification with the installed
extension revisions; CPU unit tests alone are not device evidence.
