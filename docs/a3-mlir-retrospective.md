# Retrospective A3 compiler audit

This audit does not rerun old kernels, correctness checks, timings, or profiling.
The frozen 70-cell campaign maps to 53 distinct source-dispatch specializations.
The three amplification counts are runtime values, not distinct compiled kernels.
All 36 retained amplified artifacts (18 cells × TTIR/TTAdapter) have been fetched
from their precise discovery dispatch and verified against the original SHA-256.

## Findings from original artifacts

These findings apply to the amplified controls, not automatically to the
non-volatile originals. TTAdapter is not final scheduled machine code.

| Change | Verified compiler behavior | Consequence for profiler interpretation |
|---|---|---|
| Amplify repetitions 1/32/128 | One `scf.for` bounded by a runtime `i32` load survives in TTIR and TTAdapter; all three counts share the same artifact hashes for each variant. | This really repeats a compiled body; however volatile accesses change the compiler contract relative to the original control. |
| Per-lane → scalar mask decision | Before performs three `memref.copy` operations per loop. After retains two copies plus a scalar `memref.load i8`, tensor-of-one select/extract, and conditional control. | The scalar spelling is not a universally cheaper broadcast. More scalar activity is consistent with this change, but early IR cannot prove exact scalar instruction or stall counts. |
| Load then mask → predicated load | After still copies the contiguous valid FP32 slice, then performs `arith.andi` and vector `arith.select`. | Do not explain lower MTE2 time as proven reduced physical traffic. The predicate does not remain a sparse gather in this lowering stage. |
| Runtime → specialized sparse-mode predicate | Before retains a `memref.load i64` inside the repetition loop; after does not. | This removes a runtime scalar load/predicate dependency at this stage, but does not guarantee a visible latency improvement. |

Exact original TTAdapter references (line numbers in each hash-addressed file):

- Scalar before `7c546ecc66929b6d07a2de3748290dd661b9ec70f5751852a3910c138e67d53e`: loop line 28, copies 40/47/54. After `e23952bf9fce28c10d40b182c41d11cdf39c1f52bfff503bec5cc2ba601088ff`: loop 28, copies 40/47, scalar load 49, select/extract 55/56.
- Mask before `565ccfc5ed8475e518da03186206f4e91720880b70e6685f03c39c75b7c6a648`: FP32 copy 48, final select 52. After `a0b4ae6b04f20e9962dcabb097f6f74e6900665e94dca90843fed98355e284d2`: FP32 copy 61, predicate combine 63, selects 64/67.
- Sparse before `dda3073619549b2d8c18d576578c86883abcdf289709e5832aa974faadcb4b3c`: loop 25, runtime `i64` load 49. After `b710d9955692c3992a075677566934c6b136a350a1632800f611daa078536462`: loop 24, no corresponding runtime-work load.

## Reproduction interface

`scripts/a3_mlir_retrospective.py plan --run RUN_JSON --manifest FROZEN_MANIFEST --output PLAN_JSON`
builds a selector-checked cell-to-specialization map and source fingerprints.
`import-originals --records RECORDS_JSON --output DIRECTORY` verifies and imports
compact transferred original IR. Originals are kept in ignored evidence storage,
not committed as compiler dumps or binaries.

`compile` accepts `--plan`, `--key`, `--config`, `--source`, and `--output`;
`compile-all` omits `--key`. Configuration requires an explicit `GPUTarget`,
compiler options, and an `AttrsDescriptor` serialization. The named
`aligned-pointers-reconstruction` option is explicitly unverified, not recovered
metadata. Target architecture must come from evidence, not a product nickname.

`capture --helper SKILL_CAPTURE_HELPER --output NEW_DIRECTORY -- COMMAND...`
clones Python runtime source, copies all selected executable consumers into an
owned tree, and invokes the skill-owned helper against that tree. Runtime
resources remain read-only links. It never shims shared compiler paths and does
not use compiler-path environment overrides. Activate the named runtime first.

Compile-only uses `triton.compile(ASTSource(...), target=..., options=...)` with
JIT `run` and `warmup` guarded to raise. Source file hashes are checked before
imports. A failed specialization is retained explicitly and does not erase
successful artifacts from the remaining batch.

Recompilation is labeled **unverified reconstruction** unless original
specialization metadata or matching retained artifacts establish equivalence.
Backend inputs, invocation stderr and emitted post-synchronization IR provide
the next evidence layer; a TTIR `tt.dot` alone is not proof of a Cube operation,
and `memref.copy` alone is not proof of a particular MTE or byte count.
