# A3 generated synchronization: find the dependency that waited

An explicit source barrier is only one possible origin of a wait. The compiler
adds synchronization after it knows producer/consumer pipes, aliases, memory
reuse and Cube/Vector handoffs. Inspect the captured compilation for the
measured specialization, not a nearby kernel or only its early Triton IR.

## Capture and locate the relevant stage

Retain compiler/version/target, invocation arguments, source hash, launch shape,
and hashes of artifacts from the actual compilation. The paired example's
`--artifacts DIR` saves whatever `compiled.asm` exposes; absence of a late stage
is a recorded gap, not proof that the stage has no waits. To inspect later
passes, use the installed compiler's supported IR-dump switches and retain the
real invocation. For the included controls, the concrete acquisition is:

```bash
python3 examples/a3_paired_probes.py --case matmul --variant baseline \
  --matrix-size 512 --device 0 --artifacts /absolute/compiler-artifacts/stages1
python3 examples/a3_paired_probes.py --case matmul --variant candidate \
  --matrix-size 512 --device 0 --artifacts /absolute/compiler-artifacts/stages2
```

For an existing Triton launcher, the same functional interface is:

```python
compiled = kernel[grid](*arguments, **compile_options)
artifact_directory.mkdir(parents=True, exist_ok=True)
for stage, content in compiled.asm.items():
    data = content if isinstance(content, bytes) else str(content).encode()
    (artifact_directory / stage).write_bytes(data)
```

Use a fresh artifact directory per specialization, retain SHA-256 hashes and
bind them to source/options and the measured dispatch. The included helper
sanitizes stage names into filenames and records those hashes. Available keys
and pass positions depend on the installed compiler.

### Verified late-IR diagnostic capture

On the measured Triton-Ascend 3.2.2 backend (Triton 3.5.0,
`bishengir-compile 1.2.0`, revision `8796a8ac1508`, CANN 9.1.0), ordinary
`compiled.asm` exposed only `ttir`, `ttadapter`, and `npubin`. **It did not
contain late NPUIR.** Setting the environment variable `TRITON_DEBUG=1` alone
did not produce that missing stage.

The verified route is to enable the JIT's actual debug option for a diagnostic
compilation:

```python
@triton.jit(debug=True)
def kernel(...):
    ...
compiled = kernel[grid](*arguments, **compile_options)
```

For the included probe, create a diagnostic copy with both JIT decorators
enabled, retaining the original timed source unchanged:

```bash
sed 's/@triton.jit$/@triton.jit(debug=True)/' \
  examples/a3_paired_probes.py > /absolute/diagnostic_workload.py
sha256sum examples/a3_paired_probes.py /absolute/diagnostic_workload.py
```

The diagnostic copy has a different source hash; retain both source hashes and
compare compiled binaries. Run it with fresh cache and dump locations:

```bash
TRITON_DEBUG=1 TRITON_ALWAYS_COMPILE=1 TRITON_DISABLE_CACHE=1 \
TRITON_CACHE_DIR=/absolute/evidence/debug-cache \
TRITON_DUMP_DIR=/absolute/evidence/debug-dump \
python3 /absolute/diagnostic_workload.py --case matmul --variant candidate \
  --matrix-size 512 --artifacts /absolute/evidence/debug-asm
```

The backend's `debug=True` option enabled
`--bishengir-print-ir-after=hivm-graph-sync-solver` and saved
`kernel.npuir.mlir` under the dump directory's specialization subdirectory.
Inspect its header: the verified files say
`IR Dump After GraphSyncSolver (hivm-graph-sync-solver)`. The flag alone is not
a portable pass contract; another compiler version can change the pass or
available dump. Keep this as a diagnostic compilation, recheck correctness,
and compare `npubin` hashes before attributing it to the measured kernel.
Do not treat diagnostic execution time as ordinary timing.

The retained paired captures used the same example source and shapes and
produced byte-identical `npubin` in ordinary and debug-enabled compilation for
each variant. [Compiler evidence metadata](../examples/compiler-evidence/metadata.json)
records the two binary bindings, source/version identity, raw dump hashes and
display hashes. The displayed complete dumps remove trailing compiler warnings;
the actual IR and its line numbers are retained.

A practical search in the retained artifact directory is:

```bash
rg -n -i 'set_flag|wait_flag|pipe_barrier|sync_block|sync_related_args|unit_flag|WAIT[._]FLAG|SET[._]FLAG' /absolute/compiler-artifacts
rg -n 'pointer_cast|multi_buffer|arith\.(rem|select)|cbuf|l0c|<ub>|<gm>' /absolute/compiler-artifacts
```

Inspect context around every hit and follow SSA operands into the loop and its
buffer addresses. A text match alone does not establish dynamic execution.
File names such as `ttir`, `ttadapter`, and `npuir` are not a universal guarantee
of pass position. Establish whether memory planning, multi-buffer expansion,
synchronization insertion and macro lowering have happened at that point.

The [official auto-sync guide](https://ascendnpu-ir.gitcode.com/en/developer_guide/features/auto_sync.html)
(accessed 2026-10-22) describes graph solvers and fallback injection passes. Its examples use
`hivm.set_flag`, `hivm.wait_flag` and `hivm.pipe_barrier`; pinned older artifacts
can spell these `hivm.hir.*`. Later LLVM may use `llvm.hivm.WAIT.FLAG.REG` or
related intrinsics with numeric pipe/event operands. Preserve the observed
spelling and its version's definitions. On newer pipelines, delayed cross-core
sync can run after memory planning and core splitting; an earlier dump is
therefore insufficient. This is a version-dependent compiler pipeline, not an
A3 hardware sequencing guarantee.

## Read the operations as dependencies

| Form | What to recover from the actual IR | Meaning |
| --- | --- | --- |
| `set_flag` | Producer/set pipe, consumer/wait pipe, static or dynamic event ID | Event is triggered after earlier work on the producer pipe completes. |
| `wait_flag` | Same pipe pair and matching event ID | Consumer pipe cannot pass the wait before its producer event. |
| `pipe_barrier` | Pipe scope, including whether it is all-pipe | Orders earlier and later work in that scope; not a measurement of stall duration. |
| `sync_block_set`, `sync_block_wait` | Target core type, producer/consumer pipes, flag ID and mode | Cube/Vector or other supported cross-core handoff; match the actual flags and memory. |
| Matrix macro `sync_related_args`, `unit_flag_cond`, `unit_flag_mode` | Macro operands, condition and event configuration | Synchronization can be encoded inside a compound matrix operation without a separate visible `wait_flag`. |

Pinned definitions: AscendNPU-IR
`af5499b3b9f3dbab50b2834bcfff5da5c2a1d920`,
`bishengir/include/bishengir/Dialect/HIVM/IR/HIVMSynchronizationOps.td:1–260`
and `HIVMMacroOps.td:1–300`. Consult the matching version's source when decoding
numeric enums or macro arguments. An ID is reusable over time; matching the
number alone without pipe, core, control path and lifetime can pair the wrong
producer and consumer.

This schematic uses named attributes for readability, not copyable MLIR syntax:

```text
load GM → UB slot0                         # MTE2 produces data
set_flag(MTE2, V, event0)                 # data ready
wait_flag(MTE2, V, event0)                # Vector must see the load
vector_compute(UB slot0)
set_flag(V, MTE2, event1)                 # Vector releases the storage
wait_flag(V, MTE2, event1)                # next load must not overwrite it
load next iteration → UB slot0
```

The first pair protects read-after-write. The release/reuse pair protects
write-after-read: even a correct one-buffer loop can serialize adjacent
iterations. With two real slots, the next load may target slot1 while Vector
uses slot0, with matching rotating event IDs and safe reuse before slot0 is
written again. This creates an overlap opportunity, not evidence that overlap
occurred or that the load was the critical path.

For a matrix epilogue, trace `L0C → FIXPIPE → GM/workspace or L1` and any
Cube→Vector flag before Vector reads/converts/stores the output. Check the
reverse release before the next producer reuses that workspace. Do not label a
cross-core flag as an intra-core wait or attribute all output to MTE3.

## Check whether a requested buffer change took effect

The [official multi-buffer guide](https://ascendnpu-ir.gitcode.com/en/developer_guide/features/multi_buffer.html)
describes marking, address allocation, slot selection and rotating sync IDs.
Look for markers such as `hivm.multi_buffer`, multiple physical offsets in
`pointer_cast`, modulo/selection of an iteration slot, and paired event/flag
rotation. A late artifact can have lowered away the original marker: compare
several stages instead of requiring one string to survive every pass.

Compare these together for `num_stages=1` versus `2`:

1. Actual compiler options, buffer extents and base addresses, including memory
   allocation fallback or resource diagnostics.
2. Loop-carried slot selection and the producer/consumer accesses to each slot.
3. Synchronization around load, compute, result movement, and buffer release.
4. Independent BasicInfo samples, absolute pipe times and all available
   scalar-wait/queue/IB/UB fields for matching rows.

Static wait-site counts are not executed wait counts and do not quantify time.
More flag sites can accompany a faster pipeline. Fewer sites can accompany a
slower resource-constrained kernel. Zero scalar wait with nonzero queue or
handoff stalls is consistent. If the binary cannot be decoded or the relevant
stage is unavailable, say which mechanism remains inferred and test a narrower
controlled variant instead of claiming exact instruction placement.

## Real matched matrix example: flags and buffer slots

The supplied [stages 1 dump](../examples/compiler-evidence/stages1.mlir) and
[stages 2 dump](../examples/compiler-evidence/stages2.mlir) are from the bundled
512×512 fp16/float32 matrix pair, tile 32×32×64. The early TTIR and adapter hashes
were identical across stages 1/2; these late dumps expose the transformation.

Baseline lines 65/90 have one cbuf base for each operand, offsets 0/4096. The
candidate lines 80–83 have two slots per operand, offsets 4096/12288 and0/8192:

```mlir
%50 = hivm.hir.pointer_cast(%c4096_i64, %c12288_i64) : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
annotation.mark %50 {hivm.multi_buffer = 2 : i32} : memref<2x4x16x16xf16, #hivm.address_space<cbuf>>
%51 = hivm.hir.pointer_cast(%c0_i64_0, %c8192_i64) : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
annotation.mark %51 {hivm.multi_buffer = 2 : i32} : memref<4x2x16x16xf16, #hivm.address_space<cbuf>>
```

Each operand slot is 4096 bytes. This input cbuf allocation therefore grows
8192 → 16384 bytes; it is not a statement about all memory used by the kernel.
Candidate line 71 computes modulo 2; lines 75/79 select event pairs 3/4 and 1/2.
The next input must wait until MTE1 has released the chosen slot; once loaded,
MTE2 signals that MTE1 can consume it (candidate lines 107–109, intervening
transfer operands omitted here):

```text
hivm.hir.wait_flag[<PIPE_MTE1>, <PIPE_MTE2>, %49]
hivm.hir.nd2nz ...
hivm.hir.set_flag[<PIPE_MTE2>, <PIPE_MTE1>, <EVENT_ID0>]
```

The matrix macro at candidate line 130 carries `sync_related_args`, so counting
only standalone flags misses its internal dependency operands. For the result,
lines 132/140 order `PIPE_M → PIPE_FIX`; the emitted FIXPIPE writes the CC
accumulator to GM. The `PIPE_FIX → PIPE_M` release protects accumulator reuse.
The final `PIPE_ALL` barrier is a distinct full-pipe ordering point.

This is compiler-confirmed buffering and event machinery bound to the ordinary
binary. It does not by itself prove simultaneous transfer/compute overlap or
quantify saved waiting. Use the separate timing and full wait/stall evidence in
[the paired measurement interpretation](a3-paired-examples.md).
