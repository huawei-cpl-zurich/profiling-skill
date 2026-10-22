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
for stage, content in compiled.asm.items():
    data = content if isinstance(content, bytes) else str(content).encode()
    (artifact_directory / stage).write_bytes(data)
```

Use a fresh artifact directory per specialization, retain SHA-256 hashes and
bind them to source/options and the measured dispatch. The included helper
sanitizes stage names into filenames and records those hashes. Available keys
and pass positions depend on the installed compiler.

Do not invent an environment variable or assume an installed
build accepts another version's pass name.

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
