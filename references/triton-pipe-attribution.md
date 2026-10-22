# Triton-to-pipe attribution

The reviewed inventory is
`references/triton-pipe-attribution.json`. Query it rather than guessing:

```bash
python3 scripts/query_triton_pipe_attribution.py \
  --construct gm_to_l1_copy --product a3
python3 scripts/query_triton_pipe_attribution.py --construct triton_dot --product a3
```

The inventory is pinned to Triton-Ascend 3.2.0 commit
`23ac2717c0a38ba962cbd4a0425fc069e7ae104d` and its GitCode AscendNPU-IR
submodule commit `af5499b3b9f3dbab50b2834bcfff5da5c2a1d920`. Its `ref://` selectors point
to curated immutable GitCode snapshots, not GitHub mirrors.

## Read the status literally

- `direct` means the cited compiler source assigns the lowered operation to
  the named `PIPE_*` enum.
- `inferred` means a relationship is supported but not directly defined by
  the cited source. In particular, translating `PIPE_V` to the profiler label
  `Vector` is currently an enum-name inference.
- `unknown` means no reviewed evidence establishes the relationship. A likely
  architectural answer is not a result.

The compiler pass order is direct source evidence. It does not prove that one
original Triton operation executes on one pipe. Fusion, layout conversion,
address spaces, macro operations, and unassigned HIVM operations can change or
split execution. Inspect the emitted operations before correlating a profiler
track.

Five lowered cases are directly established: UB-to-UB `CopyOp` uses `PIPE_V`,
L0C-to-GM `CopyOp` uses `PIPE_FIX`, GM-to-L1 `CopyOp` uses `PIPE_MTE2`, and
`VBrc` uses `PIPE_MTE2` for L1 destinations or `PIPE_V` for UB destinations.
General loads/stores, dot, elementwise operations, reductions, layout
execution, synchronization, Scalar, MTE1, and MTE3 remain unknown in this
pinned source inventory. That status does not forbid a more narrowly scoped
observation from actual artifacts: identify its compiler, target and stage,
and do not silently relabel the inventory's unreviewed general case.

In particular, a tensor load can become a scalar indirect-load loop, a masked
load can retain a contiguous copy followed by vector predicates, and equal
early IR can acquire different buffers and flags later. `tl.dot` requires
evidence of the actual matrix path. See [A3 performance reasoning](a3-performance-reasoning.md)
for how these distinctions change profiler interpretation.

## Resolve an unknown mapping

When exact attribution matters, optionally capture the candidate or a focused
single-operation probe with the installed `mlir-triton-dump` workflow so its
manifest retains TTIR, adapter and compiler-consumer inputs. Inspect later
scheduled IR when buffering/flags are the question. Collect product-correct
profiler evidence through `remote-access` or the supplied benchmark interface.
A qualified instruction trace can relate activity to emitted instructions;
activity alone proves neither source attribution nor capacity saturation.
Keep A3 simulation separate from A5 hardware timelines. This escalation is not
a mandatory compiler-dump gate for every optimization.

The probe's `--contract` output uses `constructs[]`; every identifier is an
independent lookup key for this inventory. Do not join several constructs into
an unreviewed composite attribution.

Do not transfer an A3 result to A2 or A5. The current inventory has live A3
scope; A2 and A5 queries deliberately return `unknown` until pinned compiler
builds and product-correct captures establish those mappings.
