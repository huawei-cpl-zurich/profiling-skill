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
execution, synchronization, Scalar, MTE1, and MTE3 remain unknown.

## Resolve an unknown mapping

Run exactly one case from `benchmarks/triton_pipe_probe.py`. Capture it with
the installed `mlir-triton-dump` workflow so the manifest retains TTIR,
adapter, and compiler-consumer inputs. Then collect a compact product-correct
profiler capture through `remote-access`. A pipe timeline can correlate
activity with emitted instructions; activity alone neither proves source
attribution nor capacity saturation.

The probe's `--contract` output uses `constructs[]`; every identifier is an
independent lookup key for this inventory. Do not join several constructs into
an unreviewed composite attribution.

Do not transfer an A3 result to A2 or A5. The current inventory has live A3
scope; A2 and A5 queries deliberately return `unknown` until pinned compiler
builds and product-correct captures establish those mappings.
