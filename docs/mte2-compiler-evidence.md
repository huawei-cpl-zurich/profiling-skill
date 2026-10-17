# Compiler evidence for MTE2 characterization

Use the user-wide `mlir-triton-dump` skill, its managed TLA helper, and the named
remote runtime. The CLI reuses `a3_mlir_retrospective.isolated_capture`: Python
runtime files and compiler consumers are copied into the fresh job directory.
Shared compiler paths are never replaced. Resolve the helper through the managed
skill checkout, or `TLA_INTEGRATION_TESTS_ROOT`; do not vendor another helper.

After activating the runtime, discover actual consumers with `command -v
bishengir-compile bishengir-compile-a5 hivmc hivmc-a5 bisheng`. The adapter also
discovers the backend's bundled consumer and passes explicit real-tool paths.

Prepare an identity JSON containing `target`, `specialization`, and `options`
for the exact workload, not guessed historic settings. The identity is labelled
caller-declared: the real manifest argv and workload runtime metadata remain
the independent evidence. Then run through the approved remote transport:

```bash
python scripts/mte2_compiler_evidence.py capture \
  --helper "$TLA_INTEGRATION_TESTS_ROOT/scripts/capture-triton-compiler-artifacts.sh" \
  --identity identity.json --source benchmarks/mte2_compiler_controls.py \
  --output job/capture-control --assembly-root job/assembly-control -- \
  python benchmarks/mte2_compiler_controls.py \
    --family mask --variant before --output job/assembly-control
```

Use a distinct fresh directory for each mask before/after and buffering
before/after. The former compares predicated loads against load-then-select.
The latter requests `num_stages=1/2` for identical matmul source and work: only
actual later artifacts can establish that a compiler changed buffering or flags.
No changed lowering is a useful negative result. Controls run correctness once,
retain returned compiled `.asm` stages and compiler metadata, and do not time.

The same generic wrapper accepts the memory-characterization workload after
`--`: direct contiguous/strided, runtime indices selecting the same addresses,
and local/global permutations. Supply its exact specialization/options/target
identity and assembly output directory; repeat `--source` for source imports
whose byte identity matters. No independent load implementation is substituted.

Read `capture/capture-manifest.json` first. `evidence.json` retains its hash,
actual invocation argv, compiler input/output hashes, complete-log locations,
up to 8 KiB per error stream, and line-numbered operation excerpts. Full logs,
IR and object files remain in the retained remote directory. Compilation or
reference failures and exit-zero runs with no real compiler input return exit
2, with diagnostics retained. `inspect --manifest ... --assembly-root ...
--output summary.json` reindexes an existing capture without recompilation.

Check source, specialization, target, options and (when available) final binary
hashes against the ordinary timed run. Forced compilation/debug capture is not
timing evidence. A `memref.copy` is not a particular pipe; early MLIR is not final
hardware scheduling. Missing later stages are explicitly reported, not inferred.
Use actual scalar loads/loops versus transfers to formulate hypotheses, then
test them against separate timings and correctly scoped pipe/memory counters.
