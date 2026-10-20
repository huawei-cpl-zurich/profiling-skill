# Profiling discovery: planning-only prompt control

`prompts/profiling-discovery/previous.md` preserves the complete matched GDN
campaign prompt. `revised.md` changes only the profiling-discovery paragraph:
read the named skill, distinguish benchmarking from diagnostic profiling, and
consider evidence that resolves an optimization hypothesis. Actual profiling
remains optional; collecting pipe metrics is not an acceptance checklist.

The runner appends the same planning-only override to both complete prompts.
It asks for a workflow in `PLAN.md`, not implementation, experiments, or commits.
The override replaces the historical budget with the current execution deadline.

## Reproduction

Provide an exported public bundle with `task.json` declaring exactly `API.md`
and `candidate.py`, identical frozen coding/profiling skill directories, and the
same explicit model configuration as the campaign being compared:

```sh
python scripts/profiling_planning_smoke.py \
  --bundle "$PUBLIC_BUNDLE" --model "$MODEL_JSON" \
  --profiling "$PROFILING_SNAPSHOT" --coding "$CODING_SNAPSHOT" \
  --output "$NEW_OUTPUT_DIRECTORY"
```

Default: three independent previous-prompt cells and three revised-prompt cells,
300 seconds each, up to six concurrent cells. Use `--preflight-only` with a
different new output directory to qualify isolation without launching models.
`--version previous|revised|both`, `--replicates 1..3`, and `--parallel 1..6`
permit focused reruns; preserve all earlier outputs and never reuse a directory.

Each cell retains the exact effective/original prompts, skill hashes, public
input hashes, model, isolation results, full command/output transcript, tool
journal, `PLAN.md`, and terminal receipt. Partial transcripts survive timeouts.
The campaign result records each cell separately; one failure does not discard
completed siblings. Secrets from authenticated Codex state are never copied into
the evidence manifest.

## Isolation and assessment

The launcher reuses `BlindLauncher`'s Bubblewrap boundary but mounts only a thin
planning client, not controller code or the production backend. The local tool
stub offers help/capabilities and rejects every operational request with exit77.
The API, candidate skeleton and skills are read-only. No optimized kernels,
oracle, fixtures, remote credentials, NPU devices, Torch or Triton runtime are
exposed. The normal system shell remains available for reading and writing a
plan; this is not a restriction against arbitrary CPU arithmetic. Network is
shared for authenticated model transport, **not disabled**. No remote client or
backend is supplied by the launcher; these boundaries are not a general-purpose
hostile-code sandbox or a guarantee against internet access.

Review transcripts rather than keyword-counting answers. For every cell record:
actual skill-read command/output, benchmark versus profiling distinction, a
conditional evidence-acquisition plan tied to competing hypotheses, and sensible
treatment of unavailable evidence/infrastructure failures. A mention of the
skill alone is not a verified read. Read commands are visible in the full agent
transcript, not inferred from the final answer or the experiment-tool journal.

Accept the revised prompt only when all three revised cells demonstrably read
the profiling skill and correctly explain when/how profiling is useful. A
reasoned choice not to collect metrics is acceptable. Preserve negative results.
This tests discoverability and planning comprehension, not actual acquisition,
kernel quality, optimization speedup, or causal performance benefit.

Functional tests:

```sh
python -m pytest -q tests/test_profiling_planning_smoke.py
```
