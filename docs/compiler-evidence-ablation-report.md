# Compiler and profiling evidence diagnostic

Status: public inputs qualified and frozen; twelve assessment contexts launched.
Agent outcomes and their assessment are pending. No benefit or pass claim is
made by this scaffold. The private case key was frozen before the first context.

## Question and design

Does actual compiler lowering help an agent connect Triton source to a measured
mechanism, beyond source/timing or profiling evidence alone? This is not a
skill-versus-no-skill comparison and does not test autonomous evidence acquisition.

Three tasks each have four fresh, isolated assessment contexts: timing/source,
plus baseline profiling, plus compiler artifacts, or both. Historical arm labels
are `timing`, `pipes`, `compiler`, `both`; `pipes` includes baseline Memory as
well as PipeUtilization, not just pipe counters. All agents receive the same
frozen profiling skill, neutral prompt, model/reasoning settings, 300-second
wall budget and30-second finish reserve. Maximum concurrency is three.

The tasks compare runtime indexed loads with block-size/warp requests, periodic
masking with source predication/block size, and a small fp16→fp32 matrix product
with buffering/K-tile changes. Candidate timings and profiles are held out.
Compiler arms see unannotated actual artifacts for all three implementations,
not merely early source IR or an expert's selected excerpts.

All nine source workloads passed real device correctness and actual compiler
capture. Qualification used five independent BasicInfo samples and three each
PipeUtilization/Memory captures per implementation, rotating order between
implementations. Compilation/debug acquisition was separate from timing.
No new NPU work is dispatched by the offline agent runner.

## Reproducible public inputs

[Public input archive](data/compiler-evidence-ablation-inputs.tar.gz) contains
123 declared inputs plus its manifest: seven common public files per task,
thirteen baseline profiling files per task, and twenty-one compiler files per
task. Source, specialization, compiler options, target and actual artifact
identities are retained. Every diagnostic metric report is bound to exactly
one raw BasicInfo row with the same baseline kernel selector.

Only explicitly configured path prefixes were normalized. Compiler manifests
retain distinct original and served hashes; no operation text was rewritten
or answer-relevant passages highlighted. Duplicate identical raw compiler
artifacts were removed, not distinct stages. No candidate performance result,
compile duration, expert explanation, profiler binary or full capture tree is
included. Missing/unvalidated counter units are not silently converted.

- Archive SHA256: `a927866ab641d6ddd1713bfbc1f2a27d63c9d0cc49b2e672d3f5bc54a97e68d5`.
- Frozen skill: repository revision `9f4a503`, `SKILL.md` and complete
  `references/` tree,76 files total. Do not substitute an installed skill.
- [Model configuration](data/compiler-evidence-ablation-model.json): explicit
  `gpt-5.6-sol`, low reasoning effort.
- Observed target: `Ascend910_9382`; Triton module3.2.0, distribution3.2.2.
  Exact CANN/profiler version unavailable; retained as unknown rather than
  inferred from an image label.

Extract the archive into a fresh directory, reconstruct the skill snapshot
from the named Git revision, and use the [protocol](compiler-evidence-ablation.md):

```bash
python scripts/run_compiler_evidence_ablation.py \
  --packets <extracted-public-inputs> --skill <frozen-skill-snapshot> \
  --model-json docs/data/compiler-evidence-ablation-model.json \
  --output <fresh-output-outside-workspace> \
  --forbidden <orchestration-workspace>
```

An actual-packet fake-launcher smoke completed twelve cells without invoking a
model. The coordinator additionally checked a real sandbox preflight. Public
inputs contain no original machine paths, compile-duration fields or expert
annotations. The private key and held-out dataset were separately frozen.

## Assessment and results

Pending completion and anonymous assessment. Report each answer's mechanism
attribution, discriminating decision/measurement, and both predictions as
supported, unsupported or indeterminate. Report critical unsupported claims
separately. Directional agreement is distinct from evidence-supported reasoning;
reading/citing an artifact is descriptive acquisition data, not success itself.
No numeric score or post-hoc acceptance threshold will be introduced.

Preserve nulls and negative results. The three selected cases and one context
per condition cannot establish universal usefulness. Static IR does not prove
hardware scheduling or concurrent overlap; baseline-only profiling does not
establish candidate counter deltas. Noisy overlapping timing ranges permit
indeterminate predictions rather than requiring the sign of a noisy median.
