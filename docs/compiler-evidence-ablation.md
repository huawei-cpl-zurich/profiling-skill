# Compiler evidence ablation

This is a supplied-evidence diagnostic, not a performance campaign or a new
skill-versus-no-skill experiment. It tests whether baseline compiler artifacts
help agents connect Triton source to a measured mechanism, beyond pipe metrics
or timing alone. Compiler arms receive actual artifacts for the baseline and
both proposed candidates; pipe metrics and timing describe only the baseline.
This tests the incremental value of supplied compiler information, not whether
agents successfully acquire profiling evidence. Measurement and capture
qualification must finish first.

## Fixed experimental design

Use exactly three independently qualified cases, with two proposed source
changes per case. Each case appears in four fresh, isolated contexts:

| Arm | Evidence supplied |
| --- | --- |
| timing | Source and timing |
| pipes | Source, timing and pipe captures |
| compiler | Source, baseline timing and all three implementations' compiler artifacts |
| both | Source, baseline timing/pipe captures and all three implementations' compiler artifacts |

The historical `pipes` arm name is retained by the interface, but its supplied
profiling evidence includes three baseline PipeUtilization captures and three
baseline Memory captures where supported. Preserve the raw CSVs and explicitly
record unavailable Memory captures. Raw counter units and denominators remain
unvalidated unless independently established; do not silently convert them to
physical bandwidth. Describe this contrast as profiling evidence versus timing,
not a pure pipe-counter ablation. No candidate profile is supplied.

All twelve agents receive the same frozen profiling skill, prompt, explicit
model/reasoning settings and 300-second budget, including tools and waiting.
At most three contexts run concurrently. Every agent has a fresh repository;
no prior agent answers, repository history or client context is imported.
The prompt does not require pipe use or name an expected mechanism. The agent
chooses a next candidate, predicts both candidates, cites evidence, explains
uncertainty, and proposes a discriminating next measurement.

The exact prompt is `PROMPT` in the runner and is retained in each campaign's
manifest and frozen snapshot. Live benchmark, remote and capture commands fail
explicitly in this offline study. Model command events and the neutral tool's
command journal remain in each agent directory. Retain all of them, not only
the final answers. Inputs are read-only sandbox mounts; answers must match
committed bytes. Agents must not merge their branches.

## Input contract

`manifest.json` has `schema_version: 1`, `tasks: ["task01", "task02", "task03"]`,
`files` mapping packet-relative filenames to SHA-256, and `evidence` mapping
each task to nonempty `pipes` and `compiler` filename lists. Example:

```json
{"schema_version": 1, "tasks": ["task01", "task02", "task03"],
 "files": {"public/task01/task.json": "<sha256>"},
 "evidence": {"task01": {"pipes": ["capture.csv"],
                         "compiler": ["capture-manifest.json", "ir/kernel.mlir"]}}}
```

The abbreviated example must be completed for all tasks and declared files.
Each `public/taskNN` has exactly the declared public inputs: `task.json`,
`TASK.md`, `baseline.py`, `candidate-a.py`, `candidate-b.py`, `timing.json`,
`environment.json`. Its `task.json` declares those names in `public_files`.
The environment requires nonempty `soc`, `compiler`, `profiler` strings with
observed values; record `unknown` explicitly if a version cannot be obtained.
All three sources implement the same qualified kernel interface.

Evidence files live in `pipes/taskNN` and `compiler/taskNN`. Lists are relative
to those directories; nested filenames are supported. Agents see only the
selected evidence, under `pipes/` and `compiler/` in their workspace. Undeclared
files are not copied into the frozen campaign or agent bundles. Symlinks and
parent traversal are rejected. The existing snapshot/hash mechanism is reused.

The coordinator must inspect *contents*, not just filenames, before freezing:
no held-out candidate measurements, qualification outcomes, expert diagnoses,
answer keys, or private provenance comments in public sources, task metadata,
CSV names, compiler manifests, logs, or paths. Supply actual baseline and both
candidates' compiler artifacts and commands from `$mlir-triton-dump`, bound to
source, specialization, compiler options and target, with neutral implementation
labels. Candidate compilation artifacts are evidence; candidate timings, pipe
profiles, measured speedups and expert annotations are not public inputs. Do
not highlight answer-relevant IR passages. Explain unavailable stages neutrally;
early MLIR is not proof of final instruction order. Never copy a raw capture
directory without selecting and checking its contents. Candidate performance
outcomes and the expert key stay outside the packet input tree in a separately
retained assessment area.

## Running and interruptions

```bash
python scripts/run_compiler_evidence_ablation.py \
  --packets /tmp/qualified-packets --skill /tmp/frozen-profiling-skill \
  --model-json /tmp/model.json --output /tmp/compiler-evidence-study \
  --forbidden /absolute/orchestration/workspace
```

Model JSON requires `name` and `reasoning_effort`. Keep output outside both
inputs and the orchestration workspace. Do not launch until qualification,
source/input review and a fake-launcher smoke succeed. The skill snapshot is
provided explicitly; the runner never loads global skills.

`--resume` checks identical inputs, model, prompt, skill, runtime source and
retained sealed answers. It dispatches only untouched cells. Partial cells,
timeouts and invalid answers remain visible failures and are not silently
replaced. Inspect retained logs before deciding whether a separately recorded
replacement is justified by infrastructure; never label a failed diagnosis as
infrastructure. Exiting successfully requires all twelve valid committed answers.

## Assessment before unblinding

Freeze a private per-case expert explanation and both candidate outcomes before
agents run. It must state what each available evidence stage can and cannot
establish, uncertainties in the measurements, and acceptable alternative
diagnoses. Do not build a gold answer by reading the agents' responses.

An assessor receives anonymously labelled answers and the case key, without arm
labels. Record mechanism attribution, a useful discriminating decision or next
measurement, and both latency predictions as supported, unsupported or
indeterminate under the retained evidence. Report critical unsupported claims
separately (for example treating pipe activity as achieved DRAM bandwidth or
early IR as final scheduling). Reading/citing files is an acquisition statistic,
not evidence of a correct conclusion and not a pass requirement.

Report all twelve outcomes and pairwise contrasts by case, plus compiler/pipe
file use, command logs, invalid contexts, ceiling effects and uncertainty. Do
not invent an acceptance score after seeing results. With three selected cases,
findings justify narrow guidance and further hypotheses, not universal causal
claims or automatic skill acceptance. Preserve null and negative findings.
