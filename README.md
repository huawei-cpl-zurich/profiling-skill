# Ascend profiling skill and development repository

Install or copy **only [`skills/ascend-profiling`](skills/ascend-profiling)** as
`ascend-profiling` in your agent's skill directory. Its `SKILL.md` lists runtime
prerequisites. Its scripts and references work without this repository's test,
experiment, or infrastructure tooling.

- `skills/ascend-profiling/`: standalone profiling instructions, references, and
  standard-library Python helpers.
- `tests/`: functional tests for the skill and repository tools; run with
  `python3 -m pytest tests` after installing pytest in your development environment.
- `scripts/`, `benchmarks/`, `experiments/`, `prompts/`, and `docs/`: development,
  infrastructure, benchmark, and experiment tooling. Do not install these as
  profiling skill resources or expose them through the skill folder.
- `skills/ascend-model-benchmarking/`: a separate skill with its own entrypoint.
- `references/`: experiment research inputs outside the installed profiling skill.

See [the context audit](docs/standalone-skill-context-audit.md) for removed
assumptions and interpretation limits. Moving helpers changes repository paths;
helper arguments and evidence schemas remain unchanged. Compiler attribution
now emits revision-qualified source coordinates instead of private broker citation
IDs; custom inventories using the previous citation form remain supported.
