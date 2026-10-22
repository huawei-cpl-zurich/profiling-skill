# Reproducing the six paired optimization pilot

Use `scripts/run_blind_optimization_pilot.py` from a checkout containing the
blind task exporter, launcher, and domain tools. The driver freezes those
first-party sources, the shared launcher prompt, explicit model configuration,
six task bundles, and the twelve-file guidance allowlist in a new output root.
The manifest records the source revision and content hashes; it is provenance,
not an authentication mechanism. No remote-access transport client is copied.
The copied domain helper resolves the installed, managed `cpl-remote` client.

Create a local model JSON containing both `name` and `reasoning_effort`, then run:

```bash
python3 scripts/run_blind_optimization_pilot.py \
  --output /outside-workspaces/new-pilot \
  --model-json /local/model.json \
  --forbidden-workspace /absolute/orchestration-workspace \
  --forbidden-workspace /absolute/cpl-skills-workspace
```

Choose a new output directory outside the source and orchestration workspaces.
Pass every workspace that must be hidden explicitly: the copied launcher cannot
infer the orchestration workspace from its own location. `--source-root` selects
an alternate source checkout, without changing the explicit model. Both arms use
the same prompt, model, tool command, isolation boundary, and selected budget;
only guidance cells receive `--skill`. The launcher exposes declared public task
files and a neutral tool endpoint; trusted runners and private oracles stay on
the controller side. Repository history and answer implementations are not copied
into agent workspaces.

The defaults remain `--task-suite original --budget-seconds 600`. Select
`--task-suite natural --budget-seconds 1200` to prepare the natural-use suite with
a 1200-second wall-clock budget per agent. Budgets must be positive integers.
Both suites retain task01 through task06 and the same balanced waves. The
original suite snapshots `blind_before_tasks.py` and `blind_task_support/runner.py`;
the natural suite snapshots `natural_before_tasks.py` and
`natural_task_support/runner.py`. Both include the trusted native host support.
Required suite dependencies and budget are checked before creating the output
root, so restoring a missing dependency permits preparation at the same path.

Preparation calls the frozen launcher's `build_prompt(budget)` and records both
the suite and wall-clock budget in the manifest. Every cell receives that same
budget through the launcher CLI. Suite and budget selection do not prescribe
which profiling operations agents must use. Changing them creates a new cohort;
the historical pilot report and its original inputs remain separate evidence.

Each suite also freezes its complete controller-only protocol, including the
blind scoring rubric: `blind-profiling-optimization-protocol.md` for original,
`natural-profiling-evaluation-protocol.md` for natural. The manifest names and
hashes that snapshot. Protocol/rubric snapshots are not declared public task
files or mounted guidance and are never exposed through the agent tool endpoint.
Before launch the controller checks the frozen file set and hashes, then
validates and executes the manifest's twelve-cell schedule. Added guidance files
or malformed schedules fail before reserving a launch. Natural preparation alone
does not establish qualification: complete its protocol's private two-batch gate
for all six tasks before starting agents.

The waves are `01 guidance / 02 control / 03 guidance`, then the complementary
arms, then `04 control / 05 guidance / 06 control`, then their complementary arms.
Each wave launches at most three fresh agents and finishes before the next wave.
Every cell is attempted once. A failed cell retains its controller log, launcher
receipt when available, and result JSON; subsequent scheduled cells still run.
The driver exits nonzero if any controller fails. It never retries or replaces a
cell, and refuses an existing output directory or a second cohort launch.
Controller success means the agent completed with a commit; it does not by
itself certify kernel correctness, valid timing, or a benefit from the skill.
Assess the retained execution evidence and the blind rubric separately.

`--prepare-only` freezes inputs without launching agents. Inspect the resulting
manifest and, when ready, run the frozen cohort once from Python:

```python
from pathlib import Path
from scripts.run_blind_optimization_pilot import run_cohort
run_cohort(Path('/outside-workspaces/new-pilot'))
```

The model-independent functional gate is:

```bash
python3 -m unittest discover -s tests -p test_run_blind_optimization_pilot.py -v
```

It substitutes an exporter and subprocess launcher at the real controller boundary
to check both suite exporters, budgets, pairing, concurrency, isolation inputs,
failed-cell retention, and reuse rejection. It does not claim that a model
optimization run or device profile passed.
