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
the same prompt, model, tool command, isolation boundary, and 600-second budget;
only guidance cells receive `--skill`. The launcher exposes declared public task
files and a neutral tool endpoint; trusted runners and private oracles stay on
the controller side. Repository history and answer implementations are not copied
into agent workspaces.

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
to check pairing, concurrency, isolation inputs, failed-cell retention, and reuse
rejection. It does not claim that a model optimization run or device profile passed.
