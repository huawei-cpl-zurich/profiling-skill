# Blind optimization runner

`scripts/blind_optimization_launcher.py` launches exactly one fresh Codex session
in the existing Bubblewrap boundary. It does not schedule jobs, merge branches,
install skills, or implement remote transport. Both arms use the same prompt,
explicit model JSON, public task bundle, and executable tool command. Only
`--skill` differs. Omit it for the genuine no-skill arm.

Export a task bundle first. Its `task.json` declares `public_files`; the launcher
copies only these files into a new independent Git repository. The trusted
runner and CPU reference remain outside the agent's filesystem. Public files
other than `candidate.*` are read-only. The optional skill is a read-only
snapshot; global skills, campaign reports and repository history are not mounted.
The evaluation output must be outside the orchestration workspace.
Invalid manifest, public-file, prompt or skill inputs are checked before an output
directory is reserved, so correcting an ordinary input error permits the same
output path to be retried. Existing outputs are never removed automatically.

```bash
python scripts/blind_optimization_launcher.py \
  --bundle "$BUNDLE" --prompt "$COMMON_PROMPT" --output "$NEW_OUTPUT" \
  --model-json "$MODEL_JSON" --tool-command-json "$TOOL_JSON" \
  --forbidden "$ORCHESTRATION_WORKSPACE" --preflight-only
```

Use a distinct new output directory without `--preflight-only` for the actual
run. Add `--skill "$FROZEN_SKILL"` only for the skill arm. Model JSON must contain
`name` and `reasoning_effort`; the launcher does not select a replacement model.
The common prompt is copied byte-for-byte without arm-specific additions.

The tool JSON is an argv array for an approved domain helper. The launcher
appends `--workspace HOST_CELL` and the agent's `experiment-tool` arguments. The
helper receives a host-configured frozen bundle path separately and must use
the resolved user-wide `remote-access/scripts/cpl-remote` for every real remote
operation. `experiment-tool` is an experiment API, not a renamed or copied
transport client. Both arms must receive identical help and capabilities.
Agent `-h`/`--help` requests map to the public tool help. Configuration flags
`--workspace`, `--bundle`, `--target` and `--runtime` belong to the host, not
the agent interface; accidental overrides return an actionable error before
helper dispatch and may be retried without these flags.

`commands.jsonl` retains every helper argv, full stdout/stderr and exit status.
`agent.jsonl` retains Codex command events and messages; `agent.stderr` retains
client diagnostics. Both files are direct child-process output sinks while the
agent runs, so already-written evidence survives an outer controller interruption
even if the launcher never reaches receipt creation. `receipt.json` records model, runtime, commits, dirty files
and timeout status. CLI exit status is nonzero for agent/client failure, timeout
(124), or a successful model process with no agent commit (3). Preflight-only
runs do not require an agent commit. A timeout retains partial output and stops the local process
group; it does not assert that retained remote jobs stopped. Observe existing
remote handles through the global client. An existing output directory is never
silently relaunched: inspect its receipt and handles before considering a new
run. A zero agent exit without candidate commits is not successful completion.

The v2 common prompt states the 600-second wall-clock limit, including tool
waiting; agents read read-only `execution.json` for the UTC start/deadline,
reserve approximately 120 seconds for notes and commits, log incrementally and
commit validated progress early. The exact v1 prompt remains available as
`PROMPT_V1` for interpreting calibration runs. Do not combine v1 and v2 cohorts
as matched experimental observations. Socket paths use a short temporary
directory independent of the evidence-root path length. Tool execution is
bounded by the remaining agent wall budget; retained remote jobs are still
observed through their original handles after local interruption.

Run the functional isolation and process tests with:

```bash
python -m pytest -q tests/test_blind_optimization_launcher.py
```
