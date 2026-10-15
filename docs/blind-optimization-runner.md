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

`commands.jsonl` retains every helper argv, full stdout/stderr and exit status.
`agent.jsonl` retains Codex command events and messages; `agent.stderr` retains
client diagnostics. `receipt.json` records model, runtime, commits, dirty files
and timeout status. A timeout retains partial output and stops the local process
group; it does not assert that retained remote jobs stopped. Observe existing
remote handles through the global client. An existing output directory is never
silently relaunched: inspect its receipt and handles before considering a new
run. A zero agent exit without candidate commits is not successful completion.

Run the functional isolation and process tests with:

```bash
python -m pytest -q tests/test_blind_optimization_launcher.py
```
