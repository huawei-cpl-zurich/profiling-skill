# Profiling-skill behavioral gate

`scripts/profile_behavioral_gate.py` runs the dependency-neutral part of the
profiling-skill evaluation. It treats skill trees, prompts, frozen evidence,
and the agent launcher as pinned black-box inputs. It does not import product
normalizers or launch a remote by itself, so a manifest can be prepared before
the A3/A5 feature branches merge.

Do not run the live gate until the product branches and a supported A5 runtime
are available. Functional tests use a fake executor and do not launch agents.
This core is not live-ready on its own: a follow-up launcher adapter must reuse
the repository's audited production launcher to provide Bubblewrap isolation,
Codex authentication, and the request/output contract below. Passing an
arbitrary unisolated Codex command is not an accepted live configuration.

## Battery

The runner executes these units and writes an atomic `state.json` after every
attempt:

- Acquisition uses only the candidate skill. One agent identity must succeed
  on A3 and A5 to complete a pair. The runner continues through candidate
  identities until three pairs complete.
- Interpretation uses four pinned cases: two A3 and two A5. Three agents use
  the current skill and three use the candidate skill. Each case therefore has
  six interpretations and the battery has 24 interpretations in total.
- The prompt and case evidence bytes are identical across current/candidate
  arms. The request exposed to the agent contains no arm name.

Only a documented failure with `failure_class: infrastructure`, a recognized
infrastructure type, and nonempty logs is discarded and retried. Compile,
runtime, profiler-command, evidence, and interpretation failures count. A
launcher timeout or missing launcher is counted because it does not prove
remote infrastructure flakiness. Resume skips terminal units and retries only
discarded infrastructure attempts or units not yet run.

Acceptance requires three paired acquisition successes, all 12 candidate
interpretations passing, no unsupported saturation claim, and a candidate
pass count strictly greater than the current-skill pass count.

## Isolation and runtime handoff

Every attempt gets a new workspace with a temporary `HOME`, `CODEX_HOME`, and
`AGENTS_HOME`. Exactly one skill tree is copied to
`.agents/skills/ascend-profiling`; user-wide skill discovery through these
homes is unavailable. The external executor is launched in that workspace
with a cleared environment.

Remote routing data must not be restored by exposing the user's real home.
Instead, pin each required nonsensitive runtime file in `runtime_inputs`:

```json
{
  "name": "remote-registry.json",
  "path": "/ignored/input/remotes.json",
  "sha256": "...",
  "environment": "PROFILE_GATE_REMOTE_REGISTRY"
}
```

The runner verifies the source hash, copies the file into the attempt, exposes
only its copied path through the named environment variable, and removes the
copy after the launcher exits. A launcher that needs an SSH agent or another
credential transport must provide that outside the agent-facing request; do
not put secrets in the manifest, answer, logs, or evidence. The launcher is
responsible for translating the scoped registry variable to its approved
`cpl-remote` invocation.

## Executor contract

Pass the launcher as a JSON argv array:

```console
python scripts/profile_behavioral_gate.py \
  --manifest /ignored/input/gate.json \
  --run-root /retained/run-id \
  --executor-json '["/approved/launch-agent"]'
```

The launcher reads `PROFILE_GATE_REQUEST`, writes one JSON object to
`PROFILE_GATE_OUTPUT`, and runs with the request workspace as its current
directory. The required audited launcher adapter is intentionally a separate
PR; this core's fake-executor tests do not substitute for that live boundary.
Interpretation success contains:

```json
{
  "status": "success",
  "case_id": "case-id",
  "evidence_sha256": "...",
  "conclusions": ["rubric-conclusion-id"],
  "saturation_claims": [{"resource": "MTE2"}],
  "reasoning": "compact evidence-based reasoning",
  "logs": ["commands or observations"]
}
```

Acquisition success contains `product`, `handle`, `commands`, `reasoning`,
`logs`, `evidence_file`, and `evidence_sha256`. `evidence_file` must be a
regular workspace-local nonsymlink whose bytes match the hash. The runner
copies it under the run's retained `evidence/` tree. Escapes, symlinks, absent
files, and hash mismatches count as evidence failures.

The final `report.json` retains prompt/skill/evidence identities, all attempts,
structured answers, scores, compact evidence paths, and aggregate acceptance.
