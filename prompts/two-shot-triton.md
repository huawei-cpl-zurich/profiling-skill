This is a two-turn correctness smoke test, not an optimization campaign.

In this first turn, inspect the immutable baseline and cases, write
`candidate.py` and `candidate.manifest.json`, then run exactly:

`$EXPERIMENT_CONTROLLER check --scope development --round 1`

Use the returned compiler, runtime, or correctness diagnostics. Stop after the
check and return control to the host. Do not profile and do not begin Round 2.
