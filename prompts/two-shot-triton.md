This is a two-round Triton-Ascend correctness smoke test, not a full
optimization campaign. Use the simple runner's JSON phase contract and act only
on the `phase` and `round` in the current input.

For each `phase=prepare`, inspect the immutable baseline and cases plus prior
committed evidence. Return exactly one JSON object containing the complete
`candidate` source, local device-free `commands`, a nonempty `hypothesis`, any
`retrieved_knowledge`, and an optional persistent `session_id`. Do not run
remote or NPU commands. The host-owned terminal gate performs the billed
compilation/runtime/correctness check after your response. If the runner
supplies candidate diagnostics, repair that failure without treating it as a
new round.

For each `phase=analyze`, use the supplied terminal result and return exactly
one JSON object with nonempty `reasoning` and `postmortem`. Stop after the
requested phase and round. The host alone retries infrastructure failures,
advances rounds, commits evidence, and leaves the experiment branch unmerged.
