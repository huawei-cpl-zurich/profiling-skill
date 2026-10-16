Implement a Triton-Ascend streaming matmul-add candidate for the supplied
reference and cases. This is a one-round correctness experiment using the
simple runner's JSON phase contract.

For `phase=prepare`, do only local, device-free work and return exactly one JSON
object containing the complete `candidate` source, the local `commands` you
ran, a nonempty `hypothesis`, any `retrieved_knowledge`, and an optional
`session_id`. Do not run remote or NPU commands. The host-owned terminal gate
will materialize the candidate and perform the one billed compilation/runtime/
correctness check; only that gate can report success.

For `phase=analyze`, use the supplied terminal result and return exactly one
JSON object with nonempty `reasoning` and `postmortem`. Stop after the requested
phase. The host records the round as a Git commit and does not merge the branch.
