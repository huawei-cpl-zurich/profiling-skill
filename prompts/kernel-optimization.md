Optimize the supplied Triton kernel for the configured benchmark while preserving
its public interface and correctness contract.

Before editing, use every Triton coding and verification skill available in this
workspace. Those skills describe the required development process; the managed
`$EXPERIMENT_CONTROLLER` command is the only permitted way to run checks or
profiles and supersedes any direct execution command mentioned by a skill.

Work for exactly three persistent optimization rounds. Use development checks and
`msprof op` profiles while iterating. Reserve enough request budget in round 3 to
run a full check, repair any failure, and rerun that full check before finishing.
Compilation, runtime, and correctness failures count as experiment failures.
Infrastructure failures should be reported without changing the benchmark or
working around the managed controller.

Do not inspect paths outside this workspace, alter the benchmark inputs, change
the controller, or use a profiling or coding skill that is not installed here.
