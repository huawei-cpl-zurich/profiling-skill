You are participating in a controlled Triton-Ascend kernel optimization experiment.

Your task is to implement and optimize a Triton kernel matching the supplied
PyTorch reference. Preserve the complete public behavior for every supplied
case. Work for exactly three persistent optimization rounds.

This invocation is Round 1 only. Execute only Round 1 now, then stop and return
control to the host. Do not begin or perform Round 2 or Round 3 in this turn.
The host will resume the same session with a separate instruction for each
later round.

The experiment contract below overrides conflicting paths, commands, output
layouts, validation procedures, or direct-execution advice in installed skills.
Use every installed Triton coding and verification skill, but interact with
hardware only through $EXPERIMENT_CONTROLLER.

FILES AND SUBMISSION CONTRACT

The workspace initially contains these read-only benchmark inputs:

- baseline.py: read-only PyTorch reference containing class Model.
- baseline.json and cases.jsonl: read-only, equivalent case inventories.

It may also contain read-only config.json, AGENTS.md, and .agents/ instructions
or treatment skills. Follow them when present, but do not assume every optional
path exists and do not modify them.

candidate.py and candidate.manifest.json may already contain a starter
submission, as they do for the matmul control. If present, inspect and improve
them; otherwise create them before the first check. In either case, the final
submission consists of exactly these required files at workspace root:

1. candidate.py
   - Must define class Model.
   - Model must preserve the constructor and callable/forward interface of
     baseline.py's Model.
   - It must implement the operation using Triton-Ascend.
   - The complete reference operation for every supplied case must execute as
     one logical Triton entry-point launch. Compiler-generated AIC/AIV
     components of that launch are allowed.
   - Do not use PyTorch/ACL compute, reference or shape fallback, a partial
     output-only kernel, or auxiliary Triton launches inside Model.forward.
   - Keep the best fully validated candidate in this file after every round.

2. candidate.manifest.json
   - Must contain exactly:
     {"schema":"profiling-skill/candidate-kernel/v2","kernel_name":"<exact exported kernel name>","entrypoint":"<Triton entrypoint>","fusion":{"schema_version":1,"mode":"single-logical-launch","complete_operator":true}}
   - kernel_name must be the exact Triton kernel selected for msprof timing.
   - kernel_name must equal entrypoint or its compiler-generated `_mix_aic` or
     `_mix_aiv` component.

You may create temporary notes or source files, but the host evaluates only
candidate.py and candidate.manifest.json. Do not rename the submission to
solution.py, output.py, or a skill-specific directory.

CONTROLLER INTERFACE

The host has already bound the benchmark, cell, physical device, candidate
path, reference, case inventory, and remote backend. Never pass --config,
--cell, --candidate, a workspace filename, or a device ID.

These are the only agent-facing controller commands:

  $EXPERIMENT_CONTROLLER help
  $EXPERIMENT_CONTROLLER budget
  $EXPERIMENT_CONTROLLER check --scope development --round 1
  $EXPERIMENT_CONTROLLER check --scope development --round 2
  $EXPERIMENT_CONTROLLER check --scope full --round 3
  $EXPERIMENT_CONTROLLER profile --repeats 3 --round 1
  $EXPERIMENT_CONTROLLER profile --repeats 3 --round 2
  $EXPERIMENT_CONTROLLER profile --repeats 3 --round 3

help and budget are local and free. Each check or profile command consumes one
of the 18 controller requests, regardless of how many cases or captures it
runs. Do not invoke rank or calibrate; benchmark selection and calibration are
host-owned.

Interpret results exactly as follows:

- status=ok: the requested operation completed successfully.
- status=config_error: your command is invalid. Read the diagnostic or run
  the free help command, correct the command, and do not classify it as
  infrastructure failure.
- status=candidate_error: the candidate failed. Inspect failure_type, which is
  one of submission_error, compile_error, runtime_error, correctness_error, or
  candidate_error, and use the diagnostic to repair the submission.
- status=infrastructure_error: report the infrastructure failure without
  bypassing the controller or running direct hardware commands.
- Exit code 75 or "remote request budget exhausted": make no more controller
  calls. Preserve the best candidate and manifest; the host will still run
  its independent terminal check and profile.

Never use SSH, SCP, Docker, npu-smi, direct Python execution on an NPU, private
validators, or a raw remote client. Do not construct or modify controller
configuration.

ROUND PLAN AND BUDGET

The 18-request budget covers all three rounds. Use no more than five billed
requests in round 1, no more than five in round 2, and retain at least eight
when round 3 begins. Prefer fewer requests when one full-scope operation
answers the question.

Round 1:
- Read the reference and complete case inventory.
- Improve a preseeded candidate.py and candidate.manifest.json when present,
  or create them before the first check when absent.
- Establish a compilable, correct Triton implementation.
- Run a development check.
- Profile only after that check passes.
- Preserve the best validated version.

Round 2:
- Optimize using the prior correctness and msprof evidence.
- Run a development check after material changes.
- Profile only a passing candidate.
- Keep or restore whichever fully validated version is faster.

Round 3:
- Start from the best retained candidate.
- Run the full check early enough to repair failures.
- If repaired, rerun the full check.
- Profile only a frozen full-check-passing candidate if budget remains.
- Leave the best full-check-passing candidate and its exact manifest at the
  workspace root.

Do not claim success from source inspection, host wall time, a failed command,
or an unprofiled variant. At the end, concisely report each round's changes,
controller results, retained candidate, and any unresolved candidate or
infrastructure failure.
