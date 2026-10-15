#!/usr/bin/env python3
"""One leased A3 timeline acquisition; no device selection or remote transport."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

from summarize_a3_simulator import summarize


def _execute(argv, *, capture_output, text, timeout):
    # A timed-out profiler may own an application/simulator process tree. Keep
    # the caller's device lease until that entire process group has terminated.
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=text, start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(argv, process.returncode, stdout, stderr)


def _run(argv, output, label, timeout, runner):
    try:
        process = runner(argv, capture_output=True, text=True, timeout=timeout)
        code, stdout, stderr = process.returncode, process.stdout, process.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout, stderr = exc.stdout or "", exc.stderr or ""
        stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
        stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        stderr += "\nacquisition timeout (child terminated by subprocess timeout)"
    logs = {}
    for stream, text in (("stdout", stdout), ("stderr", stderr)):
        raw = text.encode()
        path = output / f"{label}.{stream}.log"
        path.write_bytes(raw)
        logs[stream] = {"path": path.name, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()}
    return {"argv": argv, "exit_code": code, "logs": logs}, stdout + stderr


def acquire(application: Path, output: Path, selector: str, *, mode="hardware",
            soc_version=None, launch_count=1, timeout=180, runner=_execute):
    """Caller holds an idle physical-device lease for this complete operation.

    Reports are intentionally not overwritten or resumed here. The outer queue
    owns retained handles, retry policy, and a new output directory per attempt.
    Simulator evidence never satisfies hardware availability or source phases.
    """
    if mode not in {"hardware", "simulation"}:
        raise ValueError("mode must be hardware or simulation")
    if mode == "simulation" and not soc_version:
        raise ValueError("simulation requires observed --soc-version")
    if not selector.strip() or launch_count < 1 or timeout < 1:
        raise ValueError("selector, launch count and timeout must be valid")
    if not application.is_file() or not os.access(application, os.X_OK):
        raise ValueError("application must be an existing executable wrapper")
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
        raise ValueError("caller must select the leased physical device")
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "mode": mode, "selector": selector,
              "soc_version": soc_version, "physical_device_mask": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
              "source_phase_interpretation": "unavailable: requires independent source mapping",
              "hardware_timeline_available": False, "clean_acquisition_gate_passed": False}
    check, _ = _run([str(application.resolve())], output, "workload", timeout, runner)
    result["correctness"] = check
    if check["exit_code"]:
        result["status"] = "workload_failed"
    else:
        command = ["msprof", "op"]
        if mode == "simulation":
            command += ["simulator", "--soc-version=" + soc_version]
        command += ["--application=" + str(application.resolve()),
                    "--output=" + str((output / "report").resolve()),
                    "--kernel-name=" + selector,
                    "--launch-count=" + str(launch_count)]
        command += (["--aic-metrics=PipeUtilization", "--dump=on", "--timeout=1"] if mode == "simulation"
                    else ["--aic-metrics=TimelineDetail", "--warm-up=0", "--kill=off"])
        profile, transcript = _run(command, output, "profiler", timeout, runner)
        result["profiler"] = profile
        result["inventory"] = [
            {"path": str(path.relative_to(output)), "bytes": path.stat().st_size}
            for path in sorted((output / "report").rglob("*")) if path.is_file()]
        terminal_marker = "Profiling running finished. All task success." in transcript
        diagnostic_failures = [marker for marker in (
            "TimelineDetail data collection failed", "Child process killed by signal",
            "Running task failed", "Segmentation fault") if marker in transcript]
        failed = profile["exit_code"] != 0 or not terminal_marker or bool(diagnostic_failures)
        result["profiler"]["terminal_success_marker"] = terminal_marker
        result["profiler"]["failure_markers"] = diagnostic_failures
        if mode == "simulation":
            normalized = summarize(output / "report", selector,
                                   acquisition_status="partial" if failed else "clean",
                                   acquisition_failure="profiler_failed" if failed else None)
            result["simulation"] = normalized
            result["clean_acquisition_gate_passed"] = normalized["clean_acquisition_gate_passed"]
            result["status"] = ("profiler_failed" if failed else "available" if
                                normalized["clean_acquisition_gate_passed"] else "evidence_unavailable")
        else:
            # Hardware normalization stays in the acquisition harness, which
            # validates selector coverage and never consumes simulator paths.
            result["status"] = "profiler_failed" if failed else "requires_hardware_normalization"
    (output / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kernel-name", required=True)
    parser.add_argument("--mode", choices=("hardware", "simulation"), default="hardware")
    parser.add_argument("--soc-version")
    parser.add_argument("--launch-count", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    try:
        result = acquire(args.application, args.output, args.kernel_name, mode=args.mode,
                         soc_version=args.soc_version, launch_count=args.launch_count, timeout=args.timeout)
    except (ValueError, OSError) as exc:
        parser.exit(2, str(exc) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0 if result["clean_acquisition_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
