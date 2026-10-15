import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("timeline_probe", SCRIPTS / "a3_timeline_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.fixture
def application(tmp_path, monkeypatch):
    path = tmp_path / "application.sh"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o700)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "11")
    return path


def capture_runner(*, code=0, marker="probe", correctness=0, timeout=False, extra="", terminal=True):
    commands = []

    def run(argv, **kwargs):
        commands.append(argv)
        if len(commands) == 1:
            return subprocess.CompletedProcess(argv, correctness, "correctness checked", "")
        if timeout:
            raise subprocess.TimeoutExpired(argv, 10, output=b"partial output", stderr=b"partial error")
        output = Path(next(x.split("=", 1)[1] for x in argv if x.startswith("--output=")))
        trace = output / "OPPROF_0" / "simulator" / "trace.json"
        trace.parent.mkdir(parents=True)
        trace.write_text(json.dumps({"traceEvents": [{"ph": "X", "pid": "core0.veccore0",
            "tid": "VECTOR", "name": "vadd", "ts": 0, "dur": 2}]}))
        text = f"<ProfInit> Start profiling on kernel: {marker}\n" + extra
        if terminal:
            text += "Profiling running finished. All task success.\n"
        return subprocess.CompletedProcess(argv, code, text, "")
    return run, commands


def test_clean_simulation_is_not_hardware_or_source_phase(application, tmp_path):
    runner, commands = capture_runner()
    result = probe.acquire(application, tmp_path / "capture", "probe", mode="simulation",
                           soc_version="ObservedSoC", runner=runner)
    assert result["status"] == "available"
    assert result["clean_acquisition_gate_passed"]
    assert not result["hardware_timeline_available"]
    assert result["simulation"]["event_count"] == 1
    assert "unavailable" in result["source_phase_interpretation"]
    assert "--soc-version=ObservedSoC" in commands[1]
    assert "--warm-up=0" not in commands[1]
    assert result["physical_device_mask"] == "11"
    assert json.loads((tmp_path / "capture/result.json").read_text()) == result


@pytest.mark.parametrize("exit_code", [-11, 1, 124])
def test_trace_after_process_failure_remains_partial(application, tmp_path, exit_code):
    runner, _ = capture_runner(code=exit_code)
    result = probe.acquire(application, tmp_path / "capture", "probe", mode="simulation",
                           soc_version="ObservedSoC", runner=runner)
    assert result["status"] == "profiler_failed"
    assert result["simulation"]["partial_capture"]
    assert result["simulation"]["interpretable"]
    assert not result["clean_acquisition_gate_passed"]


def test_wrong_selector_never_passes(application, tmp_path):
    runner, _ = capture_runner(marker="other")
    result = probe.acquire(application, tmp_path / "capture", "probe", mode="simulation",
                           soc_version="ObservedSoC", runner=runner)
    assert result["status"] == "evidence_unavailable"
    assert not result["clean_acquisition_gate_passed"]


@pytest.mark.parametrize("options", [{"extra": "Child process killed by signal 11"}, {"terminal": False},
                                   {"extra": "TimelineDetail data collection failed"}])
def test_zero_exit_does_not_hide_collection_failure(application, tmp_path, options):
    runner, _ = capture_runner(**options)
    result = probe.acquire(application, tmp_path / "capture", "probe", mode="simulation",
                           soc_version="ObservedSoC", runner=runner)
    assert result["status"] == "profiler_failed"
    assert result["simulation"]["partial_capture"]
    assert not result["clean_acquisition_gate_passed"]


def test_workload_failure_prevents_profiler(application, tmp_path):
    runner, commands = capture_runner(correctness=1)
    result = probe.acquire(application, tmp_path / "capture", "probe", runner=runner)
    assert result["status"] == "workload_failed"
    assert len(commands) == 1


def test_timeout_retains_partial_transcript(application, tmp_path):
    runner, _ = capture_runner(timeout=True)
    result = probe.acquire(application, tmp_path / "capture", "probe", mode="simulation",
                           soc_version="ObservedSoC", runner=runner)
    assert result["profiler"]["exit_code"] == 124
    assert (tmp_path / "capture/profiler.stdout.log").read_text() == "partial output"
    assert "timeout" in (tmp_path / "capture/profiler.stderr.log").read_text()
    assert result["status"] == "profiler_failed"


def test_hardware_does_not_accept_simulator_files(application, tmp_path):
    runner, commands = capture_runner()
    result = probe.acquire(application, tmp_path / "capture", "probe", runner=runner)
    assert result["status"] == "requires_hardware_normalization"
    assert not result["clean_acquisition_gate_passed"]
    assert "--aic-metrics=TimelineDetail" in commands[1]
    assert "simulator" not in commands[1]


def test_simulation_requires_explicit_observed_soc(application, tmp_path):
    with pytest.raises(ValueError, match="observed"):
        probe.acquire(application, tmp_path / "capture", "probe", mode="simulation")
    assert not (tmp_path / "capture").exists()


def test_no_device_mask_or_reused_directory_is_rejected(application, tmp_path, monkeypatch):
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES")
    with pytest.raises(ValueError, match="leased"):
        probe.acquire(application, tmp_path / "capture", "probe")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    (tmp_path / "capture").mkdir()
    with pytest.raises(FileExistsError):
        probe.acquire(application, tmp_path / "capture", "probe")


def test_real_timeout_retains_stdout_and_stops_descendants(tmp_path):
    # The child inherits our profiler-like process group and output descriptors.
    # Leaving it alive would prevent communicate() from completing promptly.
    command = [sys.executable, "-c", "import subprocess,sys,time;print('started',flush=True);"
               "subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);time.sleep(60)"]
    result, transcript = probe._run(command, tmp_path, "timeout", 0.1, probe._execute)
    assert result["exit_code"] == 124
    assert "started" in transcript
    assert "timeout" in transcript
