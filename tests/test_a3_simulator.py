import copy
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("simulator", ROOT / "scripts/summarize_a3_simulator.py")
sim = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sim)
FIXTURE = ROOT / "tests/fixtures/a3-simulator-excerpts.json"


def payload():
    return json.loads(FIXTURE.read_text())


def report(tmp_path, trace=None):
    root = tmp_path / "report"
    path = root / "OPPROF_observed/simulator/trace.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(payload() if trace is None else trace))
    (tmp_path / "profiler.stdout.log").write_text("[INFO] <ProfInit> Start profiling on kernel: a3_phase_probe\n")
    return root, path


def test_real_observed_string_lane_excerpts_are_simulation_only(tmp_path):
    root, _ = report(tmp_path)
    result = sim.summarize(root, "a3_phase_probe", acquisition_status="partial", acquisition_failure="teardown_failed")
    assert result["interpretable"]
    assert result["event_count"] == 5
    assert result["cores"]["core10.veccore1"]["pipeline_active_us"]["MTE3"] == pytest.approx(.14181818068027496)
    assert result["events"][0]["start_us"] == 11.847878456115723  # displayTimeUnit is not scaling.
    assert result["events"][-1]["process_bytes"] == "512"
    assert result["device_timing"] == result["saturation"] == "unavailable"
    assert not result["clean_acquisition_gate_passed"]
    assert result["partial_capture"] and result["acquisition_failure"] == "teardown_failed"


def test_failed_acquisition_cannot_be_marked_clean(tmp_path):
    root, _ = report(tmp_path)
    with pytest.raises(ValueError, match="partial"):
        sim.summarize(root, "a3_phase_probe", acquisition_status="clean", acquisition_failure="teardown_failed")


def test_barrier_detail_does_not_become_load_or_pipeline_activity(tmp_path):
    trace = payload()
    barrier = trace["traceEvents"][1]
    barrier["tid"] = "SCALAR"
    trace["traceEvents"] = [barrier]
    root, _ = report(tmp_path, trace)
    result = sim.summarize(root, "a3_phase_probe")
    assert result["events"][0]["pipeline"] == "SCALAR"
    assert result["events"][0]["kind"] == "control"
    assert result["cores"] == {}
    assert "phase" not in result["events"][0]


def test_all_lane_is_control_and_cachemiss_is_not_execution_activity(tmp_path):
    barrier = payload()["traceEvents"][1]
    barrier["tid"] = "ALL"
    cache = copy.deepcopy(barrier)
    cache.update(tid="CACHEMISS", name="0x11512074")
    rows, skipped = sim.parse_trace({"traceEvents":[barrier, cache]}, tmp_path / "trace.json")
    assert len(rows) == 1 and rows[0]["kind"] == "control"
    assert skipped["unsupported_lane"] == 1


def test_numeric_lanes_resolve_process_and_thread_metadata(tmp_path):
    event = payload()["traceEvents"][0]
    event.update(pid=30, tid=1)
    trace = {"traceEvents": [
        {"name":"process_name", "ph":"M", "pid":30, "tid":0, "args":{"name":"VECTOR"}},
        {"name":"thread_name", "ph":"M", "pid":30, "tid":1, "args":{"name":"VECTOR_1"}},
        event,
    ]}
    rows, skipped = sim.parse_trace(trace, tmp_path / "core1.veccore0/trace.json")
    assert not skipped
    assert rows[0]["core"] == "core1.veccore0"
    assert rows[0]["pipeline"] == "VECTOR"
    assert rows[0]["thread"] == "VECTOR_1"
    assert rows[0]["lane_provenance"] == "process_name"
    trace["traceEvents"][0]["args"]["name"] = "core1.veccore0"
    trace["traceEvents"][1]["args"]["name"] = "VECTOR"
    assert sim.parse_trace(trace, tmp_path / "core1.veccore0/trace.json")[0][0]["lane_provenance"] == "thread_name"


def test_missing_metadata_never_guesses_numeric_pid(tmp_path):
    event = payload()["traceEvents"][1]
    event.update(pid=60, tid=1)
    rows, skipped = sim.parse_trace({"traceEvents":[event]}, tmp_path / "core1.veccore0/trace.json")
    assert not rows
    assert skipped["missing_core_or_pipeline_metadata"] == 1


def test_aggregate_and_percore_are_not_double_counted(tmp_path):
    root, path = report(tmp_path)
    duplicate = path.parent / "core0.veccore0/trace.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes(path.read_bytes())
    result = sim.summarize(root, "a3_phase_probe")
    assert result["event_count"] == 5
    assert result["diagnostics"]["duplicate_views_ignored"] == 1
    assert len(result["sources"]) == 1


def test_union_activity_and_overlap_are_per_core_not_cross_core(tmp_path):
    original = payload()["traceEvents"][0]
    events = []
    for core, pipe, start, duration in [(0,"MTE2",0,3),(0,"MTE2",1,3),(0,"VECTOR",2,3),(1,"VECTOR",0,5)]:
        event = copy.deepcopy(original)
        event.update(pid=f"core{core}.veccore0", tid=pipe, ts=start, dur=duration)
        events.append(event)
    root, _ = report(tmp_path, {"traceEvents":events})
    result = sim.summarize(root, "a3_phase_probe")
    core = result["cores"]["core0.veccore0"]
    assert core["pipeline_active_us"]["MTE2"] == 4
    assert core["pipeline_overlap_us"]["MTE2:VECTOR"] == 2
    assert result["cores"]["core1.veccore0"]["pipeline_overlap_us"] == {}


@pytest.mark.parametrize("failure", ["no-log", "wrong-selector", "ambiguous-log", "multiple-reports"])
def test_no_kernel_binding_fails_closed(tmp_path, failure):
    root, path = report(tmp_path)
    log = tmp_path / "profiler.stdout.log"
    if failure == "no-log":
        log.unlink()
    elif failure == "wrong-selector":
        log.write_text("<ProfInit> Start profiling on kernel: another_kernel\n")
    elif failure == "ambiguous-log":
        log.write_text(log.read_text() + "<ProfInit> Start profiling on kernel: another_kernel\n")
    else:
        duplicate = root / "OPPROF_other/simulator/trace.json"
        duplicate.parent.mkdir(parents=True)
        duplicate.write_bytes(path.read_bytes())
    result = sim.summarize(root, "a3_phase_probe")
    assert result["raw_trace_obtained"]
    assert not result["interpretable"]
    assert result["events"] == [] and result["cores"] == {}


def test_cli_output_is_bounded_but_summarizes_complete_trace(tmp_path):
    trace = payload()
    trace["traceEvents"] *= 100
    root, _ = report(tmp_path, trace)
    result = subprocess.run([sys.executable, str(SPEC.origin), "--report", str(root), "--kernel-name", "a3_phase_probe"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    value = json.loads(result.stdout)
    assert value["event_count"] == 500 and len(value["events"]) == 256
    assert value["events_truncated"]
    assert not value["clean_acquisition_gate_passed"]


def test_hardware_acquisition_excludes_simulator_and_operand_phase_guesses(tmp_path):
    spec = importlib.util.spec_from_file_location("acquire", ROOT / "scripts/acquire_profile.py")
    acquisition = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(acquisition)
    workload = tmp_path / "work.py"
    workload.write_text("pass\n")
    rendered = acquisition.render_remote_payload("timeline", "a3", "bz-a3-1", "key", workload, [], "a3_phase_probe", 1)
    names = {"number", "first", "timeline_phase", "normalize_event", "timeline_items", "compact_timeline"}
    helpers = [node for node in ast.parse(rendered).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = {"json": json, "csv": __import__("csv"), "KERNEL_NAMES":["a3_phase_probe"]}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), "rendered", "exec"), namespace)
    root, trace = report(tmp_path)
    result = namespace["compact_timeline"]([trace])
    assert result["event_count"] == 0
    assert not result["selectors_complete"]
    assert result["simulator_inputs_excluded"] == 1
    barrier = payload()["traceEvents"][1]
    assert "phase" not in namespace["normalize_event"](barrier)
