from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/batch_profile_a3.py"
CLIENT = ROOT / "scripts/gz_a3_job_client.py"


def load_client():
    spec = importlib.util.spec_from_file_location("gz_a3_job_client_batch", CLIENT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def load_driver():
    spec = importlib.util.spec_from_file_location("batch_profile_a3_test", DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path: Path, mode: str = "ok") -> tuple[list[str], Path, Path]:
    job = {
        "benchmark": "matmul", "action": "profile", "device": 2,
        "cases": [7, 9], "repeats": 3, "round": 2,
        "profiling": {"kernel_name": "chosen_kernel"},
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    runner = tmp_path / "runner.py"
    runner.write_text("# invoked by fake profiler\n")
    profiler = tmp_path / "profiler.py"
    profiler.write_text(f'''#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
args=sys.argv[1:]
def value(name): return args[args.index(name)+1]
out=Path(value("--output")); out.mkdir(parents=True)
job_path=Path(value("--job")); response_path=Path(args[-1])
job=json.loads(job_path.read_text()); case=job["case"]; iteration=job["iteration"]
mode={mode!r}
status="compile_error" if mode=="compile" and case==9 else "ok"
response={{"status":status,"diagnostics":"NameError in candidate.py" if status != "ok" else "",
          "benchmark":job["benchmark"],"action":"profile","device":job["device"],
          "case":case,"round":job["round"],"kernel_name":job["profiling"]["kernel_name"]}}
response_path.write_text(json.dumps(response))
log=out/"msprof.log"; log.write_text(f"capture {{case}}/{{iteration}}\\n")
evidence={{"status":"failure"}}
if status == "ok":
 evidence={{"status":"success","msprof_log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),
            "kernels":[{{"name":"chosen_kernel","duration_us":{{"median":case*10+iteration+1}}}}]}}
(out/"evidence.json").write_text(json.dumps(evidence))
raise SystemExit(0 if status == "ok" else 1)
''')
    output = tmp_path / "profile"
    response = tmp_path / "response.json"
    command = [sys.executable, str(DRIVER), "--job", str(job_path), "--runner", str(runner),
               "--profiler", str(profiler), "--output", str(output),
               "--response", str(response), "--kernel-name", "chosen_kernel"]
    return command, output, response


def selector_fixture(tmp_path: Path, declared: str, outcomes: dict[str, str],
                     *, cases: list[int] | None = None, repeats: int = 2):
    cases = cases or [7, 9]
    job = {
        "benchmark": "matmul", "action": "profile", "device": 2,
        "cases": cases, "repeats": repeats, "round": 2,
        "profiling": {"kernel_name": declared},
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    runner = tmp_path / "runner.py"
    runner.write_text("# invoked by fake profiler\n")
    calls = tmp_path / "selectors.jsonl"
    profiler = tmp_path / "profiler.py"
    profiler.write_text(f'''#!/usr/bin/env python3
import hashlib, json, sys
from pathlib import Path
args=sys.argv[1:]
def value(name): return args[args.index(name)+1]
out=Path(value("--output")); out.mkdir(parents=True)
selector=value("--kernel-name")
with {str(calls)!r} and Path({str(calls)!r}).open("a") as stream: stream.write(json.dumps(selector)+"\\n")
job_path=Path(value("--job")); response_path=Path(args[-1])
job=json.loads(job_path.read_text()); case=job["case"]; iteration=job["iteration"]
response={{"status":"ok","diagnostics":"","benchmark":job["benchmark"],"action":"profile",
          "device":job["device"],"case":case,"round":job["round"],
          "kernel_name":job["profiling"]["kernel_name"]}}
response_path.write_text(json.dumps(response))
log=out/"msprof.log"; log.write_text("capture\\n")
mode={outcomes!r}.get(selector, "selector_miss")
if mode == "success":
 evidence={{"status":"success","msprof_log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),
            "kernels":[{{"name":selector,"duration_us":{{"median":case*10+iteration+1}}}}]}}
 rc=0
elif mode == "selector_miss":
 evidence={{"status":"failure","failure":{{"kind":"profiling","reason":"selector_miss",
            "kernel_selector":selector,"message":f"no numeric operator rows matching {{selector!r}}",
            "msprof_returncode":0}}}}
 rc=1
elif mode == "malformed":
 evidence={{"status":"failure","failure":{{"kind":"profiling","reason":"invalid_capture",
            "message":"no OpBasicInfo CSV","msprof_returncode":0}}}}
 rc=1
else:
 evidence={{"status":"failure","failure":{{"kind":"profiling","reason":"tool_failure",
            "message":"msprof exited with status 17","msprof_returncode":17}}}}
 rc=1
(out/"evidence.json").write_text(json.dumps(evidence))
raise SystemExit(rc)
''')
    output = tmp_path / "profile"
    response = tmp_path / "response.json"
    command = [sys.executable, str(DRIVER), "--job", str(job_path), "--runner", str(runner),
               "--profiler", str(profiler), "--output", str(output),
               "--response", str(response), "--kernel-name", declared]
    return command, output, response, calls


def test_batch_profiles_ordered_matrix_and_returns_compact_evidence(tmp_path: Path):
    command, output, response = fixture(tmp_path)
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())
    evidence = json.loads((output / "evidence.json").read_text())

    assert run.returncode == 0
    assert result["cases"] == [7, 9]
    assert result["profile_cases"] == [
        {"case": 7, "median_us": 72.0, "samples_us": [71.0, 72.0, 73.0]},
        {"case": 9, "median_us": 92.0, "samples_us": [91.0, 92.0, 93.0]},
    ]
    assert [(item["case"], item["iteration"]) for item in evidence["captures"]] == [
        (7, 0), (7, 1), (7, 2), (9, 0), (9, 1), (9, 2),
    ]
    assert len((output / "msprof.log").read_text()) < 64 * 1024

    # Exercise the production client's wire validation against artifacts made
    # by the real batch driver, rather than a separately maintained fixture.
    client = load_client()
    client.attach_profile_evidence(result, output / "evidence.json")
    assert result["cases"] == [7, 9]
    assert result["profile"]["cases"] == result["profile_cases"]


def test_managed_batch_adds_normalized_pipe_activity_after_timing(tmp_path: Path):
    job = {
        "benchmark": "matmul", "action": "profile", "device": 0,
        "cases": [7], "repeats": 2, "round": 3,
        "profiling": {"kernel_name": "chosen_kernel", "pipe_activity": True},
    }
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(job))
    runner = tmp_path / "runner.py"
    runner.write_text("# invoked by fake profiler\n")
    profiler = tmp_path / "profiler.py"
    profiler.write_text('''#!/usr/bin/env python3
import csv, hashlib, json, sys
from pathlib import Path
args=sys.argv[1:]
def value(name): return args[args.index(name)+1]
out=Path(value("--output")); out.mkdir(parents=True)
job_path=Path(value("--job")); response_path=Path(args[-1])
job=json.loads(job_path.read_text())
response_path.write_text(json.dumps({"status":"ok","diagnostics":""}))
log=out/"msprof.log"; log.write_text("capture\\n")
if "--metric" in args and value("--metric") == "PipeUtilization":
 raw=out/"raw"; raw.mkdir()
 basic=raw/"OpBasicInfo.csv"; pipe=raw/"PipeUtilization.csv"
 with basic.open("w", newline="") as stream:
  writer=csv.DictWriter(stream, fieldnames=["Op Name","Task Duration(us)"])
  writer.writeheader(); writer.writerow({"Op Name":"chosen_kernel","Task Duration(us)":"7.0"})
 with pipe.open("w", newline="") as stream:
  fields=["block_id","sub_block_id","aic_time(us)","aiv_time(us)","aic_total_cycles","aic_cube_ratio","aic_mte2_ratio"]
  writer=csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
  writer.writerow({"block_id":"0","sub_block_id":"cube0","aic_time(us)":"7.0","aiv_time(us)":"NA","aic_total_cycles":"12600","aic_cube_ratio":"0.4","aic_mte2_ratio":"0.5"})
 evidence={"status":"success","metric":"PipeUtilization","sources":[
  {"role":"pipe_basic_info","path":"OpBasicInfo.csv","sha256":hashlib.sha256(basic.read_bytes()).hexdigest()},
  {"role":"pipe_utilization","path":"PipeUtilization.csv","sha256":hashlib.sha256(pipe.read_bytes()).hexdigest()}]}
else:
 evidence={"status":"success","msprof_log_sha256":hashlib.sha256(log.read_bytes()).hexdigest(),
           "kernels":[{"name":"chosen_kernel","duration_us":{"median":7.0}}]}
(out/"evidence.json").write_text(json.dumps(evidence))
''')
    output, response = tmp_path / "profile", tmp_path / "response.json"
    run = subprocess.run([
        sys.executable, str(DRIVER), "--job", str(job_path), "--runner", str(runner),
        "--profiler", str(profiler), "--normalizer",
        str(ROOT / "scripts/normalize_a3_pipe_evidence.py"), "--activity-model",
        str(ROOT / "references/a3-pipe-activity-model.json"), "--output", str(output),
        "--response", str(response), "--kernel-name", "chosen_kernel",
    ], text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    evidence = json.loads((output / "evidence.json").read_text())
    activity = evidence["pipe_activity"]
    assert activity["evidence_kind"] == "per-block-pipe-activity"
    assert activity["observations"][0]["activity"] == {
        "aic_cube": 0.4, "aic_mte2": 0.5,
    }
    assert activity["observations"][0]["saturation"]["state"] == "unknown"
    assert activity["provenance"]["launch_count"] == 1


def test_pipe_selector_miss_after_resolved_timing_is_infrastructure(tmp_path: Path):
    profiler = tmp_path / "profiler.py"
    profiler.write_text('''#!/usr/bin/env python3
import json, sys
from pathlib import Path
args=sys.argv[1:]
def value(name): return args[args.index(name)+1]
out=Path(value("--output")); out.mkdir(parents=True)
response=Path(args[-1]); response.write_text(json.dumps({"status":"ok","diagnostics":""}))
(out/"msprof.log").write_text("clean selector miss\\n")
(out/"evidence.json").write_text(json.dumps({
 "status":"failure", "failure":{"kind":"profiling", "reason":"selector_miss",
 "kernel_selector":"resolved_kernel", "message":"no PipeUtilization export for resolved_kernel",
 "msprof_returncode":0}}))
raise SystemExit(1)
''')
    work, output = tmp_path / "work", tmp_path / "output"
    work.mkdir(); output.mkdir()
    pipe, status, diagnostics = load_driver().capture_pipe(
        {
            "benchmark": "matmul", "action": "profile", "device": 0,
            "cases": [7], "repeats": 3, "round": 2,
            "profiling": {"kernel_name": "declared_mix_aic", "pipe_activity": True},
        },
        runner=tmp_path / "runner.py", profiler=profiler,
        normalizer=ROOT / "scripts/normalize_a3_pipe_evidence.py",
        activity_model=ROOT / "references/a3-pipe-activity-model.json",
        work=work, output=output, selector="resolved_kernel", logs=[],
    )
    assert pipe is None
    assert status == "infrastructure_error"
    assert diagnostics == "no PipeUtilization export for resolved_kernel"


def test_batch_preserves_candidate_failure_and_partial_evidence(tmp_path: Path):
    command, output, response = fixture(tmp_path, "compile")
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())
    evidence = json.loads((output / "evidence.json").read_text())

    assert run.returncode == 1
    assert result["status"] == "compile_error"
    assert "NameError" in result["diagnostics"]
    assert len(result["completed_captures"]) == 3
    assert evidence["failure"]["kind"] == "compile_error"


def test_suffix_selector_falls_back_once_then_caches_resolution(tmp_path: Path):
    declared = "candidate_kernel_mix_aiv"
    command, output, response, calls = selector_fixture(
        tmp_path, declared, {declared: "selector_miss", "candidate_kernel": "success"})

    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())
    evidence = json.loads((output / "evidence.json").read_text())

    assert run.returncode == 0
    assert result["kernel_name"] == "candidate_kernel"
    assert result["resolved_kernel_name"] == "candidate_kernel"
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [
        declared, "candidate_kernel", "candidate_kernel", "candidate_kernel", "candidate_kernel",
    ]
    assert evidence["kernel_name"] == "candidate_kernel"
    assert evidence["declared_kernel_name"] == declared
    assert evidence["resolved_kernel_name"] == "candidate_kernel"
    assert all(capture["kernel_name"] == "candidate_kernel"
               and capture["declared_kernel_name"] == declared
               and capture["resolved_kernel_name"] == "candidate_kernel"
               for capture in evidence["captures"])


def test_arbitrary_selector_miss_is_counted_submission_failure(tmp_path: Path):
    command, output, response, calls = selector_fixture(
        tmp_path, "BatchMatMulV2", {"BatchMatMulV2": "selector_miss"})
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())

    assert run.returncode == 1
    assert result["status"] == "submission_error"
    assert result["kernel_name"] == "BatchMatMulV2"
    assert result["completed_captures"] == []
    assert [json.loads(line) for line in calls.read_text().splitlines()] == ["BatchMatMulV2"]
    assert "selector" in result["diagnostics"]
    assert json.loads((output / "evidence.json").read_text())["captures"] == []


def test_suffix_fallback_final_miss_is_submission_failure(tmp_path: Path):
    declared = "missing_mix_aic"
    command, _output, response, calls = selector_fixture(tmp_path, declared, {})
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())

    assert run.returncode == 1
    assert result["status"] == "submission_error"
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [declared, "missing"]
    assert "missing" in result["diagnostics"]


def test_profiler_tool_failure_does_not_trigger_fallback(tmp_path: Path):
    declared = "candidate_kernel_mix_aiv"
    command, _output, response, calls = selector_fixture(
        tmp_path, declared, {declared: "tool_failure"})
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())

    assert run.returncode == 1
    assert result["status"] == "infrastructure_error"
    assert "status 17" in result["diagnostics"]
    assert [json.loads(line) for line in calls.read_text().splitlines()] == [declared]


def test_malformed_profiler_evidence_preserves_completed_captures(tmp_path: Path):
    declared = "candidate_kernel_mix_aiv"
    command, output, response, calls = selector_fixture(
        tmp_path, declared, {declared: "selector_miss", "candidate_kernel": "success"},
        cases=[7, 9], repeats=2)
    # Let the resolved selector fail only after two successful captures.
    profiler = Path(command[command.index("--profiler") + 1])
    source = profiler.read_text()
    source = source.replace(
        'mode={',
        'mode="malformed" if selector == "candidate_kernel" and case == 9 else {',
        1,
    )
    profiler.write_text(source)

    run = subprocess.run(command, text=True, capture_output=True, check=False)
    result = json.loads(response.read_text())
    evidence = json.loads((output / "evidence.json").read_text())

    assert run.returncode == 1
    assert result["status"] == "infrastructure_error"
    assert len(result["completed_captures"]) == 2
    assert len(evidence["captures"]) == 2
    assert result["cases"] == [7, 9]
    assert [json.loads(line) for line in calls.read_text().splitlines()][-1] == "candidate_kernel"
