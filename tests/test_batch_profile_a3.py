from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "scripts/batch_profile_a3.py"


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
