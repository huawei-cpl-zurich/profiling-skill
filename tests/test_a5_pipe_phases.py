from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
SUMMARIZER = ROOT / "scripts" / "summarize_timelines.py"
ANALYZER = ROOT / "scripts" / "analyze_pipe_saturation.py"
MODEL = ROOT / "references" / "a5-pipe-capacity-model.json"


def write_trace(path: Path, rows: list[dict], *, truncated: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "displayTimeUnit": "us",
                "metadata": {"truncated": truncated},
                "traceEvents": rows,
            }
        )
    )


def event(pipe: str, start: int, duration: int, *, core: int = 0) -> dict:
    return {
        "ph": "X",
        "ts": start,
        "dur": duration,
        "name": pipe.upper(),
        "cat": pipe,
        "pid": core,
        "tid": 0,
    }


def run_summary(
    tmp_path: Path, rows: list[dict], *, capacity: dict | None = None, truncated: bool = False
):
    trace = tmp_path / "pipe.json"
    output = tmp_path / "out"
    write_trace(trace, rows, truncated=truncated)
    command = [
        sys.executable,
        str(SUMMARIZER),
        "--pipe-timeline",
        str(trace),
        "--output",
        str(output),
    ]
    if capacity is not None:
        capacity_path = tmp_path / "capacity.json"
        capacity_path.write_text(json.dumps(capacity))
        command += ["--capacity-evidence", str(capacity_path)]
    run = subprocess.run(command, text=True, capture_output=True, check=False)
    return run, output, trace


def capacity(trace: Path, phases: list[dict], **changes: object) -> dict:
    payload = {
        "schema_version": 1,
        "product": "a5",
        "target_product": "Ascend950/V6",
        "clock": "pipe-timeline-common-clock-ns",
        "provenance": {
            "capture_id": "bz-a5:profile-17",
            "timeline_source_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
        },
        "phases": phases,
    }
    payload.update(changes)
    return payload


def metrics(cube: object, gm_l1: object) -> dict:
    return {
        "accumulated_cube_block_cycles": cube,
        "whole_device_cube_capacity_cycles": 100,
        "gm_to_l1_bandwidth_gib_s": gm_l1,
        "gm_to_l1_reference_gib_s": 162.77,
    }


def test_common_clock_segmentation_is_deterministic_and_keeps_gaps(tmp_path: Path):
    rows = [
        event("cube", 0, 10),
        event("cube", 0, 10, core=1),
        event("vector", 5, 10),
        event("mte2", 20, 5),
    ]

    first, output, _ = run_summary(tmp_path, rows)
    assert first.returncode == 0, first.stderr
    first_result = json.loads((output / "phase-evidence.json").read_text())
    second, output, _ = run_summary(tmp_path, list(reversed(rows)))
    assert second.returncode == 0, second.stderr
    second_result = json.loads((output / "phase-evidence.json").read_text())
    assert second_result["phases"] == first_result["phases"]

    result = first_result
    assert [
        (phase["start_ns"], phase["end_ns"], phase["activity"]["pipes"])
        for phase in result["phases"]
    ] == [
        (0, 5_000, ["cube"]),
        (5_000, 10_000, ["cube", "vector"]),
        (10_000, 15_000, ["vector"]),
        (15_000, 20_000, []),
        (20_000, 25_000, ["mte2"]),
    ]
    assert result["phases"][3]["metrics"] == {}
    assert result["phases"][3]["capacity_join"]["state"] == "unavailable"


def test_profiler_fractional_nanoseconds_are_quantized_deterministically(tmp_path: Path):
    rows = [
        event("scalar", 1.8915151357650757, 1.7296969890594482),
        event("cube", 2.2799999713897705, 0.001212121220305562),
    ]

    run, output, _ = run_summary(tmp_path, rows)

    assert run.returncode == 0, run.stderr
    result = json.loads((output / "phase-evidence.json").read_text())
    assert result["time_quantization"] == (
        "nearest-nanosecond-half-up; positive events minimum 1ns"
    )
    assert [(phase["start_ns"], phase["end_ns"]) for phase in result["phases"]] == [
        (1_892, 2_280),
        (2_280, 2_281),
        (2_281, 3_621),
    ]


def test_only_exact_common_clock_capacity_windows_are_joined(tmp_path: Path):
    rows = [event("cube", 0, 10), event("mte2", 10, 10)]
    trace = tmp_path / "pipe.json"
    write_trace(trace, rows)
    supplied = capacity(
        trace,
        [
            {"start_ns": 0, "end_ns": 10_000, "metrics": metrics(100, 100)},
            # Whole-task evidence must not be redistributed across either phase.
            {"start_ns": 0, "end_ns": 20_000, "metrics": metrics(100, 162.77)},
        ],
    )

    run, output, _ = run_summary(tmp_path, rows, capacity=supplied)

    assert run.returncode == 0, run.stderr
    phases = json.loads((output / "phase-evidence.json").read_text())["phases"]
    assert phases[0]["metrics"] == metrics(100, 100)
    assert phases[0]["capacity_join"]["state"] == "matched-exact-window"
    assert phases[1]["metrics"] == {}
    assert phases[1]["capacity_join"]["state"] == "unavailable"


def test_capacity_product_and_capture_mismatch_fail_closed(tmp_path: Path):
    rows = [event("cube", 0, 10)]
    trace = tmp_path / "pipe.json"
    write_trace(trace, rows)

    wrong_product = capacity(trace, [], product="a3")
    run, _, _ = run_summary(tmp_path, rows, capacity=wrong_product)
    assert run.returncode != 0
    assert "product" in run.stderr

    wrong_capture = capacity(trace, [])
    wrong_capture["provenance"]["timeline_source_sha256"] = "0" * 64
    run, _, _ = run_summary(tmp_path, rows, capacity=wrong_capture)
    assert run.returncode != 0
    assert "timeline_source_sha256" in run.stderr


def test_truncated_common_timeline_preserves_phases_but_forbids_capacity_join(tmp_path: Path):
    rows = [event("cube", 0, 10)]
    trace = tmp_path / "pipe.json"
    write_trace(trace, rows, truncated=True)
    supplied = capacity(
        trace,
        [{"start_ns": 0, "end_ns": 10_000, "metrics": metrics(100, 162.77)}],
    )

    run, output, _ = run_summary(tmp_path, rows, capacity=supplied, truncated=True)

    assert run.returncode == 0, run.stderr
    result = json.loads((output / "phase-evidence.json").read_text())
    assert result["timeline_complete"] is False
    assert result["phases"][0]["metrics"] == {}
    assert result["phases"][0]["capacity_join"] == {
        "state": "incompatible",
        "reason": "common-clock timeline is marked truncated",
    }


def test_activity_is_not_capacity_and_exact_metrics_drive_all_three_states(tmp_path: Path):
    rows = [
        event("cube", 0, 10),
        event("mte2", 10, 10),
        event("cube", 20, 10),
    ]
    trace = tmp_path / "pipe.json"
    write_trace(trace, rows)
    supplied = capacity(
        trace,
        [
            {"start_ns": 0, "end_ns": 10_000, "metrics": metrics(100, 100)},
            {"start_ns": 10_000, "end_ns": 20_000, "metrics": metrics(50, 80)},
            {
                "start_ns": 20_000,
                "end_ns": 30_000,
                "metrics": {**metrics("NA", 100), "aic_cube_ratio": 0.99},
            },
        ],
    )
    run, output, _ = run_summary(tmp_path, rows, capacity=supplied)
    assert run.returncode == 0, run.stderr

    analyzed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--product",
            "a5",
            "--model",
            str(MODEL),
            "--evidence",
            str(output / "phase-evidence.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert analyzed.returncode == 0, analyzed.stderr
    phases = json.loads(analyzed.stdout)["phases"]
    assert [phase["state"] for phase in phases] == [
        "saturated",
        "unsaturated",
        "unknown",
    ]
    assert phases[1]["resources"][0]["ratio"] == 0.5
    assert phases[1]["state"] != "saturated"  # MTE2 activity is not capacity.
    # High per-row composition cannot replace missing whole-device capacity.
    assert phases[2]["state"] == "unknown"
