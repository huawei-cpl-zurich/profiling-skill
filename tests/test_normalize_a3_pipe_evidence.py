from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
NORMALIZER = ROOT / "scripts" / "normalize_a3_pipe_evidence.py"
MODEL = ROOT / "references" / "a3-pipe-activity-model.json"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aic_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "block_id": "0", "sub_block_id": "cube0",
        "aic_time(us)": "11.595000", "aic_total_cycles": "20871",
        "aic_cube_ratio": "0.045333", "aic_mte1_ratio": "0.098444",
        "aic_mte2_ratio": "0.544556", "aic_mte3_ratio": "0.000074",
        "aic_fixpipe_ratio": "0.031333", "aic_scalar_ratio": "0.421926",
        "aiv_time(us)": "NA", "aiv_total_cycles": "NA",
        "aiv_vec_ratio": "NA", "aiv_mte2_ratio": "NA",
        "aiv_mte3_ratio": "NA", "aiv_scalar_ratio": "NA",
    }
    row.update(updates)
    return row


def aiv_row(**updates: object) -> dict[str, object]:
    row = aic_row(sub_block_id="vector0", **{
        "aic_time(us)": "NA", "aic_total_cycles": "NA",
        "aic_cube_ratio": "NA", "aic_mte1_ratio": "NA",
        "aic_mte2_ratio": "NA", "aic_mte3_ratio": "NA",
        "aic_fixpipe_ratio": "NA", "aic_scalar_ratio": "NA",
        "aiv_time(us)": "10.5", "aiv_total_cycles": "18900",
        "aiv_vec_ratio": "0.75", "aiv_mte2_ratio": "0.25",
        "aiv_mte3_ratio": "0.1", "aiv_scalar_ratio": "0.2",
    })
    row.update(updates)
    return row


def memory_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "block_id": "0", "sub_block_id": "cube0",
        "GM_to_L1_datas(KB)": "96", "L0C_to_L1_datas(KB)": "0",
        "L0C_to_GM_datas(KB)": "12", "GM_to_UB_datas(KB)": "NA",
        "UB_to_GM_datas(KB)": "NA",
    }
    row.update(updates)
    return row


def run_normalizer(
    tmp_path: Path,
    pipe_rows: list[dict[str, object]],
    *,
    basic_rows: list[dict[str, object]] | None = None,
    memory_rows: list[dict[str, object]] | None = None,
    memory_capture_id: str = "remote:bz-a3-1:job:memory",
    extra: tuple[str, ...] = (),
):
    pipe_dir = tmp_path / "pipe"
    basic = pipe_dir / "OpBasicInfo.csv"
    pipe = pipe_dir / "PipeUtilization.csv"
    write_csv(basic, basic_rows or [
        {"Op Name": "triton_kernel", "Task Duration(us)": "12.125"}
    ])
    write_csv(pipe, pipe_rows)
    command = [
        sys.executable, str(NORMALIZER), "--product", "a3",
        "--activity-model", str(MODEL), "--basic-info", str(basic),
        "--pipe-utilization", str(pipe), "--capture-id",
        "remote:bz-a3-1:job:pipe", "--kernel-name", "triton_kernel",
    ]
    if memory_rows is not None:
        memory_dir = tmp_path / "memory"
        memory_basic = memory_dir / "OpBasicInfo.csv"
        memory = memory_dir / "Memory.csv"
        write_csv(memory_basic, [
            {"Op Name": "triton_kernel", "Task Duration(us)": "12.5"}
        ])
        write_csv(memory, memory_rows)
        command.extend([
            "--memory-basic-info", str(memory_basic), "--memory-access",
            str(memory), "--memory-capture-id", memory_capture_id,
        ])
    return subprocess.run(
        [*command, *extra], text=True, capture_output=True, check=False
    )


def test_emits_non_temporal_unknown_observations(tmp_path: Path):
    run = run_normalizer(tmp_path, [aic_row()])
    assert run.returncode == 0, run.stderr
    evidence = json.loads(run.stdout)
    assert "phases" not in evidence
    assert evidence["evidence_kind"] == "per-block-pipe-activity"
    assert evidence["provenance"]["launch_count"] == 1
    observation = evidence["observations"][0]
    assert observation["observation_id"] == "triton_kernel:block-0:cube0"
    assert observation["duration_ns"] == 11595
    assert "start_ns" not in observation and "end_ns" not in observation
    assert observation["saturation"]["state"] == "unknown"
    assert observation["activity"]["aic_mte2"] == 0.544556
    assert evidence["activity_model"]["model_id"] == "ascend-a2-a3-msprof-activity-v1"


@pytest.mark.parametrize("basic_rows", [
    [
        {"Op Name": "triton_kernel", "Task Duration(us)": "12"},
        {"Op Name": "another_kernel", "Task Duration(us)": "9"},
    ],
    [
        {"Op Name": "triton_kernel", "Task Duration(us)": "12"},
        {"Op Name": "triton_kernel", "Task Duration(us)": "13"},
    ],
])
def test_rejects_multi_row_basic_info_even_when_selector_is_unique(
    tmp_path: Path, basic_rows: list[dict[str, object]]
):
    run = run_normalizer(tmp_path, [aic_row()], basic_rows=basic_rows)
    assert run.returncode == 2
    assert "single-launch" in run.stderr


def test_rejects_duplicate_pipe_topology_keys(tmp_path: Path):
    run = run_normalizer(tmp_path, [aic_row(), aic_row(aic_mte2_ratio="0.4")])
    assert run.returncode == 2
    assert "duplicate pipe topology key" in run.stderr


def test_normalizes_aiv_without_temporal_claim(tmp_path: Path):
    run = run_normalizer(tmp_path, [aiv_row()])
    assert run.returncode == 0, run.stderr
    observation = json.loads(run.stdout)["observations"][0]
    assert observation["engine"] == "aiv"
    assert observation["duration_ns"] == 10500
    assert observation["diagnostics"]["memory_bound"]["interpretation"] == "no_memory_bottleneck"
    assert observation["saturation"]["state"] == "unknown"


def test_memory_join_preserves_distinct_capture_and_unknown_units(tmp_path: Path):
    run = run_normalizer(tmp_path, [aic_row()], memory_rows=[memory_row()])
    assert run.returncode == 0, run.stderr
    evidence = json.loads(run.stdout)
    assert evidence["provenance"]["capture_id"] == "remote:bz-a3-1:job:pipe"
    assert evidence["provenance"]["memory_capture_id"] == "remote:bz-a3-1:job:memory"
    value = evidence["observations"][0]["memory_access"]["gm_to_l1"]
    assert value == {
        "raw_value": "96", "unit": None,
        "source_column_label": "GM_to_L1_datas(KB)", "scaling": "unvalidated",
    }
    assert {item["role"] for item in evidence["provenance"]["sources"]} == {
        "pipe_basic_info", "pipe_utilization", "memory_basic_info", "memory_access",
    }


def test_rejects_same_memory_capture_identity(tmp_path: Path):
    run = run_normalizer(
        tmp_path, [aic_row()], memory_rows=[memory_row()],
        memory_capture_id="remote:bz-a3-1:job:pipe",
    )
    assert run.returncode == 2
    assert "distinct" in run.stderr


def test_rejects_duplicate_memory_topology_keys(tmp_path: Path):
    run = run_normalizer(tmp_path, [aic_row()], memory_rows=[memory_row(), memory_row()])
    assert run.returncode == 2
    assert "duplicate memory topology key" in run.stderr


@pytest.mark.parametrize("memory_rows", [
    [memory_row(block_id="1")],
    [memory_row(), memory_row(block_id="1")],
])
def test_rejects_disjoint_or_extra_memory_topology(
    tmp_path: Path, memory_rows: list[dict[str, object]]
):
    run = run_normalizer(tmp_path, [aic_row()], memory_rows=memory_rows)
    assert run.returncode == 2
    assert "topology key set" in run.stderr


def test_source_digest_binds_stable_roles(tmp_path: Path):
    run = run_normalizer(tmp_path, [aic_row()])
    assert run.returncode == 0, run.stderr
    provenance = json.loads(run.stdout)["provenance"]
    assert [item["role"] for item in provenance["sources"]] == [
        "pipe_basic_info", "pipe_utilization",
    ]
    digest = hashlib.sha256()
    for source in provenance["sources"]:
        digest.update(source["role"].encode())
        digest.update(b"\0")
        digest.update(source["name"].encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(source["sha256"]))
    assert provenance["source_sha256"] == digest.hexdigest()


def test_rejects_bad_ratio_and_preserves_sub_nanosecond_duration(tmp_path: Path):
    invalid = run_normalizer(tmp_path, [aic_row(aic_mte2_ratio="1.01")])
    assert invalid.returncode == 2
    assert "aic_mte2_ratio" in invalid.stderr

    tiny = run_normalizer(tmp_path, [aic_row(**{"aic_time(us)": "0.0005"})])
    assert tiny.returncode == 0, tiny.stderr
    duration = json.loads(tiny.stdout)["observations"][0]["duration_provenance"]
    assert duration == {
        "reported_duration_us": "0.0005", "resolution_ns": 1,
        "rounding": "ceiling",
    }
