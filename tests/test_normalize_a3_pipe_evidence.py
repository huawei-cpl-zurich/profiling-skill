from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]
NORMALIZER = ROOT / "scripts" / "normalize_a3_pipe_evidence.py"
ANALYZER = ROOT / "scripts" / "analyze_pipe_saturation.py"
MODEL = ROOT / "references" / "a3-pipe-model.json"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_normalizer(tmp_path: Path, pipe_rows: list[dict[str, object]], *extra: str):
    pipe = tmp_path / "OpPipeUtilization.csv"
    basic = tmp_path / "OpBasicInfo.csv"
    write_csv(pipe, pipe_rows)
    write_csv(
        basic,
        [{"Op Name": "triton_kernel", "Task Duration(us)": "12.125", "Device Id": "4"}],
    )
    return subprocess.run(
        [
            sys.executable,
            str(NORMALIZER),
            "--product",
            "a3",
            "--pipe-utilization",
            str(pipe),
            "--basic-info",
            str(basic),
            "--capture-id",
            "remote:bz-a3-1:job:fixture",
            "--kernel-name",
            "triton_kernel",
            *extra,
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def representative_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "block_id": "0",
        "sub_block_id": "cube0",
        "aic_time(us)": "11.595000",
        "aic_total_cycles": "20871",
        "aic_cube_ratio": "0.045333",
        "aic_mte1_ratio": "0.098444",
        "aic_mte2_ratio": "0.544556",
        "aic_mte3_ratio": "0.000074",
        "aic_fixpipe_ratio": "0.031333",
        "aic_scalar_ratio": "0.421926",
        "aiv_time(us)": "NA",
        "aiv_total_cycles": "NA",
        "aiv_vec_ratio": "NA",
        "aiv_mte2_ratio": "NA",
        "aiv_mte3_ratio": "NA",
        "aiv_scalar_ratio": "NA",
    }
    row.update(updates)
    return row


def test_normalizes_activity_without_claiming_capacity(tmp_path: Path):
    run = run_normalizer(tmp_path, [representative_row()], "--kernel-name", "triton_kernel")

    assert run.returncode == 0, run.stderr
    evidence = json.loads(run.stdout)
    assert evidence["product"] == "a3"
    assert evidence["provenance"]["capture_id"] == "remote:bz-a3-1:job:fixture"
    assert len(evidence["provenance"]["source_sha256"]) == 64
    phase = evidence["phases"][0]
    assert phase["phase_id"] == "triton_kernel:block-0:cube0"
    assert (phase["start_ns"], phase["end_ns"]) == (0, 11595)
    assert phase["activity"] == {
        "aic_cube": 0.045333,
        "aic_fixpipe": 0.031333,
        "aic_mte1": 0.098444,
        "aic_mte2": 0.544556,
        "aic_mte3": 0.000074,
        "aic_scalar": 0.421926,
    }
    assert phase["metrics"]["aic_total_cycles"] == 20871
    assert phase["diagnostics"]["memory_bound"]["interpretation"] == "memory_activity_dominates"

    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(run.stdout)
    analyzed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--product",
            "a3",
            "--model",
            str(MODEL),
            "--evidence",
            str(evidence_path),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert analyzed.returncode == 0, analyzed.stderr
    result = json.loads(analyzed.stdout)
    assert result["phases"][0]["state"] == "unknown"
    assert all(item["state"] == "unknown" for item in result["phases"][0]["resources"])


def test_preserves_memory_paths_with_unknown_scaling(tmp_path: Path):
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    memory = memory_dir / "Memory.csv"
    memory_basic = memory_dir / "OpBasicInfo.csv"
    write_csv(memory_basic, [{"Op Name": "triton_kernel", "Task Duration(us)": "12.5"}])
    write_csv(
        memory,
        [
            {
                "block_id": "0",
                "sub_block_id": "cube0",
                "GM_to_L1_datas(KB)": "96",
                "L0C_to_L1_datas(KB)": "0",
                "L0C_to_GM_datas(KB)": "12",
                "GM_to_UB_datas(KB)": "NA",
                "UB_to_GM_datas(KB)": "NA",
            }
        ],
    )
    run = run_normalizer(
        tmp_path,
        [representative_row()],
        "--kernel-name",
        "triton_kernel",
        "--memory-access",
        str(memory),
        "--memory-basic-info",
        str(memory_basic),
    )

    assert run.returncode == 0, run.stderr
    phase = json.loads(run.stdout)["phases"][0]
    assert phase["memory_access"] == {
        "gm_to_l1": {"raw_value": "96", "reported_unit": "KB", "scaling": "unvalidated"},
        "gm_to_ub": {"raw_value": "NA", "reported_unit": "KB", "scaling": "unvalidated"},
        "l0c_to_gm": {"raw_value": "12", "reported_unit": "KB", "scaling": "unvalidated"},
        "l0c_to_l1": {"raw_value": "0", "reported_unit": "KB", "scaling": "unvalidated"},
        "ub_to_gm": {"raw_value": "NA", "reported_unit": "KB", "scaling": "unvalidated"},
    }


def test_memory_bound_boundary_is_unclassified(tmp_path: Path):
    run = run_normalizer(
        tmp_path,
        [representative_row(aic_mte2_ratio="0.5", aic_cube_ratio="0.5")],
    )
    assert run.returncode == 0, run.stderr
    diagnostic = json.loads(run.stdout)["phases"][0]["diagnostics"]["memory_bound"]
    assert diagnostic["interpretation"] == "unclassified_boundary"


def test_normalizes_aiv_row_and_no_memory_bottleneck(tmp_path: Path):
    row = representative_row(
        sub_block_id="vector0",
        **{
            "aic_time(us)": "NA",
            "aic_total_cycles": "NA",
            "aic_cube_ratio": "NA",
            "aic_mte1_ratio": "NA",
            "aic_mte2_ratio": "NA",
            "aic_mte3_ratio": "NA",
            "aic_fixpipe_ratio": "NA",
            "aic_scalar_ratio": "NA",
            "aiv_time(us)": "10.5",
            "aiv_total_cycles": "18900",
            "aiv_vec_ratio": "0.75",
            "aiv_mte2_ratio": "0.25",
            "aiv_mte3_ratio": "0.1",
            "aiv_scalar_ratio": "0.2",
        },
    )
    run = run_normalizer(tmp_path, [row])
    assert run.returncode == 0, run.stderr
    phase = json.loads(run.stdout)["phases"][0]
    assert phase["end_ns"] == 10500
    assert phase["activity"] == {
        "aiv_mte2": 0.25,
        "aiv_mte3": 0.1,
        "aiv_scalar": 0.2,
        "aiv_vector": 0.75,
    }
    assert phase["diagnostics"]["memory_bound"]["interpretation"] == "no_memory_bottleneck"


def test_rejects_selector_miss_and_bad_activity_ratio(tmp_path: Path):
    missed = run_normalizer(tmp_path, [representative_row()], "--kernel-name", "another_kernel")
    assert missed.returncode == 2
    assert "expected exactly one basic-info row" in missed.stderr

    invalid = run_normalizer(tmp_path, [representative_row(aic_mte2_ratio="1.01")])
    assert invalid.returncode == 2
    assert "aic_mte2_ratio" in invalid.stderr


def test_rejects_a5_product_and_preserves_sub_nanosecond_report(tmp_path: Path):
    pipe = tmp_path / "OpPipeUtilization.csv"
    basic = tmp_path / "OpBasicInfo.csv"
    write_csv(pipe, [representative_row(**{"aic_time(us)": "0.0005"})])
    write_csv(basic, [{"Op Name": "triton_kernel", "Task Duration(us)": "1", "Device Id": "4"}])
    wrong_product = subprocess.run(
        [
            sys.executable,
            str(NORMALIZER),
            "--product",
            "a5",
            "--pipe-utilization",
            str(pipe),
            "--basic-info",
            str(basic),
            "--capture-id",
            "capture",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert wrong_product.returncode == 2
    assert "invalid choice" in wrong_product.stderr

    sub_nanosecond = run_normalizer(
        tmp_path, [representative_row(**{"aic_time(us)": "0.0005"})]
    )
    assert sub_nanosecond.returncode == 0, sub_nanosecond.stderr
    phase = json.loads(sub_nanosecond.stdout)["phases"][0]
    assert phase["end_ns"] == 1
    assert phase["interval_provenance"]["reported_duration_us"] == "0.0005"
    assert phase["interval_provenance"]["resolution_ns"] == 1
    assert phase["interval_provenance"]["rounding"] == "ceiling"
    assert phase["interval_provenance"]["scope"] == "per-export-row-task-relative"
    assert phase["interval_provenance"]["cross_row_alignable"] is False
