from __future__ import annotations

import csv
import gzip
import io
import importlib.util
import json
import sqlite3
import subprocess
import tarfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).parents[1]


def fake_msprof(path: Path, body: str, *, returncode: int = 0, success=True) -> Path:
    script = path / "msprof"
    script.write_text(
        "#!/usr/bin/env python3\n"
        "import csv\n"
        "import pathlib\n"
        "import sys\n"
        "output = next(x.split('=', 1)[1] for x in sys.argv if x.startswith('--output='))\n"
        "target = pathlib.Path(output) / 'OPPROF_test' / 'kernel' / '0'\n"
        "target.mkdir(parents=True)\n"
        + textwrap.dedent(body).rstrip()
        + "\n"
        + (f"print({SUCCESS_LINE!r})\n" if success else "print('profiler stopped')\n")
        + f"raise SystemExit({returncode})\n"
    )
    script.chmod(0o755)
    return script


SUCCESS_LINE = "Profiling running finished. All task success."


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def sample_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE STRING_IDS(id INTEGER,value TEXT);"
        "CREATE TABLE SAMPLE_PMU_TIMELINE(deviceId INTEGER,timestampNs INTEGER,totalCycle INTEGER,usage REAL,freq REAL,coreId INTEGER,coreType INTEGER);"
        "CREATE TABLE SAMPLE_PMU_SUMMARY(deviceId INTEGER,metric INTEGER,value REAL,coreId INTEGER,coreType INTEGER);"
        "CREATE TABLE TASK(globalTaskId INTEGER,startNs INTEGER,endNs INTEGER);"
        "CREATE TABLE COMPUTE_TASK_INFO(globalTaskId INTEGER,name INTEGER);"
    )
    conn.executemany(
        "INSERT INTO STRING_IDS VALUES(?,?)",
        [(1, "AIC"), (2, "AIV"), (3, "target_kernel"), (4, "util")],
    )
    conn.execute("INSERT INTO TASK VALUES(7,100,400)")
    conn.execute("INSERT INTO COMPUTE_TASK_INFO VALUES(7,3)")
    conn.executemany(
        "INSERT INTO SAMPLE_PMU_TIMELINE VALUES(?,?,?,?,?,?,?)",
        [
            (0, 50, 4, 0, 1650, 0, 1),
            (0, 100, 10, 0.5, 1650, 0, 1),
            (0, 200, 20, 1.0, 1650, 0, 1),
            (0, 300, 15, 0.75, 1600, 1, 1),
            (0, 200, 8, 0.25, 1650, 0, 2),
            (0, 450, 3, 0, 1650, 0, 2),
        ],
    )
    conn.execute("INSERT INTO SAMPLE_PMU_SUMMARY VALUES(0,4,0.7,0,1)")
    conn.commit()
    conn.close()


def test_sample_summary_filters_task_and_separates_domains(tmp_path: Path):
    sample_database(tmp_path / "msprof_1.db")
    module = load("summarize_sample_pmu")
    summary, rows = module.summarize(tmp_path, "target*")
    assert len(rows) == 4
    assert summary["task_window"] == (100, 400, 1)
    assert summary["domains"]["AIC"]["cycles"] == 45
    assert summary["domains"]["AIC"]["active_cores"] == 2
    assert summary["domains"]["AIV"]["cycles"] == 8
    assert summary["domains"]["AIC"]["frequency_mhz"]["min"] == 1600
    assert summary["sample_pmu_summary"] == []


def test_sample_summary_keeps_capture_wide_rows_without_task_filter(tmp_path: Path):
    sample_database(tmp_path / "msprof_1.db")
    module = load("summarize_sample_pmu")
    summary, _ = module.summarize(tmp_path, None)
    assert summary["sample_pmu_summary"] == [
        {
            "device_id": 0,
            "metric": "util",
            "value": 0.7,
            "core_id": 0,
            "core_type": "AIC",
        }
    ]


def test_sample_summary_rejects_ambiguous_database_root(tmp_path: Path):
    sample_database(tmp_path / "msprof_1.db")
    nested = tmp_path / "another-capture"
    nested.mkdir()
    sample_database(nested / "msprof_2.db")
    module = load("summarize_sample_pmu")
    try:
        module.summarize(tmp_path, None)
    except module.InvalidProfile as error:
        assert "multiple sampled-PMU databases" in str(error)
    else:
        raise AssertionError("ambiguous sampled-PMU root unexpectedly summarized")


def test_sample_summary_rejects_multi_device_capture(tmp_path: Path):
    database = tmp_path / "msprof.db"
    sample_database(database)
    conn = sqlite3.connect(database)
    conn.execute("INSERT INTO SAMPLE_PMU_TIMELINE VALUES(1,200,10,1,1650,0,1)")
    conn.commit()
    conn.close()
    module = load("summarize_sample_pmu")
    try:
        module.summarize(tmp_path, None)
    except module.InvalidProfile as error:
        assert "multiple devices" in str(error)
    else:
        raise AssertionError("multi-device capture unexpectedly summarized")


def test_sample_summary_rejects_unrelated_task_between_matches(tmp_path: Path):
    database = tmp_path / "msprof.db"
    sample_database(database)
    conn = sqlite3.connect(database)
    conn.execute("INSERT INTO STRING_IDS VALUES(5, 'unrelated')")
    conn.execute("INSERT INTO TASK VALUES(8,500,600)")
    conn.execute("INSERT INTO COMPUTE_TASK_INFO VALUES(8,3)")
    conn.execute("INSERT INTO TASK VALUES(9,425,475)")
    conn.execute("INSERT INTO COMPUTE_TASK_INFO VALUES(9,5)")
    conn.execute("INSERT INTO SAMPLE_PMU_TIMELINE VALUES(0,450,10,1,1650,0,1)")
    conn.commit()
    conn.close()
    module = load("summarize_sample_pmu")
    try:
        module.summarize(tmp_path, "target*")
    except module.InvalidProfile as error:
        assert "encloses unrelated task" in str(error)
    else:
        raise AssertionError("interleaved task envelope unexpectedly summarized")


def test_sample_summary_accepts_substring_fallback_task_match(tmp_path: Path):
    database = tmp_path / "msprof.db"
    sample_database(database)
    conn = sqlite3.connect(database)
    conn.execute(
        "UPDATE STRING_IDS SET value='prefix_target_kernel_suffix' WHERE id=3"
    )
    conn.commit()
    conn.close()
    module = load("summarize_sample_pmu")
    summary, rows = module.summarize(tmp_path, "target_kernel")
    assert len(rows) == 4
    assert summary["task_window"] == (100, 400, 1)


def test_sample_summary_accepts_task_touching_filter_boundary(tmp_path: Path):
    database = tmp_path / "msprof.db"
    sample_database(database)
    conn = sqlite3.connect(database)
    conn.execute("INSERT INTO STRING_IDS VALUES(5, 'unrelated')")
    conn.execute("INSERT INTO TASK VALUES(8,400,500)")
    conn.execute("INSERT INTO COMPUTE_TASK_INFO VALUES(8,5)")
    conn.execute("INSERT INTO SAMPLE_PMU_TIMELINE VALUES(0,400,10,1,1650,0,1)")
    conn.commit()
    conn.close()
    module = load("summarize_sample_pmu")
    summary, rows = module.summarize(tmp_path, "target*")
    assert summary["task_window"] == (100, 400, 1)
    assert len(rows) == 4
    assert all(row["timestamp_ns"] < 400 for row in rows)


def test_sample_cli_writes_compact_outputs(tmp_path: Path):
    sample_database(tmp_path / "msprof.db")
    output = tmp_path / "out"
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/summarize_sample_pmu.py"),
            "--input",
            str(tmp_path),
            "--output",
            str(output),
            "--task-pattern",
            "target_kernel",
        ],
        check=True,
    )
    assert (
        json.loads((output / "sample-pmu-summary.json").read_text())["domains"]["AIC"][
            "state"
        ]
        == "ok"
    )
    with gzip.open(output / "sample-pmu-rows.csv.gz", "rt") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert (output / "sample-pmu-rows.csv.gz").read_bytes()[4:8] == b"\0\0\0\0"


def test_sample_summary_rejects_unresolvable_explicit_task_filter(tmp_path: Path):
    database = tmp_path / "msprof.db"
    conn = sqlite3.connect(database)
    conn.execute(
        "CREATE TABLE SAMPLE_PMU_TIMELINE(deviceId INTEGER,timestampNs INTEGER,totalCycle INTEGER,usage REAL,freq REAL,coreId INTEGER,coreType INTEGER)"
    )
    conn.execute("INSERT INTO SAMPLE_PMU_TIMELINE VALUES(0,1,1,1,1600,0,1)")
    conn.commit()
    conn.close()
    module = load("summarize_sample_pmu")
    try:
        module.summarize(tmp_path, "target*")
    except module.InvalidProfile as error:
        assert "cannot resolve requested task filter" in str(error)
    else:
        raise AssertionError("explicit task filter unexpectedly summarized all rows")


def trace(path: Path, events: list[dict]) -> None:
    path.write_text(json.dumps({"traceEvents": events}))


def test_timeline_overlap_uses_only_common_pipe_capture(tmp_path: Path):
    common = tmp_path / "pipe.json"
    cube = tmp_path / "cube.json"
    trace(
        common,
        [
            {
                "ph": "X",
                "ts": 0,
                "dur": 10,
                "name": "MMAD",
                "cat": "cube",
                "pid": 0,
                "tid": 0,
            },
            {
                "ph": "X",
                "ts": 5,
                "dur": 10,
                "name": "VF",
                "cat": "vector",
                "pid": 0,
                "tid": 1,
            },
        ],
    )
    trace(
        cube,
        [{"ph": "X", "ts": 100, "dur": 3, "name": "MMAD", "args": {"pc_addr": "0x10"}}],
    )
    output = tmp_path / "out"
    subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/summarize_timelines.py"),
            "--pipe-timeline",
            str(common),
            "--instr",
            f"cube={cube}",
            "--output",
            str(output),
        ],
        check=True,
    )
    result = json.loads((output / "timeline-summary.json").read_text())
    assert result["pipe_timeline"]["overlap_duration"]["cube:vector"] == 5
    assert result["instruction_timelines"]["simultaneous"] is False
    assert result["instruction_timelines"]["events"] == 1
    assert (output / "timeline-events.csv.gz").read_bytes()[4:8] == b"\0\0\0\0"


def test_timeline_overlap_merges_same_pipe_multi_core_intervals(tmp_path: Path):
    path = tmp_path / "pipe.json"
    trace(
        path,
        [
            {"ph": "X", "ts": 0, "dur": 10, "cat": "cube", "pid": 0},
            {"ph": "X", "ts": 0, "dur": 10, "cat": "cube", "pid": 1},
            {"ph": "X", "ts": 0, "dur": 10, "cat": "vector", "pid": 2},
        ],
    )
    module = load("summarize_timelines")
    assert module.overlaps(module.events(path, "pipe-timeline"))["cube:vector"] == 10


def test_timeline_rejects_instruction_trace_at_record_cap(tmp_path: Path):
    path = tmp_path / "cube.json"
    trace(
        path,
        [
            {"ph": "X", "ts": index, "dur": 1, "name": "MMAD"}
            for index in range(1024)
        ],
    )
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/summarize_timelines.py"),
            "--instr",
            f"cube={path}",
            "--output",
            str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "1024-event cap" in result.stderr


def test_scalar_only_pipe_timeline_is_rejected(tmp_path: Path):
    path = tmp_path / "trace.json"
    trace(path, [{"ph": "X", "ts": 0, "dur": 1, "name": "WAIT", "cat": "scalar"}])
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/summarize_timelines.py"),
            "--pipe-timeline",
            str(path),
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert result.returncode != 0


def test_curator_filters_kernel_and_is_deterministic(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    with (source / "OpBasicInfo.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["Op Name", "Task Duration(us)"])
        writer.writeheader()
        writer.writerows(
            [
                {"Op Name": "wanted", "Task Duration(us)": "12"},
                {"Op Name": "other", "Task Duration(us)": "99"},
            ]
        )
    (source / "sample-pmu-summary.json").write_text("{}\n")
    module = load("curate_profile")
    first = module.curate(source, tmp_path / "one", {"wanted"}, "analysis")
    second = module.curate(source, tmp_path / "two", {"wanted"}, "analysis")
    assert first.read_bytes() == second.read_bytes()
    with tarfile.open(first, "r:gz") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        metric_name = manifest["metric_exports"][0]["artifact"]
        metric = archive.extractfile(metric_name).read()
        rows = list(csv.DictReader(gzip.open(io.BytesIO(metric), "rt")))
    assert [row["Op Name"] for row in rows] == ["wanted"]
    assert metric[4:8] == b"\0\0\0\0"
    assert (source / "OpBasicInfo.csv").exists()


def test_curator_omits_unscoped_csv_when_filtering_kernel(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "unscoped.csv").write_text("Metric,Value\ncycles,12\n")
    (source / "scoped.csv").write_text("kernel_name,Value\nwanted,12\nother,99\n")
    module = load("curate_profile")
    archive = module.curate(source, tmp_path / "out", {"wanted"}, "summary")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        assert [item["source"] for item in manifest["metric_exports"]] == ["scoped.csv"]
        metric = bundle.extractfile(manifest["metric_exports"][0]["artifact"])
        rows = list(csv.DictReader(gzip.open(metric, "rt")))
    assert rows == [{"kernel_name": "wanted", "Value": "12"}]


def test_curator_omits_derived_artifacts_when_filtering_kernel(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "metrics.csv").write_text("kernel_name,Value\nwanted,12\n")
    (source / "sample-pmu-summary.json").write_text("{}\n")
    (source / "sample-pmu-rows.csv.gz").write_bytes(b"unscoped")
    module = load("curate_profile")
    archive = module.curate(source, tmp_path / "out", {"wanted"}, "analysis")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        names = bundle.getnames()
    assert manifest["derived_artifacts"] == []
    assert not any(name.startswith("derived/") for name in names)


def test_curator_rejects_output_inside_input(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "metrics.csv").write_text("Op Name,Value\nwanted,12\n")
    output = source / "compact"
    module = load("curate_profile")
    try:
        module.curate(source, output, set(), "summary")
    except module.InvalidEvidence as error:
        assert "output must not be inside input" in str(error)
    else:
        raise AssertionError("nested curation output unexpectedly succeeded")
    assert not output.exists()


def test_deterministic_archive_normalizes_file_modes(tmp_path: Path):
    source = tmp_path / "profile"
    source.mkdir()
    evidence = source / "evidence.txt"
    evidence.write_text("stable\n")
    module = load("curate_profile")
    evidence.chmod(0o600)
    first = tmp_path / "one.tar.gz"
    module.deterministic_archive(source, first)
    evidence.chmod(0o644)
    second = tmp_path / "two.tar.gz"
    module.deterministic_archive(source, second)
    assert first.read_bytes() == second.read_bytes()


def test_curator_preserves_metric_paths_without_collisions(tmp_path: Path):
    source = tmp_path / "raw"
    nested = source / "a"
    nested.mkdir(parents=True)
    (nested / "b.csv").write_text("Op Name,Value\nfirst,1\n")
    (source / "a__b.csv").write_text("Op Name,Value\nsecond,2\n")
    module = load("curate_profile")
    archive = module.curate(source, tmp_path / "out", set(), "summary")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        exports = {item["source"]: item["artifact"] for item in manifest["metric_exports"]}
        contents = {
            source_name: list(
                csv.DictReader(gzip.open(bundle.extractfile(artifact), "rt"))
            )
            for source_name, artifact in exports.items()
        }
    assert exports == {
        "a/b.csv": "metrics/a/b.csv.gz",
        "a__b.csv": "metrics/a__b.csv.gz",
    }
    assert contents["a/b.csv"] == [{"Op Name": "first", "Value": "1"}]
    assert contents["a__b.csv"] == [{"Op Name": "second", "Value": "2"}]


def test_curator_preserves_derived_paths_without_collisions(tmp_path: Path):
    source = tmp_path / "raw"
    nested = source / "a" / "b"
    nested.mkdir(parents=True)
    (nested / "timeline-summary.json").write_text('{"capture": "nested"}\n')
    flat = source / "a__b"
    flat.mkdir()
    (flat / "timeline-summary.json").write_text('{"capture": "flat"}\n')
    module = load("curate_profile")
    archive = module.curate(source, tmp_path / "out", set(), "summary")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
        contents = {
            artifact: bundle.extractfile(artifact).read().decode()
            for artifact in manifest["derived_artifacts"]
        }
    assert contents == {
        "derived/a/b/timeline-summary.json": '{"capture": "nested"}\n',
        "derived/a__b/timeline-summary.json": '{"capture": "flat"}\n',
    }


def test_curator_does_not_follow_symlink(tmp_path: Path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "real.csv").write_text("Op Name,value\nk,1\n")
    (source / "link.csv").symlink_to(source / "real.csv")
    module = load("curate_profile")
    archive = module.curate(source, tmp_path / "out", {"k"}, "summary")
    with tarfile.open(archive, "r:gz") as bundle:
        manifest = json.load(bundle.extractfile("manifest.json"))
    assert manifest["ignored_symlinks"] == ["link.csv"]
    assert [item["path"] for item in manifest["source_inventory"]] == ["real.csv"]


def test_pipe_name_falls_back_to_vendor_track_identity(tmp_path: Path):
    path = tmp_path / "trace.json"
    trace(
        path,
        [{"ph": "X", "ts": 0, "dur": 1, "name": "CUBE", "pid": "core0", "tid": "CUBE"}],
    )
    module = load("summarize_timelines")
    assert module.events(path, "pipe-timeline")[0]["pipe"] == "cube"


def test_collection_dry_run_routes_by_implementation(tmp_path: Path):
    script = ROOT / "scripts/collect_profile.sh"
    common = [
        str(script),
        "--remote-root",
        "artifacts/raw",
        "--output",
        str(tmp_path),
        "--dry-run",
    ]
    env = {"TLA_ROOT": "/configured/tla"}
    ascendc = subprocess.run(
        common + ["--implementation", "ascendc"],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert "/configured/tla/execution-profiles/bz-a5/run.sh" in ascendc.stdout
    dsl = subprocess.run(
        common + ["--implementation", "dsl", "--catlass-src", "worktrees/catlass/x"],
        text=True,
        capture_output=True,
        env=env,
        check=True,
    )
    assert "catlass-validation.sh" in dsl.stdout
    assert "--profile bz-a5" in dsl.stdout


def test_a3_profile_runs_basic_info_and_emits_compact_evidence(tmp_path: Path):
    msprof = fake_msprof(
        tmp_path,
        """\
with (target / 'OpBasicInfo_1.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['Op Name', 'Task Duration(us)', 'Pid'])
    writer.writeheader()
    writer.writerows([
        {'Op Name': 'wanted', 'Task Duration(us)': '12.0', 'Pid': '98'},
        {'Op Name': 'wanted', 'Task Duration(us)': '8.0', 'Pid': '98'},
        {'Op Name': 'other', 'Task Duration(us)': '99.0', 'Pid': '98'},
    ])
""",
    )
    output = tmp_path / "capture"
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/profile_a3.py"),
            "--output",
            str(output),
            "--kernel-name",
            "wanted",
            "--warm-up",
            "3",
            "--launch-count",
            "2",
            "--msprof",
            str(msprof),
            "--",
            "python3",
            "case.py",
            "--case",
            "47",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "success"
    assert evidence["target_family"] == "Ascend-A2-A3"
    assert evidence["protocol"] == {
        "warm_up": 3,
        "launch_count": 2,
        "replay_mode": "kernel",
        "kernel_selector": "wanted",
    }
    assert evidence["kernels"] == [
        {
            "name": "wanted",
            "samples": 2,
            "duration_us": {"min": 8.0, "median": 10.0, "p90": 12.0, "max": 12.0},
            "sample_values_us": [12.0, 8.0],
        }
    ]
    assert evidence["application"] == ["python3", "case.py", "--case", "47"]
    assert json.loads((output / "evidence.json").read_text()) == evidence
    assert len(evidence["sources"][0]["sha256"]) == 64


def test_a3_profile_rejects_success_without_matching_kernel(tmp_path: Path):
    msprof = fake_msprof(
        tmp_path,
        """\
with (target / 'OpBasicInfo.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=['Op Name', 'Task Duration(us)'])
    writer.writeheader()
    writer.writerow({'Op Name': 'other', 'Task Duration(us)': '4.0'})
""",
    )
    output = tmp_path / "capture"
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/profile_a3.py"),
            "--output",
            str(output),
            "--kernel-name",
            "wanted",
            "--msprof",
            str(msprof),
            "--",
            "python3",
            "case.py",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 1
    evidence = json.loads(result.stdout)
    assert evidence["status"] == "failure"
    assert evidence["failure"]["kind"] == "profiling"
    assert "matching 'wanted'" in evidence["failure"]["message"]
    assert (output / "msprof.log").read_text().endswith(SUCCESS_LINE + "\n")


def test_a3_profile_preserves_nonzero_profiler_failure(tmp_path: Path):
    msprof = fake_msprof(tmp_path, "pass", returncode=17, success=False)
    output = tmp_path / "capture"
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/profile_a3.py"),
            "--output",
            str(output),
            "--msprof",
            str(msprof),
            "--",
            "python3",
            "broken.py",
        ],
        text=True,
        capture_output=True,
    )
    evidence = json.loads(result.stdout)
    assert result.returncode == 1
    assert evidence["failure"]["msprof_returncode"] == 17
    assert evidence["failure"]["message"] == "msprof exited with status 17"
    assert "profiler stopped" in (output / "msprof.log").read_text()


def test_a3_summary_reports_multiple_kernels_separately(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\nfirst,2.5\nsecond,7.5\n"
    )
    module = load("profile_a3")
    evidence = module.summarize(raw, None, warm_up=0, launch_count=1)
    assert [(item["name"], item["duration_us"]["median"]) for item in evidence["kernels"]] == [
        ("first", 2.5),
        ("second", 7.5),
    ]


def test_a3_profile_refuses_to_mix_with_existing_evidence(tmp_path: Path):
    output = tmp_path / "capture"
    output.mkdir()
    (output / "old.json").write_text("{}\n")
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "scripts/profile_a3.py"),
            "--output",
            str(output),
            "--",
            "python3",
            "case.py",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "output directory is not empty" in result.stderr
    assert (output / "old.json").read_text() == "{}\n"
