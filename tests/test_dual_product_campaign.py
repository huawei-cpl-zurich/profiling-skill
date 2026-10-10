from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dual_product_audited_campaign", ROOT / "scripts/audited_campaign.py"
)
campaign = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def _inputs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    prompt = root / "prompt.md"
    prompt.write_text("one invariant prompt\n")
    task_files = {}
    for product in campaign.PRODUCTS:
        task_files[product] = {}
        for task in campaign.TASKS:
            path = root / f"{product}-{task}.md"
            path.write_text(f"{product} independently ranked {task} cases\n")
            task_files[product][task] = path
    provenance = {
        "source_revision": "a" * 40,
        "controller_sha256": "b" * 64,
        "model": {"name": "gpt-5.6-sol", "reasoning_effort": "low"},
        "skills": {
            "cannbot": "c" * 64,
            "ascend-profiling": "d" * 64,
            "triton-guarded-kernel": "e" * 64,
        },
        "products": {},
    }
    for index, product in enumerate(campaign.PRODUCTS):
        provenance["products"][product] = {
            "runtime": campaign.PRODUCT_RUNTIMES[product],
            "runtime_image_digest": "sha256:" + str(index + 1) * 64,
            "baselines": {task: str(index + 3) * 64 for task in campaign.TASKS},
            "starters": {
                task: {
                    "candidate": {
                        "path": f"/frozen/{product}/{task}/candidate.py",
                        "sha256": str(index + 5) * 64,
                    },
                    "manifest": {
                        "path": f"/frozen/{product}/{task}/candidate.manifest.json",
                        "sha256": str(index + 7) * 64,
                    },
                }
                for task in campaign.TASKS
            },
        }
    return prompt, task_files, provenance


def _manifest(root: Path):
    prompt, tasks, provenance = _inputs(root)
    return campaign.build_manifest(
        "dual-product", prompt, {}, provenance, "seed", products=campaign.PRODUCTS,
        product_task_files=tasks, request_budget=24,
    )


def _terminal(cell: dict, slot: dict) -> dict:
    return {
        "status": "complete",
        "durable_handle": f"{slot['target']}:job:{cell['cell_id']}",
        "rounds_completed": 4,
        "rounds": [
            {"round": number, "status": "ok", "handle": f"round-{number}"}
            for number in range(1, 5)
        ],
        "commits": [f"commit-{number}" for number in range(1, 5)],
    }


def test_dual_product_manifest_pins_independent_inputs_and_runtime(tmp_path: Path):
    document = _manifest(tmp_path)

    assert document["schema_version"] == 4
    assert document["dimensions"]["products"] == ["a3", "a5"]
    assert len(document["cells"]) == 18
    assert len({cell["branch"] for cell in document["cells"]}) == 18
    assert {(cell["product"], cell["task"], cell["treatment"])
            for cell in document["cells"]} == {
        (product, task, treatment)
        for product in campaign.PRODUCTS
        for task in campaign.TASKS
        for treatment in campaign.TREATMENTS
    }
    for cell in document["cells"]:
        assert cell["runtime"] == campaign.PRODUCT_RUNTIMES[cell["product"]]
        expected = hashlib.sha256(
            f"{cell['product']} independently ranked {cell['task']} cases\n".encode()
        ).hexdigest()
        assert cell["task_sha256"] == expected
        assert cell["cell_id"].startswith(f"{cell['product']}-")
        assert "target" not in cell and "device" not in cell
    campaign.verify_manifest(document)


def test_generate_cli_accepts_product_specific_task_inputs(tmp_path: Path):
    prompt, tasks, provenance = _inputs(tmp_path)
    provenance_path = tmp_path / "provenance.json"
    provenance_path.write_text(json.dumps(provenance))
    output = tmp_path / "manifest.json"
    arguments = [
        "generate", "--run-id", "dual-cli", "--prompt", str(prompt),
        "--provenance", str(provenance_path), "--ordering-seed", "seed",
        "--request-budget", "24", "--output", str(output),
    ]
    for product in campaign.PRODUCTS:
        arguments += ["--product", product]
        for task in campaign.TASKS:
            arguments += [f"--{product}-{task}-task", str(tasks[product][task])]

    assert campaign.main(arguments) == 0
    document = json.loads(output.read_text())
    assert document["schema_version"] == 4
    assert document["dimensions"]["products"] == ["a3", "a5"]


def test_schema_v4_explicit_cannbot_treatment_uses_product_rules(tmp_path: Path):
    prompt, tasks, provenance = _inputs(tmp_path)
    document = campaign.build_manifest(
        "explicit-cannbot", prompt, {}, provenance, "seed",
        products=campaign.PRODUCTS, product_task_files=tasks,
        treatments=("cannbot",), request_budget=24,
    )

    assert document["schema_version"] == 4
    assert document["dimensions"]["treatments"] == ["cannbot"]
    assert len(document["cells"]) == 6
    campaign.verify_manifest(document)


def test_schema_v4_cli_simulation_uses_compatible_product_slots(tmp_path: Path):
    document = _manifest(tmp_path / "inputs")
    manifest_path = tmp_path / "manifest.json"
    ledger_path = tmp_path / "ledger.json"
    manifest_path.write_text(json.dumps(document))

    assert campaign.main([
        "simulate", "--manifest", str(manifest_path),
        "--ledger", str(ledger_path), "--slots", "4",
    ]) == 0
    ledger = json.loads(ledger_path.read_text())
    assert ledger["status"] == "complete"
    assert {state["attempts"][0].get("product")
            for state in ledger["cells"].values()} == {"a3", "a5"}


@pytest.mark.parametrize("field,value,message", [
    ("product", "a5", "cell identity"),
    ("runtime", "cann91", "runtime"),
])
def test_dual_product_manifest_rejects_identity_drift(
    tmp_path: Path, field: str, value: str, message: str,
):
    document = _manifest(tmp_path)
    cell = next(item for item in document["cells"] if item["product"] == "a3")
    cell[field] = value
    document["manifest_sha256"] = campaign._document_digest(document)

    with pytest.raises(campaign.CampaignError, match=message):
        campaign.verify_manifest(document)


def test_dual_product_manifest_rejects_cross_product_task_binding(tmp_path: Path):
    document = _manifest(tmp_path)
    cell = next(item for item in document["cells"] if item["product"] == "a3")
    cell["task_sha256"] = document["tasks"]["a5"][cell["task"]]["sha256"]
    document["manifest_sha256"] = campaign._document_digest(document)

    with pytest.raises(campaign.CampaignError, match="task or prompt binding"):
        campaign.verify_manifest(document)


@pytest.mark.parametrize("artifact,malformation", [
    ("prompt", "relative-path"),
    ("prompt", "invalid-sha"),
    ("task", "relative-path"),
    ("task", "invalid-sha"),
])
def test_schema_v4_rejects_malformed_artifact_bindings(
    tmp_path: Path, artifact: str, malformation: str,
):
    document = _manifest(tmp_path)
    binding = (document["prompt"] if artifact == "prompt"
               else document["tasks"]["a3"]["matmul"])
    if malformation == "relative-path":
        binding["path"] = "relative/input.md"
    else:
        binding["sha256"] = "not-a-sha256"
        if artifact == "prompt":
            for cell in document["cells"]:
                cell["prompt_contract"]["invariant_sha256"] = "not-a-sha256"
        else:
            for cell in document["cells"]:
                if cell["product"] == "a3" and cell["task"] == "matmul":
                    cell["task_sha256"] = "not-a-sha256"
                    cell["prompt_contract"]["task_sha256"] = "not-a-sha256"
    document["manifest_sha256"] = campaign._document_digest(document)

    with pytest.raises(campaign.CampaignError, match="artifact binding"):
        campaign.verify_manifest(document)


class Pool:
    def __init__(self, slots):
        self.slots = slots

    def admit(self):
        return self.slots


def test_scheduler_fills_mixed_products_and_refills_without_cross_placement(
    tmp_path: Path,
):
    document = _manifest(tmp_path)
    a3_started = threading.Event()
    a5_started = threading.Event()
    release = threading.Event()
    calls = []

    class Launcher:
        def launch(self, cell, slot):
            calls.append((cell["cell_id"], cell["product"], slot.copy()))
            (a3_started if cell["product"] == "a3" else a5_started).set()
            assert release.wait(5)
            return _terminal(cell, slot)

    pool = Pool([
        {"product": "a3", "runtime": "py311-torch", "target": "bz-a3-1",
         "device": 6, "healthy": True, "idle": True},
        {"product": "a5", "runtime": "cann91", "target": "bz-a5",
         "device": 2, "healthy": True, "idle": True},
    ])
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            campaign.run_campaign, document, tmp_path / "ledger.json", pool, Launcher()
        )
        assert a3_started.wait(5) and a5_started.wait(5)
        assert len(calls) == 2
        release.set()
        assert future.result(timeout=10)["status"] == "complete"

    assert len(calls) == 18
    assert all(
        (product == "a3" and slot["target"].startswith("bz-a3-"))
        or (product == "a5" and slot["target"] == "bz-a5")
        for _cell_id, product, slot in calls
    )


def test_scheduler_does_not_let_unavailable_product_block_runnable_cells(tmp_path: Path):
    document = _manifest(tmp_path)
    only_a5 = Pool([
        {"product": "a5", "runtime": "cann91", "target": "bz-a5",
         "device": 3, "healthy": True, "idle": True},
    ])
    calls = []

    class Launcher:
        def launch(self, cell, slot):
            calls.append(cell["product"])
            return _terminal(cell, slot)

    with pytest.raises(campaign.CampaignPaused):
        campaign.run_campaign(
            document, tmp_path / "ledger.json", only_a5, Launcher()
        )
    assert calls == ["a5"] * 9
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert all(
        state["status"] == ("complete" if cell["product"] == "a5" else "queued")
        for cell in document["cells"]
        for state in [ledger["cells"][cell["cell_id"]]]
    )


def test_dual_product_resume_observes_exact_handle_on_original_adapter(tmp_path: Path):
    document = _manifest(tmp_path)
    target = next(cell for cell in document["cells"] if cell["product"] == "a5")
    retained = "remote:bz-a5:job:retained-123"
    observed = []

    class Launcher:
        def __init__(self):
            self.interrupted = False

        def launch(self, cell, slot):
            if cell["cell_id"] == target["cell_id"] and not self.interrupted:
                self.interrupted = True
                raise campaign.InfrastructureFailure("observer lost", retained)
            return _terminal(cell, slot)

        def observe(self, cell, slot, durable_handle):
            observed.append((cell["cell_id"], slot.copy(), durable_handle))
            return _terminal(cell, slot) | {"durable_handle": durable_handle}

    pool = Pool([
        {"product": "a3", "runtime": "py311-torch", "target": "bz-a3-1",
         "device": 7, "healthy": True, "idle": True},
        {"product": "a5", "runtime": "cann91", "target": "bz-a5",
         "device": 4, "healthy": True, "idle": True},
    ])
    launcher = Launcher()
    ledger_path = tmp_path / "resume-ledger.json"
    with pytest.raises(campaign.CampaignPaused):
        campaign.run_campaign(document, ledger_path, pool, launcher)
    result = campaign.run_campaign(document, ledger_path, pool, launcher, resume=True)

    assert result["status"] == "complete"
    assert observed == [(
        target["cell_id"],
        {"product": "a5", "runtime": "cann91", "target": "bz-a5", "device": 4},
        retained,
    )]
