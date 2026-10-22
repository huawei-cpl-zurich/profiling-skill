#!/usr/bin/env python3
"""Reproducible, hardware-free prompt-discovery smoke using BlindLauncher."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blind_optimization_launcher import BlindLauncher, LaunchError, execute, write_json

HERE = Path(__file__).resolve().parent
PROMPTS = HERE.parent / "prompts" / "profiling-discovery"
SUFFIX = """

PLANNING-ONLY CONTROL: The optimization request above is task context, not
authorization to implement or execute it in this smoke. Inspect the available
materials and propose your optimization workflow in /workspace/PLAN.md. Explain
your evidence strategy and the conditions for choosing tools. Do not implement
or execute kernels, submit workloads, benchmark, profile, or access remotes.
The execution.json deadline overrides the historical task budget. No code
changes or experiment commits are required. Finish with your proposed plan.
"""


def hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


class PlanningLauncher(BlindLauncher):
    def __init__(self, *args, coding, **kwargs):
        self.coding = coding.resolve()
        if not (self.coding / "SKILL.md").is_file() or any(
                p.is_symlink() for p in self.coding.rglob("*")):
            raise LaunchError("coding snapshot must be a directory without symlinks")
        super().__init__(*args, **kwargs)

    def prepare(self):
        super().prepare()
        shutil.copytree(self.coding, self.frozen / "coding")
        shutil.copy2(HERE / "planning_tool_client.py", self.frozen / "client.py")
        shutil.copy2(self.frozen / "prompt.md", self.frozen / "original-prompt.md")
        with (self.frozen / "prompt.md").open("a") as stream:
            stream.write(SUFFIX)
        write_json(self.root / "manifest.json", {
            "mode": "planning-only", "model": self.model, "timeout": self.timeout,
            "frozen": hashes(self.frozen), "public": {
                name: hashlib.sha256((self.workspace / name).read_bytes()).hexdigest()
                for name in ("API.md", "candidate.py")},
            "runner_dependencies": {
                name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                for name in ("profiling_planning_smoke.py", "blind_optimization_launcher.py",
                             "production_launcher.py", "planning_tool_client.py")},
        })

    def base_command(self):
        original = super().base_command()
        result = []
        index = 0
        while index < len(original):
            if original[index] == "--ro-bind":
                destination = original[index + 2]
                if destination in ("/experiment/production_launcher.py", "/workspace/PLAN.md"):
                    index += 3
                    continue
                if destination == "/experiment/launcher.py":
                    result += ["--ro-bind", str(self.frozen / "client.py"), destination]
                    index += 3
                    continue
            result.append(original[index])
            index += 1
        position = result.index("--chdir")
        mounts = ["--ro-bind", str(self.frozen / "coding"),
                  "/workspace/.agents/skills/triton-guarded-kernel"]
        # Candidate is task context, not writable implementation in this test.
        for item in self.workspace.glob("candidate.*"):
            mounts += ["--ro-bind", str(item), "/workspace/" + item.name]
        result[position:position] = mounts
        return result

    def preflight(self):
        checks = """from pathlib import Path
import importlib.util
assert importlib.util.find_spec('torch') is None
assert importlib.util.find_spec('triton') is None
assert not list(Path('/dev').glob('davinci*'))
assert not Path('/dev/davinci_manager').exists()
assert not Path('/experiment/production_launcher.py').exists()
assert not Path('/workspace/baseline.py').exists()
assert not Path('/workspace/oracle.py').exists()
assert not Path('/home/agent/.ssh').exists()
assert not Path('/codex-home/skills').exists()
assert not Path('/codex-home/plugins').exists()
assert Path('/workspace/.agents/skills/ascend-profiling/SKILL.md').is_file()
assert Path('/workspace/.agents/skills/triton-guarded-kernel/SKILL.md').is_file()
for p in Path('/workspace').glob('candidate.*'):
    try:
        p.write_text('tampered')
    except OSError:
        pass
    else:
        raise AssertionError('candidate writable')
Path('/workspace/PLAN.md').write_text('')
"""
        result = execute([*self.base_command(), "python3", "-c", checks], self.root, 30)
        write_json(self.root / "planning-isolation.json", result)
        if result["returncode"]:
            raise LaunchError("planning isolation failed: " + result["stderr"])
        return super().preflight()

    def launch(self, preflight_only=False):
        result = super().launch(preflight_only)
        plan = self.workspace / "PLAN.md"
        # A completed planning turn is not required to produce kernel commits.
        result["plan_present"] = plan.is_file() and bool(plan.read_text().strip())
        result["exit_code"] = (124 if result["timed_out"] else
                               max(1, abs(result["returncode"])) if result["returncode"] else
                               0 if preflight_only or result["plan_present"] else 3)
        result["assessment"] = "pending human transcript review; no keyword scoring"
        write_json(self.root / "receipt.json", result)
        return result


def run(args):
    versions = ("previous", "revised") if args.version == "both" else (args.version,)
    # A closed allowlist prevents accidental oracle/fixture export by a caller.
    files = json.loads((args.bundle / "task.json").read_text())["public_files"]
    if sorted(files) != ["API.md", "candidate.py"]:
        raise LaunchError("planning bundle must export only API.md and candidate.py")
    if args.output.exists():
        raise LaunchError("output exists; inspect receipts instead of restarting")
    model = json.loads(args.model.read_text())
    if not isinstance(model, dict) or not all(
            isinstance(model.get(key), str) and model[key].strip()
            for key in ("name", "reasoning_effort")):
        raise LaunchError("explicit model name and reasoning_effort are required")
    for name in files:
        source = args.bundle / name
        if source.is_symlink() or not source.is_file():
            raise LaunchError("public bundle file is unavailable: " + name)
    for name, source in (("coding", args.coding), ("profiling", args.profiling)):
        if not (source / "SKILL.md").is_file() or any(
                p.is_symlink() for p in source.rglob("*")):
            raise LaunchError(name + " snapshot must be available without symlinks")
    for version in versions:
        if not (PROMPTS / f"{version}.md").is_file():
            raise LaunchError("prompt version is unavailable: " + version)
    # Bad campaign-wide inputs must not reserve an unusable output directory.
    args.output.mkdir(parents=True)
    # Source paths never enter the agent-visible bundle.
    write_json(args.output / "campaign.json", {
        "mode": "planning-only", "model": model, "versions": versions,
        "replicates": args.replicates, "budget_seconds": args.timeout,
        "network": "model transport shared; no remote credentials/backend mounted",
        "assessment": "manual transcript review; skill read, distinction, conditional evidence plan",
    })

    def cell(item):
        version, number = item
        name = f"{version}-{number}"
        output = args.output / name
        try:
            launcher = PlanningLauncher(
                output, args.bundle, PROMPTS / f"{version}.md",
                model, [sys.executable, str(HERE / "planning_tool_client.py"), "serve"],
                coding=args.coding, skill=args.profiling, timeout=args.timeout,
                finish_reserve_seconds=0)
            return {"cell": name, **launcher.launch(args.preflight_only)}
        except Exception as error:
            # Preserve failures from one cell without abandoning siblings.
            result = {"cell": name, "exit_code": 1, "error": str(error),
                      "mode": "planning-only", "plan_present": False}
            output.mkdir(parents=True, exist_ok=True)
            write_json(output / "receipt.json", result)
            return result

    cells = [(v, n) for v in versions for n in range(1, args.replicates + 1)]
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        results = list(pool.map(cell, cells))
    write_json(args.output / "results.json", results)
    return int(any(r["exit_code"] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "model", "profiling", "coding", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", choices=("previous", "revised", "both"), default="both")
    parser.add_argument("--replicates", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--parallel", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--timeout", type=int, choices=range(1, 301), default=300)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, KeyError, LaunchError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
