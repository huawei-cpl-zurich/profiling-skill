#!/usr/bin/env python3
"""Blinded, report-only optimization comprehension evaluations (no kernel execution)."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blind_optimization_launcher import execute, write_json
from production_launcher import LaunchError, ProductionLauncher

REPORT_FIELDS = {
    "decision": {"type": "string", "enum": ["apply", "avoid", "insufficient_evidence"]},
    **{key: {"type": "string"} for key in (
        "transformation", "guards", "benefit", "tradeoffs", "profiler_interpretation",
        "counterinterpretation", "falsifying_followup")},
    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
}
REPORT_SCHEMA = {"type": "object", "properties": REPORT_FIELDS,
                 "required": list(REPORT_FIELDS), "additionalProperties": False}
GRADE_SCHEMA = {"type": "object", "properties": {
    "verdict": {"type": "string", "enum": ["supported", "unsupported", "inconclusive"]},
    **{key: {"type": "string"} for key in (
        "decision_assessment", "transformation_assessment", "guards_assessment",
        "tradeoffs_assessment", "profile_assessment", "falsifier_assessment")}},
    "required": ["verdict", "decision_assessment", "transformation_assessment",
                 "guards_assessment", "tradeoffs_assessment", "profile_assessment",
                 "falsifier_assessment"], "additionalProperties": False}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_artifact(root, name):
    """Only explicit regular files under the artifact root may enter a prompt."""
    if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("artifact must be a relative file")
    path = root / name
    if any(part.is_symlink() for part in [path, *path.parents] if part != root.parent):
        raise ValueError("symlink artifact is forbidden")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact is unavailable or outside artifact root")
    return path.read_text()


def load_wave(path):
    """Freeze allowlisted contents before dispatch; private manifest is never mounted."""
    manifest = json.loads(path.read_text())
    model = manifest["model"]
    if set(model) != {"name", "reasoning_effort"} or not all(
            isinstance(v, str) and v for v in model.values()):
        raise ValueError("explicit model name and reasoning_effort required")
    families = manifest["families"]
    if len(families) != 12 or len({f["id"] for f in families}) != 12:
        raise ValueError("wave requires twelve distinct families")
    root = path.parent.resolve()
    skills = {name: read_artifact(root, source) for name, source in manifest["allowed_skills"].items()}
    if not skills or manifest.get("skill_verified") is not True:
        raise ValueError("verified skill snapshot required before evaluation")
    common = {name: read_artifact(root, source) for name, source in manifest.get("common_skills", {}).items()}
    if set(common) & set(skills):
        raise ValueError("common and treatment skill names must differ")
    cases = []
    for family in families:
        if len(family["heldouts"]) != 2:
            raise ValueError("each family requires two held-outs")
        for case in family["heldouts"]:
            public = case["public"]
            if set(public) != {"source", "profile", "contract"}:
                raise ValueError("public context must contain only source, profile, contract")
            context = {key: read_artifact(root, value) for key, value in public.items()}
            gold = json.loads(read_artifact(root, case["gold"]))
            if gold.get("basis") != "measured_pair" or gold.get("status") not in (
                    "conclusive", "inconclusive") or not gold.get("rubric"):
                raise ValueError("sealed measured-pair gold with semantic rubric required")
            # Retain the exact actual paired evidence for the separate blinded grader.
            evidence = json.loads(read_artifact(root, gold["pair_artifact"]))
            if not isinstance(evidence, dict) or not evidence:
                raise ValueError("paired evidence must be a nonempty object")
            gold["paired_evidence"] = evidence
            gold.pop("pair_artifact")
            cases.append({"family": family["id"], "case": case["id"],
                          "public": context, "gold": gold})
    identities = [(c["family"], c["case"]) for c in cases]
    if len(set(identities)) != 24:
        raise ValueError("held-out identities must be distinct")
    return {"model": model, "skills": skills, "common_skills": common,
            "cases": cases, "seed": manifest.get("seed", 0)}


def evaluation_prompt(case, role, skills):
    return (f"You are the {role} in a no-code optimization review. Return only the requested JSON. "
            "Use only supplied source, profile and mathematical contract. Do not write kernel code, "
            "run commands, compile, access devices, or request external information. Describe the "
            "transformation in prose, prerequisites/numeric guards, likely benefit and tradeoffs. "
            "Interpret the actual profiler evidence; do not invent throughput or roofline estimates. "
            "Give a counterinterpretation and a falsifying follow-up. Choose insufficient_evidence "
            "when the evidence does not justify a decision.\n" + json.dumps({
                "artifacts": case["public"], "allowed_guidance": skills}, sort_keys=True))


def grading_prompt(case, response):
    return ("Semantically assess this anonymous no-code review against the sealed measured pair. "
            "Judge reasoning and applicability, not matching terminology. Assess decision, proposed "
            "transformation, numerical/resource guards, benefits/tradeoffs, actual profile evidence "
            "and falsifier separately. An inconclusive pair cannot justify a confident benefit claim. "
            "Do not infer treatment, skill availability, or family labels. Return the requested JSON.\n"
            + json.dumps({"artifacts": case["public"], "review": response,
                          "sealed_ground_truth": case["gold"]}, sort_keys=True))


def validate_response(response, schema):
    if not isinstance(response, dict) or set(response) != set(schema["required"]):
        raise ValueError("response fields do not match schema")
    for key, rule in schema["properties"].items():
        value = response[key]
        if rule["type"] == "string" and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"invalid {key}")
        if "enum" in rule and value not in rule["enum"]:
            raise ValueError(f"invalid {key}")
        if rule["type"] == "number" and (type(value) not in (int, float) or not 0 <= value <= 1):
            raise ValueError(f"invalid {key}")
    return response


class ReportLauncher(ProductionLauncher):
    """Fresh filesystem roots expose embedded context and no experiment transport."""

    def command(self, session, model):
        command = [self.bwrap, "--die-with-parent", "--new-session", "--unshare-all", "--share-net",
                   "--clearenv", "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
        for source in ("/usr", "/bin", "/lib", "/lib64", "/etc"):
            if Path(source).exists():
                command += ["--ro-bind", source, source]
        command += self._resolver_mounts()
        command += ["--dir", "/home", "--dir", "/home/agent", "--dir", "/runtime",
                    "--ro-bind", str(self._runtime_mount()), "/runtime/node",
                    "--ro-bind", str(session / "input"), "/input",
                    "--bind", str(session / "output"), "/output",
                    "--bind", str(session / "state"), "/codex-home",
                    "--ro-bind", str(self.auth_home / "auth.json"), "/codex-home/auth.json",
                    "--chdir", "/input", "--setenv", "HOME", "/home/agent",
                    "--setenv", "CODEX_HOME", "/codex-home",
                    "--setenv", "PATH", "/runtime/node/bin:/usr/bin:/bin"]
        executable = Path("/runtime/node") / self.codex.relative_to(self._runtime_mount())
        command += [str(executable), "exec", "--json", "--ignore-user-config", "--ignore-rules",
                    "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only",
                    "--model", model["name"], "-c", f'model_reasoning_effort="{model["reasoning_effort"]}"',
                    "-c", 'web_search="disabled"', "--output-schema", "/input/schema.json",
                    "--output-last-message", "/output/response.json"]
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "browser_use",
                        "computer_use", "image_generation", "multi_agent", "hooks"):
            command += ["--disable", feature]
        # Feed embedded context through stdin; large kernels/skill snapshots exceed argv limits.
        prefix_length = command.index(str(executable))
        return [*command[:prefix_length], "sh", "-c",
                'exec "$@" < /input/prompt.txt', "report-only", *command[prefix_length:], "-"]

    def __call__(self, session, prompt, schema, model, timeout):
        session = session.resolve()
        for name in ("input", "output", "state"):
            (session / name).mkdir()
        (session / "state" / "auth.json").touch(mode=0o600)
        (session / "input" / "prompt.txt").write_text(prompt)
        write_json(session / "input" / "schema.json", schema)
        result = execute(self.command(session, model), session, timeout,
                         (session / "stdout.jsonl", session / "stderr.txt"))
        write_json(session / "process.json", {k: v for k, v in result.items() if k not in ("stdout", "stderr")})
        if result["returncode"] or result["timed_out"]:
            raise LaunchError("model invocation failed; raw diagnostics retained")
        # Report-only sessions must not silently become tool-using experiments.
        for line in result["stdout"].splitlines():
            event = json.loads(line)
            item_type = event.get("item", {}).get("type")
            if item_type and item_type not in ("agent_message", "reasoning"):
                raise LaunchError("report-only invocation attempted a tool")
        return json.loads((session / "output" / "response.json").read_text())


def run_wave(wave, output, invoke, *, timeout=600, workers=1):
    """Never overwrite/retry a dispatched session; preserve failures and initial grades."""
    if type(workers) is not int or workers <= 0:
        raise ValueError("workers must be a positive integer")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "frozen-private.json", wave)
    jobs = [(i, role, treatment) for i in range(24)
            for role in ("actor", "analyzer") for treatment in (False, True)]
    random.Random(wave["seed"]).shuffle(jobs)
    def review(job):
        index, (case_index, role, treatment) = job
        session = output / f"review-{index:03d}"
        session.mkdir()
        case = wave["cases"][case_index]
        receipt = {"session": session.name, "case_index": case_index, "role": role,
                   "with_skill": treatment, "artifact_digest": digest(case["public"]), "status": "dispatched"}
        write_json(session / "receipt.json", receipt)
        try:
            guidance = {**wave.get("common_skills", {}), **(wave["skills"] if treatment else {})}
            raw = invoke(session, evaluation_prompt(case, role, guidance), REPORT_SCHEMA,
                         wave["model"], timeout)
            write_json(session / "raw-response.json", raw)
            receipt["response"] = validate_response(raw, REPORT_SCHEMA)
            receipt["status"] = "completed"
        except Exception as error:
            receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(session / "receipt.json", receipt)
        return receipt

    def assess(job):
        index, receipt = job
        if receipt["status"] != "completed":
            return receipt
        session = output / f"grade-{index:03d}"
        session.mkdir()
        grade = {"status": "dispatched"}
        write_json(session / "receipt.json", grade)
        try:
            raw = invoke(session, grading_prompt(wave["cases"][receipt["case_index"]],
                         receipt["response"]), GRADE_SCHEMA, wave["model"], timeout)
            write_json(session / "raw-response.json", raw)
            grade["assessment"] = validate_response(raw, GRADE_SCHEMA)
            grade["status"] = "completed"
        except Exception as error:
            grade.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(session / "receipt.json", grade)
        receipt["initial_grade"] = grade
        return receipt

    # map retains seeded receipt identities regardless of completion order. Materializing
    # the complete review stage is the barrier: no worker can see gold before it.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        receipts = list(pool.map(review, enumerate(jobs)))
        receipts = list(pool.map(assess, enumerate(receipts)))
    result = {"reviews": receipts, "review_count": len(receipts),
              "failed_reviews": sum(r["status"] != "completed" for r in receipts),
              "failed_grades": sum(r.get("initial_grade", {}).get("status") == "failed" for r in receipts)}
    write_json(output / "results.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.timeout <= 0 or args.workers <= 0:
        parser.error("timeout and workers must be positive")
    wave = load_wave(args.manifest)
    result = run_wave(wave, args.output, ReportLauncher(), timeout=args.timeout, workers=args.workers)
    return int(bool(result["failed_reviews"] or result["failed_grades"]))


if __name__ == "__main__":
    raise SystemExit(main())
