"""Collection: validate runs, write their JSON-LD records, load the store."""

import json
import math
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote

import pyoxigraph as ox

from up_lab.store import experiment_config, graph_iri, open_store
from up_lab.validation import validate_run

CONTEXT = Path(__file__).parent / "vocab" / "context.jsonld"


def iri(kind: str, *parts: Any) -> str:
    """
    Returns the IRI of a resource.

    :param kind: The kind of resource, e.g. `run` or `task`.
    :param parts: The identifying parts; the first is the experiment name.
    :return: `urn:uplab:<kind>:<part>:...` with the parts percent-encoded.
    """
    return ":".join(["urn:uplab", kind] + [quote(str(p), safe="/") for p in parts])


def _clean(value: Any) -> Any:
    """Drops unknown (`None`) values, which JSON-LD would ignore anyway."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_clean(v) for v in value if v is not None]
    return value


def _metric_value(value: Any) -> Union[int, float, bool, str]:
    if isinstance(value, str):
        for number in (int, float):
            try:
                value = number(value)
                break
            except ValueError:
                continue
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (int, float, bool, str)):
        return value
    return json.dumps(value, sort_keys=True)


def outcome(status: str, summary: Dict[str, Any]) -> str:
    """
    Derives the outcome of a run from the worker status and the validation.

    :param status: The `status` of `result.json`.
    :param summary: The `summary` of `validation.json`.
    :return: The status, or `INVALID` for a `SOLVED` run with no valid plan
        and at least one invalid plan.
    """
    if status == "SOLVED" and not summary["valid"] and summary["invalid"]:
        return "INVALID"
    return status


def build_record(run_dir: Path, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merges the files of a validated run into one JSON-LD document.

    :param run_dir: The run directory.
    :param context: The JSON-LD context to inline.
    :return: The document, with one node per resource.
    """
    run = json.loads((run_dir / "run.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    validation = json.loads((run_dir / "validation.json").read_text())
    plans_file = run_dir / "plans.jsonl"
    lines = plans_file.read_text().splitlines() if plans_file.exists() else []
    found = {e["index"]: e for e in map(json.loads, filter(str.strip, lines))}
    exp, run_id = run["experiment"], run["run_id"]
    task, variant, request = run["task"], run["variant"], run["request"]
    env, limits = run.get("planner_env") or {}, run["limits"]
    phases, machine = result.get("phases") or {}, result.get("machine") or {}
    resources, problem = result.get("resources") or {}, result.get("problem") or {}
    engine, summary = result.get("engine") or {}, validation["summary"]
    compile_steps = phases.get("compile") or []

    def phase(name: str, clock: str) -> Optional[float]:
        return (phases.get(name) or {}).get(clock)

    def total(clock: str) -> Optional[float]:
        times = [step[clock] for step in compile_steps if step.get(clock) is not None]
        return sum(times) if times else None

    ids = {
        "experiment": iri("experiment", exp),
        "run": iri("run", exp, run_id),
        "task": iri("task", exp, task["id"]),
        "domain": iri("domain", exp, f"{task['source']}/{task['domain']}"),
        "source": iri("source", exp, task["source"]),
        "planner": iri("planner", exp, variant["planner"]),
        "variant": iri("variant", exp, variant["name"]),
        "request": iri("request", exp, request["name"]),
    }
    metrics = [("engine", k, v) for k, v in (engine.get("metrics") or {}).items()]
    metrics += [("parsed", k, v) for k, v in (validation.get("parsed") or {}).items()]
    run_node = {
        "@id": ids["run"],
        "@type": ["Run", "Activity"],
        "name": run_id,
        "experiment": ids["experiment"],
        "task": ids["task"],
        "variant": ids["variant"],
        "request": ids["request"],
        "used": ids["task"],
        "wasAssociatedWith": ids["variant"],
        "seed": run["seed"],
        "k": request["k"],
        "qualityBound": request["quality_bound"],
        "timeLimit": limits["time"],
        "memoryLimitMb": limits["memory_mb"],
        "cores": limits["cores"],
        "referenceCost": task["reference_cost"],
        "status": result["status"],
        "outcome": outcome(result["status"], summary),
        "upStatus": result.get("up_status"),
        "exitCode": result.get("exit_code"),
        "signal": result.get("signal"),
        "error": result.get("error"),
        "startedAt": result.get("started_at"),
        "endedAt": result.get("ended_at"),
        "wallTime": resources.get("wall_time"),
        "cpuTime": resources.get("cpu_time"),
        "peakMemoryMb": resources.get("peak_memory_mb"),
        "parseWallTime": phase("parse", "wall"),
        "parseCpuTime": phase("parse", "cpu"),
        "compileWallTime": total("wall"),
        "compileCpuTime": total("cpu"),
        "solveWallTime": phase("solve", "wall"),
        "solveCpuTime": phase("solve", "cpu"),
        "firstPlanWallTime": phase("first_plan", "wall"),
        "firstPlanCpuTime": phase("first_plan", "cpu"),
        "lastPlanWallTime": phase("last_plan", "wall"),
        "lastPlanCpuTime": phase("last_plan", "cpu"),
        "problemKind": problem.get("kind"),
        "problemActions": problem.get("actions"),
        "problemFluents": problem.get("fluents"),
        "problemObjects": problem.get("objects"),
        "plansReturned": (result.get("plans") or {}).get("count"),
        "claimedOptimal": (result.get("plans") or {}).get("claimed_optimal"),
        "claimedComplete": (result.get("plans") or {}).get("claimed_complete"),
        "engineName": engine.get("name"),
        "validator": validation.get("validator"),
        "plansValid": summary["valid"],
        "plansInvalid": summary["invalid"],
        "plansUnvalidated": summary["unvalidated"],
        "plansDistinctValid": summary["distinct_valid"],
        "bestCost": summary["best_cost"],
        "hostname": machine.get("hostname"),
        "cpuModel": machine.get("cpu_model"),
        "cpuCount": machine.get("cpu_count"),
        "machineMemoryMb": machine.get("memory_mb"),
        "os": machine.get("os"),
        "pythonVersion": machine.get("python"),
        "upVersion": machine.get("unified_planning"),
        "workerVersion": machine.get("up_lab_worker"),
        "slurmJobId": machine.get("slurm_job_id"),
        "slurmArrayJobId": machine.get("slurm_array_job_id"),
        "slurmArrayTaskId": machine.get("slurm_array_task_id"),
        "metric": [
            {
                "@id": iri("metric", exp, run_id, source, name),
                "@type": "Metric",
                "name": name,
                "metricSource": source,
                "value": _metric_value(value),
            }
            for source, name, value in metrics
        ],
    }
    nodes: List[Dict[str, Any]] = [
        run_node,
        {"@id": ids["experiment"], "@type": "Experiment", "name": exp},
        {
            "@id": ids["task"],
            "@type": "Task",
            "name": task["id"],
            "problem": task["problem"],
            "domain": ids["domain"],
            "format": task["format"],
            "domainFile": task.get("domain_file"),
            "problemFile": task.get("problem_file"),
            "factory": task.get("factory"),
            "referenceCost": task["reference_cost"],
            "tag": task.get("tags"),
        },
        {
            "@id": ids["domain"],
            "@type": "Domain",
            "name": task["domain"],
            "source": ids["source"],
        },
        {
            "@id": ids["source"],
            "@type": "BenchmarkSource",
            "name": task["source"],
            "revision": task.get("source_rev"),
        },
        {
            "@id": ids["planner"],
            "@type": "Planner",
            "name": variant["planner"],
            "requirement": env.get("source"),
            "environmentHash": env.get("freeze_sha256"),
        },
        {
            "@id": ids["variant"],
            "@type": ["Variant", "SoftwareAgent"],
            "name": variant["name"],
            "planner": ids["planner"],
            "compileOnly": variant.get("compile_only"),
            "engine": [step["engine"] for step in variant["pipeline"]],
            "pipeline": json.dumps(variant["pipeline"], sort_keys=True),
        },
        {
            "@id": ids["request"],
            "@type": "Request",
            "name": request["name"],
            "mode": request["mode"],
            "k": request["k"],
            "qualityBound": request["quality_bound"],
            "diversity": (
                None
                if request["diversity"] is None
                else json.dumps(request["diversity"], sort_keys=True)
            ),
        },
    ]
    for plan in validation["plans"]:
        entry, duplicate_of = found.get(plan["index"], {}), plan["duplicate_of"]
        nodes.append(
            {
                "@id": iri("plan", exp, run_id, plan["index"]),
                "@type": ["Plan", "Entity"],
                "wasGeneratedBy": ids["run"],
                "index": plan["index"],
                "file": entry.get("file"),
                "kind": entry.get("kind"),
                "foundWallTime": entry.get("wall_time"),
                "foundCpuTime": entry.get("cpu_time"),
                "claimedCost": entry.get("claimed_cost"),
                "cost": plan["cost"],
                "length": plan["length"],
                "valid": plan["valid"],
                "duplicate": duplicate_of is not None,
                "duplicateOf": duplicate_of and iri("plan", exp, run_id, duplicate_of),
                "withinBound": plan["within_bound"],
            }
        )
    return {"@context": context, "@graph": _clean(nodes)}


def load_records(exp_dir: Path, records: List[Path]) -> None:
    """
    Replaces the records graph of the store with the given records.

    The store is opened here and not in :func:`collect`: the PDDL parser keeps
    tracebacks alive, which would keep the frame of `collect`, hence the
    store and its write lock, alive after it returns.

    :param exp_dir: The experiment directory.
    :param records: The `record.jsonld` files to load.
    """
    graph = graph_iri(experiment_config(exp_dir)["name"])
    store = open_store(exp_dir)
    store.remove_graph(graph)
    for record in records:
        store.bulk_load(path=str(record), format=ox.RdfFormat.JSON_LD, to_graph=graph)
    store.flush()


def collect(exp_dir: Union[str, Path], jobs: int = 1) -> int:
    """
    Validates the done runs not validated yet, writes the `record.jsonld` of
    every done run and reloads the records graph of the store.

    :param exp_dir: The experiment directory.
    :param jobs: The number of validation processes.
    :return: The number of runs loaded into the store.
    """
    exp_dir = Path(exp_dir)
    done = sorted(p.parent for p in exp_dir.glob("runs/*/result.json"))
    pending = [d for d in done if not (d / "validation.json").exists()]
    if jobs > 1 and len(pending) > 1:
        with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as pool:
            list(pool.map(validate_run, pending, chunksize=4))
    else:
        for run_dir in pending:
            validate_run(run_dir)
    context = json.loads(CONTEXT.read_text())["@context"]
    for run_dir in done:
        record = build_record(run_dir, context)
        (run_dir / "record.jsonld").write_text(json.dumps(record, indent=1) + "\n")
    load_records(exp_dir, [run_dir / "record.jsonld" for run_dir in done])
    return len(done)
