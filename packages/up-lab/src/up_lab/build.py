"""Run matrix: tasks x variants x requests x seeds -> run directories."""

import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, Dict, Iterator, List

from up_lab.benchmarks import find_tasks
from up_lab.config import ExperimentConfig, UPLabError

RUN_SCHEMA = "uplab/run/1"
LOCK_SCHEMA = "uplab/lock/1"


def run_id(task_id: str, variant: str, request: str, seed: int) -> str:
    """
    :return: the first 12 hex digits of sha1 over
        ``task.id|variant.name|request.name|seed``.
    """
    key = f"{task_id}|{variant}|{request}|{seed}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def substitute(value: Any, values: Dict[str, Any]) -> Any:
    """
    Replaces ``${name}`` placeholders in (nested) parameters. A string that
    is exactly one placeholder becomes the typed value.

    :param value: a parameter value.
    :param values: placeholder names and their values.
    :return: the value with placeholders substituted.
    """
    if isinstance(value, dict):
        return {k: substitute(v, values) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute(v, values) for v in value]
    if isinstance(value, str):
        for name, replacement in values.items():
            if value == f"${{{name}}}":
                return replacement
            value = value.replace(f"${{{name}}}", str(replacement))
    return value


def read_lock(exp_dir: Path) -> Dict[str, Any]:
    """
    :param exp_dir: the experiment directory.
    :return: the content of ``lock.json``.
    """
    try:
        lock: Dict[str, Any] = json.loads((exp_dir / "lock.json").read_text())
    except OSError as error:
        raise UPLabError(f"no lock.json in {exp_dir}: run 'uplab resolve'") from error
    return lock


def read_index(exp_dir: Path) -> List[Dict[str, Any]]:
    """
    :param exp_dir: the experiment directory.
    :return: the lines of ``runs/index.jsonl``.
    """
    try:
        lines = (exp_dir / "runs" / "index.jsonl").read_text().splitlines()
    except OSError as error:
        raise UPLabError(f"no run index in {exp_dir}: run 'uplab build'") from error
    return [json.loads(line) for line in lines if line.strip()]


def pending(exp_dir: Path) -> List[str]:
    """
    :param exp_dir: the experiment directory.
    :return: ids of the runs without ``result.json``, in index order.
    """
    runs = exp_dir / "runs"
    return [
        line["run_id"]
        for line in read_index(exp_dir)
        if not (runs / line["run_id"] / "result.json").exists()
    ]


def expand(config: ExperimentConfig, lock: Dict[str, Any]) -> Iterator[Dict]:
    """
    :param config: the experiment.
    :param lock: the content of ``lock.json``.
    :return: the ``run.json`` content of every run of the matrix.
    """
    try:
        sources = [(b, lock["benchmarks"][b.name]) for b in config.benchmarks]
        envs = {p.name: lock["planners"][p.name] for p in config.planners}
    except KeyError as error:
        raise UPLabError(f"{error} is not in lock.json: run 'uplab resolve'") from error
    tasks = [task for b, entry in sources for task in find_tasks(b, entry)]
    env_keys = ("python", "source", "freeze_sha256", "packages")
    for task in tasks:
        for variant in config.variants:
            for request in config.requests:
                for seed in config.seeds:
                    values = {
                        "k": request.k,
                        "quality_bound": request.quality_bound,
                        "seed": seed,
                    }
                    described = variant.model_dump(mode="json")
                    for step in described["pipeline"]:
                        step["params"] = substitute(step["params"], values)
                    yield {
                        "schema": RUN_SCHEMA,
                        "run_id": run_id(task["id"], variant.name, request.name, seed),
                        "experiment": config.name,
                        "seed": seed,
                        "task": task,
                        "variant": described,
                        "request": request.model_dump(mode="json"),
                        "limits": config.limits.model_dump(mode="json"),
                        "planner_env": {
                            k: envs[variant.planner].get(k) for k in env_keys
                        },
                    }


def build(exp_dir: Path, config: ExperimentConfig) -> List[str]:
    """
    Writes ``runs/<run-id>/{run.json,run.sh}`` and ``runs/index.jsonl``.
    Runs that already have a result are left untouched.

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    :return: the run ids of the matrix.
    """
    runs = exp_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    index = []
    for run in expand(config, read_lock(exp_dir)):
        index.append(
            {
                "run_id": run["run_id"],
                "task": run["task"]["id"],
                "variant": run["variant"]["name"],
                "request": run["request"]["name"],
                "seed": run["seed"],
            }
        )
        run_dir = runs / run["run_id"]
        if (run_dir / "result.json").exists():
            continue
        run_dir.mkdir(exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
        python = shlex.quote(run["planner_env"]["python"])
        launcher = run_dir / "run.sh"
        launcher.write_text(
            "#!/bin/bash\n"
            f'exec {python} -m up_lab_worker "$(dirname "$(readlink -f "$0")")"\n'
        )
        launcher.chmod(0o755)
    ids = [line["run_id"] for line in index]
    if len(set(ids)) != len(ids):
        raise UPLabError("run id collision: task ids must be unique")
    (runs / "index.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in index)
    )
    return ids
