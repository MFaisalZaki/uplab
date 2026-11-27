"""
Not a `conftest.py`: the other test directories import theirs by that name.

A hand-made experiment: 3 variants x 2 domains x 5 tasks, request `sat`.

Covered runs as `cost / CPU seconds` (time limit 100 s):

    task        ref  A            B             C
    route/p1    2    2 / 1        5 / 10 (*)    timeout
    route/p2    4    4 / 2 (+)    7 / 4         7 / 50
    route/p3    2    error        2 / 20 (*)    timeout
    plain/q1    2    2 / 5        2 / 5         2 / 1
    plain/q2    2    2 / 8        error         2 / 3

(*) claims optimality; (+) finds a plan of cost 7 after 1.0 s and one of
cost 4 after 1.1 s. The request `set` has one task: A covers it, B does not.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_PATH = Path(__file__).parents[1] / "results" / "conftest.py"
_SPEC = importlib.util.spec_from_file_location("results_fixtures", _PATH)
assert _SPEC and _SPEC.loader
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

LONG, SHORT = base.LONG, base.SHORT
FILES = {"p3": "p1", "q2": "q1"}  # tasks reusing the problem file of another

REPORTS: List[Dict[str, Any]] = [
    {"name": name, "query": f"builtin:{name}", "render": "csv"}
    for name in ("wrong-claims", "pairwise-tests", "double-hits", "domain-scores")
]
REPORTS += [
    {"name": "portfolio", "query": "plugin:portfolio", "render": "csv"},
    {"name": "lab-scores", "query": "builtin:lab-scores", "render": "latex"},
    {"name": "anytime", "query": "builtin:anytime-quality", "render": "anytime"},
]
ANALYSES = ["optimality-claims", "pairwise-tests", "double-hits", "domain-scores"]
ANALYSES += ["portfolio", "lab-scores"]


def write_run(
    exp_dir: Path,
    task_id: str,
    variant: str,
    seed: int = 0,
    claimed_optimal: bool = False,
    peak_memory_mb: Optional[float] = None,
    **kwargs: Any,
) -> Path:
    """
    `write_run` of the results tests, plus what it does not let one set.
    Of the runs differing only in the seed, write the one of seed 0 last.
    """
    run_dir = base.write_run(exp_dir, task_id, variant, **kwargs)
    run = json.loads((run_dir / "run.json").read_text())
    result = json.loads((run_dir / "result.json").read_text())
    problem = Path(run["task"]["problem_file"])
    run["task"]["problem_file"] = str(
        problem.with_name(FILES.get(problem.stem, problem.stem) + ".pddl")
    )
    result["plans"]["claimed_optimal"] = claimed_optimal
    if peak_memory_mb is not None:
        result["resources"]["peak_memory_mb"] = peak_memory_mb
    if seed:
        key = f"{task_id}|{variant}|{run['request']['name']}|{seed}"
        run["seed"], run["run_id"] = seed, hashlib.sha1(key.encode()).hexdigest()[:12]
        result["run_id"] = run["run_id"]
        run_dir = run_dir.rename(run_dir.with_name(run["run_id"]))
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return run_dir


def write_config(exp_dir: Path, variants: List[str], **extra: Any) -> None:
    config = {
        "name": "demo",
        "benchmarks": [{"name": "local", "source": str(base.FIXTURES)}],
        "planners": [{"name": "pyperplan", "source": ["up-pyperplan"]}],
        "variants": [
            {"name": name, "planner": "pyperplan", "engine": "pyperplan"}
            for name in variants
        ],
        "requests": list(base.REQUESTS.values()),
        "limits": {"time": 100, "memory_mb": 2048, "cores": 1},
        "analyses": [{"name": name} for name in ANALYSES],
        **extra,
    }
    exp_dir.mkdir(parents=True)
    (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def write_experiment(exp_dir: Path) -> Path:
    """Writes the experiment of the module docstring."""
    write_config(exp_dir, ["A", "B", "C"], reports=REPORTS)
    p1, p2, p3 = (f"local/route/p{i}" for i in (1, 2, 3))
    q1, q2 = "local/plain/q1", "local/plain/q2"
    write_run(exp_dir, p1, "A", plans=[LONG], cpu=1.0, first=0.5, peak_memory_mb=1.0)
    write_run(
        exp_dir, p1, "B", plans=[SHORT], cpu=10.0, claimed_optimal=True,
        peak_memory_mb=2048.0,
    )  # fmt: skip
    write_run(exp_dir, p1, "C", status="TIMEOUT", cpu=100.0)
    write_run(exp_dir, p2, "A", plans=[SHORT, LONG], cpu=2.0, first=1.0)
    write_run(exp_dir, p2, "B", plans=[SHORT], cpu=4.0)
    write_run(exp_dir, p2, "C", plans=[SHORT], cpu=50.0)
    write_run(exp_dir, p3, "A", status="ERROR", cpu=0.25, error="boom")
    write_run(exp_dir, p3, "B", plans=[LONG], cpu=20.0, claimed_optimal=True)
    write_run(exp_dir, p3, "C", status="TIMEOUT", cpu=100.0)
    write_run(exp_dir, q1, "A", plans=[LONG], cpu=5.0)
    write_run(exp_dir, q1, "B", plans=[LONG], cpu=5.0)
    write_run(exp_dir, q1, "C", plans=[LONG], cpu=1.0)
    write_run(exp_dir, q2, "A", plans=[LONG], cpu=8.0)
    write_run(exp_dir, q2, "B", status="ERROR", cpu=0.25, error="boom")
    write_run(exp_dir, q2, "C", plans=[LONG], cpu=3.0)
    write_run(exp_dir, p1, "A", request="set", plans=[LONG, SHORT], cpu=0.75)
    write_run(exp_dir, p1, "B", request="set", status="TIMEOUT", cpu=100.0)
    return exp_dir
