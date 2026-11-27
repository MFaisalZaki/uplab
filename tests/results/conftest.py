"""A hand-made experiment directory following docs/architecture.md."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"
LONG, SHORT, BAD = "move(a, b)\nmove(b, c)\n", "move(a, c)\n", "move(b, c)\n"
REQUESTS = {
    "sat": {
        "name": "sat",
        "mode": "oneshot",
        "k": 1,
        "quality_bound": None,
        "diversity": None,
    },
    "set": {
        "name": "set",
        "mode": "planset",
        "k": 3,
        "quality_bound": 2.0,
        "diversity": None,
    },
}
# task id -> manifest reference cost
TASKS = {"local/route/p1": 2, "local/route/p2": None, "local/plain/q1": None}

FAST_RQ = """PREFIX uplab: <urn:uplab:ns#>
CONSTRUCT { ?run uplab:fast true } WHERE { ?run uplab:cpuTime ?t FILTER(?t < 1) }
"""


def write_run(
    exp_dir: Path,
    task_id: str,
    variant: str,
    request: str = "sat",
    status: str = "SOLVED",
    plans: Optional[List[str]] = None,
    cpu: float = 1.0,
    first: Optional[float] = None,
    error: Optional[str] = None,
    metrics: Optional[Dict[str, str]] = None,
) -> Path:
    """Writes run.json, result.json, plans.jsonl, plans/ and logs of one run."""
    source, domain, problem = task_id.split("/")
    plans = plans or []
    run_id = hashlib.sha1(f"{task_id}|{variant}|{request}|0".encode()).hexdigest()[:12]
    run_dir = exp_dir / "runs" / run_id
    (run_dir / "plans").mkdir(parents=True)
    run = {
        "schema": "uplab/run/1",
        "run_id": run_id,
        "experiment": "demo",
        "seed": 0,
        "task": {
            "id": task_id,
            "source": source,
            "source_rev": None,
            "domain": domain,
            "problem": problem,
            "format": "pddl",
            "domain_file": str(FIXTURES / domain / "domain.pddl"),
            "problem_file": str(FIXTURES / domain / f"{problem}.pddl"),
            "files_sha256": {"domain_file": "0", "problem_file": "0"},
            "factory": None,
            "factory_args": {},
            "python_path": None,
            "reference_cost": TASKS.get(task_id),
            "tags": ["tiny"],
        },
        "variant": {
            "name": variant,
            "planner": "pyperplan",
            "compile_only": False,
            "pipeline": [
                {"engine": "pyperplan", "params": {}, "compilation_kind": None}
            ],
        },
        "request": REQUESTS[request],
        "limits": {"time": 100, "memory_mb": 2048, "cores": 1},
        "planner_env": {
            "python": "/abs/envs/pyperplan/bin/python",
            "source": ["up-pyperplan"],
            "freeze_sha256": "abc",
            "packages": {"up-pyperplan": "1.1.0"},
        },
    }
    times = [(first or cpu) + 0.1 * i for i in range(len(plans))]
    found = {"wall": times[0], "cpu": times[0]} if plans else None
    last = {"wall": times[-1], "cpu": times[-1]} if plans else None
    result = {
        "schema": "uplab/result/1",
        "run_id": run_id,
        "status": status,
        "up_status": "SOLVED_SATISFICING" if status == "SOLVED" else None,
        "exit_code": 1 if status == "ERROR" else 0,
        "signal": None,
        "error": error,
        "started_at": "2026-11-27T10:00:00+00:00",
        "ended_at": "2026-11-27T10:00:02+00:00",
        "resources": {"wall_time": cpu + 0.25, "cpu_time": cpu, "peak_memory_mb": 91.5},
        "phases": {
            "parse": {"wall": 0.25, "cpu": 0.125},
            "compile": [],
            "solve": {"wall": cpu, "cpu": cpu},
            "first_plan": found,
            "last_plan": last,
        },
        "problem": {
            "kind": ["ACTION_BASED", "FLAT_TYPING"],
            "actions": 1,
            "fluents": 2,
            "objects": 3,
        },
        "plans": {
            "count": len(plans),
            "claimed_optimal": False,
            "claimed_complete": None,
        },
        "engine": {"name": "pyperplan", "metrics": metrics or {}, "log_messages": []},
        "machine": {
            "hostname": "n01",
            "cpu_model": "cpu",
            "cpu_count": 8,
            "memory_mb": 32000,
            "os": "Linux-6.1",
            "python": "3.12.3",
            "unified_planning": "1.3.0",
            "up_lab_worker": "0.1.0",
            "slurm_job_id": None,
            "slurm_array_job_id": None,
            "slurm_array_task_id": None,
        },
    }
    lines = []
    for index, (text, time) in enumerate(zip(plans, times), start=1):
        name = f"plans/{index:05d}.plan"
        (run_dir / name).write_text(text)
        entry = {
            "index": index,
            "file": name,
            "wall_time": time,
            "cpu_time": time,
            "claimed_cost": None,
            "length": len(text.splitlines()),
            "kind": "SEQUENTIAL_PLAN",
        }
        lines.append(json.dumps(entry))
    (run_dir / "run.json").write_text(json.dumps(run, indent=2))
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    (run_dir / "plans.jsonl").write_text("".join(line + "\n" for line in lines))
    (run_dir / "stdout.log").write_text("expanded 7 nodes\n")
    (run_dir / "stderr.log").write_text("")
    return run_dir


def write_experiment(
    exp_dir: Path, reports: Optional[List[Dict[str, Any]]] = None
) -> Path:
    """Writes experiment.yaml and the eight runs of the fixture experiment."""
    config = {
        "name": "demo",
        "benchmarks": [{"name": "local", "source": str(FIXTURES)}],
        "planners": [{"name": "pyperplan", "source": ["up-pyperplan"]}],
        "variants": [
            {
                "name": name,
                "planner": "pyperplan",
                "pipeline": [{"engine": "pyperplan"}],
            }
            for name in ("good", "bad")
        ],
        "requests": list(REQUESTS.values()),
        "seeds": [0],
        "limits": {"time": 100, "memory_mb": 2048, "cores": 1},
        "analyses": [{"name": "fast.rq", "options": {}}],
        "reports": reports or [],
    }
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "experiment.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (exp_dir / "fast.rq").write_text(FAST_RQ)
    p1, p2, q1 = TASKS
    write_run(
        exp_dir, p1, "good", plans=[LONG], cpu=0.5, first=0.4, metrics={"expanded": "7"}
    )
    write_run(exp_dir, p1, "bad", plans=[SHORT], cpu=12.0, first=10.0)
    write_run(
        exp_dir, p1, "good", "set", plans=[LONG, SHORT, LONG], cpu=0.75, first=0.2
    )
    write_run(exp_dir, p1, "bad", "set", status="TIMEOUT", cpu=100.0)
    write_run(exp_dir, p2, "good", plans=[LONG], cpu=12.0, first=10.0)
    write_run(exp_dir, p2, "bad", plans=[BAD], cpu=3.0)
    write_run(exp_dir, q1, "good", plans=[LONG], cpu=50.0, first=10.0)
    write_run(exp_dir, q1, "bad", status="ERROR", cpu=0.25, error="boom")
    return exp_dir


@pytest.fixture
def make_run() -> Callable[..., Path]:
    return write_run


@pytest.fixture(scope="session")
def exp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The fixture experiment, collected and analyzed once."""
    from up_lab.analysis import analyze
    from up_lab.collect import collect

    exp_dir = write_experiment(tmp_path_factory.mktemp("exp") / "demo")
    assert collect(exp_dir) == 8
    analyze(exp_dir)
    return exp_dir


@pytest.fixture
def exp_copy(exp: Path, tmp_path: Path) -> Path:
    """A private copy of the collected experiment, for tests that modify it."""
    return Path(shutil.copytree(exp, tmp_path / "demo"))
