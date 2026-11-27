"""Fixtures shared by the core tests: a small benchmark and a built experiment."""

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from up_lab import benchmarks
from up_lab.config import ExperimentConfig, parse_config

DOMAIN = "(define (domain d) (:predicates (p)))\n"
PROBLEM = "(define (problem {name}) (:domain d) (:init) (:goal (p)))\n"


@pytest.fixture
def bench(tmp_path: Path) -> Path:
    """Three domains covering the built-in PDDL conventions."""
    root = tmp_path / "bench"
    for name in ("gripper/domain.pddl", "htn/domain.hddl", "ipc/p01-domain.pddl"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(DOMAIN)
    (root / "ipc" / "domain_p02.pddl").write_text(DOMAIN)
    problems = ("gripper/p1", "gripper/p2", "gripper/p10", "htn/t1.hddl")
    for name in (*problems, "ipc/p01", "ipc/p02"):
        path = root / (name if name.endswith(".hddl") else name + ".pddl")
        path.write_text(PROBLEM.format(name=path.stem))
    (root / "gripper" / "README.md").write_text("not a task\n")
    return root


def raw_config(bench: Path, **overrides: Any) -> Dict[str, Any]:
    """A valid configuration as it would be written in YAML."""
    data: Dict[str, Any] = {
        "name": "demo",
        "benchmarks": [{"name": "b", "source": str(bench)}],
        "planners": [{"name": "pp", "source": ["up-pyperplan"]}],
        "variants": [{"name": "v", "planner": "pp", "engine": "pyperplan"}],
        "requests": [{"name": "sat"}],
    }
    data.update(overrides)
    return data


@pytest.fixture
def make_exp(tmp_path: Path, bench: Path) -> Callable[..., Any]:
    """Returns ``make(**overrides) -> (exp_dir, config)`` with a ``lock.json``."""

    def make(**overrides: Any) -> Any:
        config: ExperimentConfig = parse_config(raw_config(bench, **overrides))
        exp_dir = tmp_path / "exp"
        exp_dir.mkdir(exist_ok=True)
        lock = {
            "benchmarks": {
                b.name: benchmarks.fetch(b, exp_dir) for b in config.benchmarks
            },
            "planners": {
                p.name: {
                    "python": sys.executable,
                    "source": p.source,
                    "freeze_sha256": "0" * 64,
                    "packages": {"up-pyperplan": "1.1.0"},
                }
                for p in config.planners
            },
        }
        (exp_dir / "lock.json").write_text(json.dumps(lock))
        return exp_dir, config

    return make
