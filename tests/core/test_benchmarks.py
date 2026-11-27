import hashlib
import subprocess
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml

from up_lab import benchmarks
from up_lab.config import BenchmarkConfig, UPLabError


def tasks_of(source: Path, exp_dir: Path, **options: Any) -> List[Dict[str, Any]]:
    benchmark = BenchmarkConfig(name="b", source=str(source), **options)
    return benchmarks.find_tasks(benchmark, benchmarks.fetch(benchmark, exp_dir))


def test_pddl_layout(bench: Path, tmp_path: Path) -> None:
    tasks = tasks_of(bench, tmp_path)
    assert [t["id"] for t in tasks] == [
        "b/gripper/p1",
        "b/gripper/p2",
        "b/gripper/p10",
        "b/htn/t1",
        "b/ipc/p01",
        "b/ipc/p02",
    ]
    by_id = {t["id"]: t for t in tasks}
    assert by_id["b/htn/t1"]["domain_file"] == str(bench / "htn" / "domain.hddl")
    assert by_id["b/ipc/p01"]["domain_file"] == str(bench / "ipc" / "p01-domain.pddl")
    assert by_id["b/ipc/p02"]["domain_file"] == str(bench / "ipc" / "domain_p02.pddl")
    task = by_id["b/gripper/p1"]
    problem = bench / "gripper" / "p1.pddl"
    assert task == {
        "id": "b/gripper/p1",
        "source": "b",
        "source_rev": None,
        "domain": "gripper",
        "problem": "p1",
        "format": "pddl",
        "domain_file": str(bench / "gripper" / "domain.pddl"),
        "problem_file": str(problem),
        "files_sha256": {
            "domain_file": hashlib.sha256(
                (bench / "gripper" / "domain.pddl").read_bytes()
            ).hexdigest(),
            "problem_file": hashlib.sha256(problem.read_bytes()).hexdigest(),
        },
        "factory": None,
        "factory_args": {},
        "python_path": None,
        "reference_cost": None,
        "tags": [],
    }


def test_selection(bench: Path, tmp_path: Path) -> None:
    ids = lambda **o: [t["id"] for t in tasks_of(bench, tmp_path, **o)]
    assert ids(include=["gripper/*"]) == [
        "b/gripper/p1",
        "b/gripper/p2",
        "b/gripper/p10",
    ]
    assert ids(include=["gripper/*", "htn/*"], exclude=["*/p1"]) == [
        "b/gripper/p2",
        "b/gripper/p10",
        "b/htn/t1",
    ]
    assert ids(max_per_domain=1) == ["b/gripper/p1", "b/htn/t1", "b/ipc/p01"]
    assert ids(subdir="gripper") == ["b/gripper/p1", "b/gripper/p2", "b/gripper/p10"]


def test_anml_layout_and_explicit_layout(bench: Path, tmp_path: Path) -> None:
    (bench / "temporal").mkdir()
    (bench / "temporal" / "match.anml").write_text("fluent boolean lit;\n")
    tasks = tasks_of(bench, tmp_path, layout="anml")
    assert len(tasks) == 1
    assert tasks[0]["id"] == "b/temporal/match" and tasks[0]["format"] == "anml"
    assert tasks[0]["domain_file"] is None
    assert list(tasks[0]["files_sha256"]) == ["problem_file"]
    assert len(tasks_of(bench, tmp_path, layout="auto")) == 6  # pddl matches first
    with pytest.raises(UPLabError, match="unknown layout"):
        tasks_of(bench, tmp_path, layout="nope")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(UPLabError, match="no tasks"):
        tasks_of(empty, tmp_path)


def test_manifest_overrides_layout(bench: Path, tmp_path: Path) -> None:
    manifest = {
        "tasks": [
            {
                "domain": "g",
                "problem": "one",
                "format": "pddl",
                "domain_file": "gripper/domain.pddl",
                "problem_file": "gripper/p1.pddl",
                "reference_cost": 11,
                "tags": ["ipc-1998"],
            },
            {
                "domain": "gen",
                "problem": "n3",
                "format": "python",
                "factory": "factories:make",
                "factory_args": {"n": 3},
            },
        ]
    }
    (bench / benchmarks.MANIFEST).write_text(yaml.safe_dump(manifest))
    one, generated = tasks_of(bench, tmp_path)
    assert (one["id"], one["reference_cost"], one["tags"]) == (
        "b/g/one",
        11,
        ["ipc-1998"],
    )
    assert one["problem_file"] == str(bench / "gripper" / "p1.pddl")
    assert generated["format"] == "python" and generated["python_path"] == str(bench)
    assert generated["factory"] == "factories:make"
    assert generated["factory_args"] == {"n": 3}
    assert generated["files_sha256"] == {} and generated["problem_file"] is None

    outside = tmp_path / "outside.yaml"
    outside.write_text(yaml.safe_dump({"tasks": manifest["tasks"][:1]}))
    assert len(tasks_of(bench, tmp_path, manifest=str(outside))) == 1
    with pytest.raises(UPLabError, match="invalid manifest"):
        tasks_of(bench, tmp_path, manifest=str(tmp_path / "missing.yaml"))


def git(repo: Path, *args: str) -> str:
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.org"]
    command = ["git", "-C", str(repo), *identity, *args]
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def test_pinned_revision_is_cloned(bench: Path, tmp_path: Path) -> None:
    git(bench, "init", "-q")
    git(bench, "add", ".")
    git(bench, "commit", "-q", "-m", "one")
    first = git(bench, "rev-parse", "HEAD").strip()
    (bench / "gripper" / "p3.pddl").write_text("(define (problem p3) (:domain d))\n")
    git(bench, "add", ".")
    git(bench, "commit", "-q", "-m", "two")
    second = git(bench, "rev-parse", "HEAD").strip()

    exp_dir = tmp_path / "exp"
    pinned = BenchmarkConfig(name="b", source=str(bench), rev=first)
    lock = benchmarks.fetch(pinned, exp_dir)
    assert lock["commit"] == first and lock["path"] == str(exp_dir / "benchmarks" / "b")
    tasks = benchmarks.find_tasks(pinned, lock)
    assert len(tasks) == 6 and tasks[0]["source_rev"] == first
    assert tasks[0]["problem_file"].startswith(lock["path"])

    moved = BenchmarkConfig(name="b", source=str(bench), rev=second)
    lock = benchmarks.fetch(moved, exp_dir)  # fetches into the existing clone
    assert lock["commit"] == second
    assert len(benchmarks.find_tasks(moved, lock)) == 7

    in_place = benchmarks.fetch(BenchmarkConfig(name="b", source=str(bench)), exp_dir)
    assert in_place == {
        "source": str(bench),
        "rev": None,
        "path": str(bench),
        "commit": second,
    }
