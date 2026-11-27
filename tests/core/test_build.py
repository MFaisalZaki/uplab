import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest

from up_lab import build
from up_lab.config import UPLabError


def test_run_id_is_the_contract_hash() -> None:
    # sha1("classical/gripper/prob01|pyperplan|sat|0")[:12]
    assert build.run_id("classical/gripper/prob01", "pyperplan", "sat", 0) == (
        "ed0429899599"
    )


def test_substitute() -> None:
    values = {"k": 5, "quality_bound": 1.5, "seed": 7}
    params = {
        "k": "${k}",
        "bound": "${quality_bound}",
        "options": ["--seed=${seed}", {"n": "${seed}"}, 3],
        "plain": "text",
    }
    assert build.substitute(params, values) == {
        "k": 5,
        "bound": 1.5,
        "options": ["--seed=7", {"n": 7}, 3],
        "plain": "text",
    }
    assert build.substitute("${k}", {"k": None}) is None


def test_build_writes_the_matrix(make_exp: Callable[..., Any]) -> None:
    variant = {
        "name": "v",
        "planner": "pp",
        "pipeline": [
            {"engine": "up_grounder", "compilation_kind": "GROUNDING"},
            {"engine": "mod:Engine", "params": {"k": "${k}", "seed": "${seed}"}},
        ],
    }
    exp_dir, config = make_exp(
        variants=[variant],
        requests=[{"name": "top", "mode": "planset", "k": 3, "quality_bound": 1.2}],
        seeds=[0, 1],
        limits={"time": 10, "memory_mb": 512, "cores": 2},
    )
    ids = build.build(exp_dir, config)
    assert len(ids) == len(set(ids)) == 6 * 1 * 1 * 2
    index = build.read_index(exp_dir)
    assert [line["run_id"] for line in index] == ids
    assert index[1] == {
        "run_id": build.run_id("b/gripper/p1", "v", "top", 1),
        "task": "b/gripper/p1",
        "variant": "v",
        "request": "top",
        "seed": 1,
    }
    run_dir = exp_dir / "runs" / ids[1]
    run = json.loads((run_dir / "run.json").read_text())
    assert list(run) == [
        "schema",
        "run_id",
        "experiment",
        "seed",
        "task",
        "variant",
        "request",
        "limits",
        "planner_env",
    ]
    assert (run["schema"], run["run_id"], run["experiment"]) == (
        "uplab/run/1",
        ids[1],
        "demo",
    )
    assert run["task"]["id"] == "b/gripper/p1" and run["seed"] == 1
    assert run["variant"] == {
        "name": "v",
        "planner": "pp",
        "pipeline": [
            {"engine": "up_grounder", "params": {}, "compilation_kind": "GROUNDING"},
            {
                "engine": "mod:Engine",
                "params": {"k": 3, "seed": 1},
                "compilation_kind": None,
            },
        ],
        "compile_only": False,
    }
    assert run["request"] == {
        "name": "top",
        "mode": "planset",
        "k": 3,
        "quality_bound": 1.2,
        "diversity": None,
    }
    assert run["limits"] == {"time": 10, "memory_mb": 512, "cores": 2}
    assert set(run["planner_env"]) == {"python", "source", "freeze_sha256", "packages"}
    launcher = run_dir / "run.sh"
    assert os.access(launcher, os.X_OK)
    lines = launcher.read_text().splitlines()
    assert lines[0] == "#!/bin/bash"
    assert lines[1].startswith("exec ") and "-m up_lab_worker" in lines[1]
    assert lines[1].endswith('"$(dirname "$(readlink -f "$0")")"')


def test_rebuild_keeps_finished_runs(make_exp: Callable[..., Any]) -> None:
    exp_dir, config = make_exp()
    ids = build.build(exp_dir, config)
    done, todo = exp_dir / "runs" / ids[0], exp_dir / "runs" / ids[1]
    (done / "result.json").write_text("{}")
    (done / "run.sh").write_text("edited")
    before = (done / "run.json").read_bytes()
    assert build.pending(exp_dir) == ids[1:]

    exp_dir, changed = make_exp(limits={"time": 1}, seeds=[0, 5])
    again = build.build(exp_dir, changed)
    assert again[::2] == ids and len(again) == 12
    assert (done / "run.json").read_bytes() == before
    assert (done / "run.sh").read_text() == "edited"
    assert json.loads((todo / "run.json").read_text())["limits"]["time"] == 1
    assert len(build.read_index(exp_dir)) == 12


def test_build_needs_resolve(make_exp: Callable[..., Any], tmp_path: Path) -> None:
    exp_dir, config = make_exp()
    (exp_dir / "lock.json").write_text(json.dumps({"benchmarks": {}, "planners": {}}))
    with pytest.raises(UPLabError, match="uplab resolve"):
        build.build(exp_dir, config)
    (exp_dir / "lock.json").unlink()
    with pytest.raises(UPLabError, match="uplab resolve"):
        build.build(exp_dir, config)
    with pytest.raises(UPLabError, match="uplab build"):
        build.pending(tmp_path)
