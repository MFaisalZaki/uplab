"""End-to-end tests of the worker driver on hand-written run directories."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from unified_planning.engines import ValidationResultStatus
from unified_planning.exceptions import UPException
from unified_planning.io import PDDLReader
from unified_planning.plans import ActionInstance, SequentialPlan, TimeTriggeredPlan
from unified_planning.shortcuts import (
    DurativeAction,
    IntType,
    PlanValidator,
    Problem,
    get_environment,
)

from up_lab_worker import PlanSetGenerationResult, PlanSetPlannerMixin, __version__
from up_lab_worker.plans import canonical_plan, read_plan, write_plan

FIXTURES = Path(__file__).parent / "fixtures"
get_environment().credits_stream = None

RESULT_KEYS = {
    "schema": None,
    "run_id": None,
    "status": None,
    "up_status": None,
    "exit_code": None,
    "signal": None,
    "error": None,
    "started_at": None,
    "ended_at": None,
    "resources": ["wall_time", "cpu_time", "peak_memory_mb"],
    "phases": ["parse", "compile", "solve", "first_plan", "last_plan"],
    "problem": None,
    "plans": ["count", "claimed_optimal", "claimed_complete"],
    "engine": ["name", "metrics", "log_messages"],
    "machine": [
        "hostname",
        "cpu_model",
        "cpu_count",
        "memory_mb",
        "os",
        "python",
        "unified_planning",
        "up_lab_worker",
        "slurm_job_id",
        "slurm_array_job_id",
        "slurm_array_task_id",
    ],
}


def pddl_task() -> Dict[str, Any]:
    return {
        "id": "test/roads/two-routes",
        "source": "test",
        "source_rev": None,
        "domain": "roads",
        "problem": "two-routes",
        "format": "pddl",
        "domain_file": str(FIXTURES / "domain.pddl"),
        "problem_file": str(FIXTURES / "problem.pddl"),
        "files_sha256": {},
        "factory": None,
        "factory_args": {},
        "python_path": None,
        "reference_cost": None,
        "tags": [],
    }


def execute(
    run_dir: Path,
    pipeline: List[Dict[str, Any]],
    mode: str = "oneshot",
    k: Optional[int] = 1,
    time: float = 60,
    memory_mb: Optional[int] = 4096,
    task: Optional[Dict[str, Any]] = None,
    compile_only: bool = False,
    diversity: Any = None,
) -> Dict[str, Any]:
    """Writes `run.json`, runs the driver and returns the content of `result.json`."""
    steps = [{"params": {}, "compilation_kind": None, **step} for step in pipeline]
    run = {
        "schema": "uplab/run/1",
        "run_id": "0123456789ab",
        "experiment": "test",
        "seed": 0,
        "task": task or pddl_task(),
        "variant": {
            "name": "v",
            "planner": "p",
            "compile_only": compile_only,
            "pipeline": steps,
        },
        "request": {
            "name": "r",
            "mode": mode,
            "k": k,
            "quality_bound": None,
            "diversity": diversity,
        },
        "limits": {"time": time, "memory_mb": memory_mb, "cores": 1},
        "planner_env": {"python": sys.executable, "source": [], "packages": {}},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(run))
    path = os.pathsep.join([str(FIXTURES), os.environ.get("PYTHONPATH", "")])
    env = {**os.environ, "PYTHONPATH": path, "SLURM_JOB_ID": "42"}
    done = subprocess.run(
        [sys.executable, "-m", "up_lab_worker", str(run_dir)], env=env
    )
    assert done.returncode == 0
    assert not (run_dir / "result.json.tmp").exists()
    result = json.loads((run_dir / "result.json").read_text())
    assert set(result) == set(RESULT_KEYS)
    for key, subkeys in RESULT_KEYS.items():
        if subkeys is not None:
            assert set(result[key]) == set(subkeys), key
    assert result["schema"] == "uplab/result/1"
    assert result["run_id"] == "0123456789ab"
    assert result["machine"]["slurm_job_id"] == "42"
    assert result["machine"]["up_lab_worker"] == __version__
    assert result["resources"]["wall_time"] > 0
    return result


def plan_records(run_dir: Path) -> List[Dict[str, Any]]:
    lines = (run_dir / "plans.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def events(run_dir: Path) -> List[str]:
    lines = (run_dir / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["event"] for line in lines]


def valid_plans(run_dir: Path, problem: Optional[Problem] = None) -> List[Any]:
    """Reads every recorded plan back and validates it on the original problem."""
    if problem is None:
        problem = PDDLReader().parse_problem(
            str(FIXTURES / "domain.pddl"), str(FIXTURES / "problem.pddl")
        )
    plans = []
    for record in plan_records(run_dir):
        plan = read_plan(problem, run_dir / record["file"])
        assert record["length"] == len(plan.actions)
        with PlanValidator(problem_kind=problem.kind, plan_kind=plan.kind) as validator:
            assert (
                validator.validate(problem, plan).status == ValidationResultStatus.VALID
            )
        plans.append(plan)
    return plans


def test_oneshot_pyperplan(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "pyperplan"}])
    assert result["status"] == "SOLVED"
    assert result["up_status"] == "SOLVED_SATISFICING"
    assert (result["exit_code"], result["signal"], result["error"]) == (0, None, None)
    assert result["engine"]["name"] == "pyperplan"
    assert result["problem"]["kind"] == ["ACTION_BASED", "FLAT_TYPING"]
    assert result["problem"]["actions"] == 1 and result["problem"]["objects"] == 4
    assert result["plans"] == {
        "count": 1,
        "claimed_optimal": False,
        "claimed_complete": None,
    }
    for phase in ("parse", "solve", "first_plan", "last_plan"):
        assert set(result["phases"][phase]) == {"wall", "cpu"}
    assert result["phases"]["compile"] == []
    assert result["machine"]["unified_planning"]
    (record,) = plan_records(tmp_path)
    assert record["index"] == 1 and record["file"] == "plans/00001.plan"
    assert record["kind"] == "SEQUENTIAL_PLAN" and record["length"] == 2
    assert record["wall_time"] <= result["resources"]["wall_time"]
    text = (tmp_path / record["file"]).read_text()
    assert text in ("move(a, b1)\nmove(b1, c)\n", "move(a, b2)\nmove(b2, c)\n")
    assert len(valid_plans(tmp_path)) == 1
    assert events(tmp_path) == [
        "start",
        "parse_end",
        "solve_start",
        "plan",
        "solve_end",
    ]
    assert not (tmp_path / "stderr.log").read_text().count("Traceback")


def test_timeout(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "toy_engines:Sleeper"}], time=1)
    assert result["status"] == "TIMEOUT"
    assert result["signal"] == 9 and result["exit_code"] is None
    assert result["plans"]["count"] == 0
    assert result["phases"]["parse"] is not None and result["phases"]["solve"] is None
    assert result["problem"]["actions"] == 1
    assert result["engine"]["name"] == "toy_engines:Sleeper"
    assert result["resources"]["wall_time"] < 30
    assert events(tmp_path)[-1] == "limit"


def test_timeout_keeps_plans(tmp_path: Path) -> None:
    step = {"engine": "toy_engines:ToyAnytime", "params": {"sleep": 600}}
    result = execute(tmp_path, [step], mode="anytime", k=None, time=4)
    assert result["status"] == "TIMEOUT"
    assert result["plans"]["count"] == 2
    assert (
        result["phases"]["last_plan"]["wall"] >= result["phases"]["first_plan"]["wall"]
    )
    assert len(valid_plans(tmp_path)) == 2


def test_unsupported(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "toy_engines:Picky"}])
    assert result["status"] == "UNSUPPORTED"
    assert "toy_engines:Picky" in result["error"]
    assert result["plans"]["count"] == 0
    assert events(tmp_path)[-1] == "error"


def test_error(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "toy_engines:Crasher"}])
    assert result["status"] == "ERROR"
    assert "the toy engine crashed" in result["error"]
    assert "the toy engine crashed" in (tmp_path / "stderr.log").read_text()


def test_unknown_engine(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "no_such_module:Nothing"}])
    assert result["status"] == "ERROR" and result["error"]


def test_memout(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "toy_engines:Hog"}], memory_mb=1024)
    assert result["status"] == "MEMOUT"
    assert events(tmp_path)[-1] == "limit"


def test_missing_run_json(tmp_path: Path) -> None:
    done = subprocess.run([sys.executable, "-m", "up_lab_worker", str(tmp_path)])
    assert done.returncode == 0
    result = json.loads((tmp_path / "result.json").read_text())
    assert set(result) == set(RESULT_KEYS)
    assert result["status"] == "ERROR" and result["run_id"] is None


def test_planset_mixin(tmp_path: Path) -> None:
    step = {"engine": "toy_engines:ToyPlanSet"}
    result = execute(tmp_path, [step], mode="planset", k=2, diversity={"metric": "x"})
    assert result["status"] == "SOLVED"
    assert result["plans"] == {
        "count": 2,
        "claimed_optimal": False,
        "claimed_complete": False,
    }
    assert result["engine"]["metrics"] == {"diversity": "{'metric': 'x'}"}
    first, second = plan_records(tmp_path)
    assert second["wall_time"] - first["wall_time"] >= 0.2  # on_plan stamps early
    assert len(set(map(canonical_plan, valid_plans(tmp_path)))) == 2


def test_planset_anytime_adapter(tmp_path: Path) -> None:
    step = {"engine": "toy_engines:ToyAnytime"}
    result = execute(tmp_path, [step], mode="planset", k=2)
    assert result["status"] == "SOLVED"
    assert result["plans"]["count"] == 2  # the engine yields every plan twice
    assert len(set(map(canonical_plan, valid_plans(tmp_path)))) == 2
    assert events(tmp_path).count("plan") == 2


def test_anytime_stops_at_k(tmp_path: Path) -> None:
    step = {"engine": "toy_engines:ToyAnytime", "params": {"sleep": 600}}
    result = execute(tmp_path, [step], mode="anytime", k=1)
    assert result["status"] == "SOLVED" and result["up_status"] == "INTERMEDIATE"
    assert result["plans"]["count"] == 1


def test_planset_oneshot_fallback(tmp_path: Path) -> None:
    result = execute(tmp_path, [{"engine": "pyperplan"}], mode="planset", k=3)
    assert result["status"] == "SOLVED" and result["plans"]["count"] == 1


def test_compiler_pipeline(tmp_path: Path) -> None:
    pipeline = [
        {"engine": "up_grounder", "compilation_kind": "GROUNDING"},
        {"engine": "pyperplan"},
    ]
    result = execute(tmp_path, pipeline)
    assert result["status"] == "SOLVED"
    (phase,) = result["phases"]["compile"]
    assert phase["engine"] == "up_grounder"
    assert set(phase) == {"engine", "wall", "cpu", "output"}
    assert phase["output"]["actions"] > result["problem"]["actions"] == 1
    (plan,) = valid_plans(tmp_path)  # expressed with the lifted `move` action
    assert [str(a.action.name) for a in plan.actions] == ["move", "move"]
    assert "compile_end" in events(tmp_path)


def test_compile_only(tmp_path: Path) -> None:
    pipeline = [{"engine": "up_grounder", "compilation_kind": "GROUNDING"}]
    result = execute(tmp_path, pipeline, compile_only=True)
    assert result["status"] == "COMPILED" and result["up_status"] is None
    assert len(result["phases"]["compile"]) == 1 and result["phases"]["solve"] is None


def test_python_factory(tmp_path: Path) -> None:
    sys.path.insert(0, str(FIXTURES))
    from toy_problems import make_problem

    task = {
        **pddl_task(),
        "format": "python",
        "domain_file": None,
        "problem_file": None,
        "factory": "toy_problems:make_problem",
        "factory_args": {"steps": 3},
        "python_path": str(FIXTURES),
    }
    result = execute(tmp_path, [{"engine": "pyperplan"}], task=task)
    assert result["status"] == "SOLVED" and result["problem"]["actions"] == 3
    assert (
        tmp_path / "plans/00001.plan"
    ).read_text() == "step_1()\nstep_2()\nstep_3()\n"
    assert len(valid_plans(tmp_path, make_problem(3))) == 1


def test_rerun_starts_clean(tmp_path: Path) -> None:
    execute(tmp_path, [{"engine": "pyperplan"}])
    result = execute(tmp_path, [{"engine": "toy_engines:Picky"}])
    assert result["plans"]["count"] == 0
    assert not list((tmp_path / "plans").glob("*.plan"))


def test_time_triggered_round_trip(tmp_path: Path) -> None:
    problem = Problem("temporal")
    action = DurativeAction("work", amount=IntType(0, 5), flag=IntType(0, 1))
    action.set_fixed_duration(2)
    problem.add_action(action)
    instance = ActionInstance(action, (3, 1))
    plan = TimeTriggeredPlan([(0, instance, 2), (2.5, instance, None)])
    write_plan(plan, tmp_path / "t.plan")
    text = (tmp_path / "t.plan").read_text()
    assert text == "0.0: work(3, 1) [2.0]\n2.5: work(3, 1)\n"
    assert canonical_plan(read_plan(problem, tmp_path / "t.plan")) == canonical_plan(
        plan
    )
    assert read_plan(problem, tmp_path / "t.plan") == plan


def test_unreadable_plans(tmp_path: Path) -> None:
    problem = PDDLReader().parse_problem(
        str(FIXTURES / "domain.pddl"), str(FIXTURES / "problem.pddl")
    )
    for text in ("fly(a, c)\n", "move(a, nowhere)\n", "move(a)\n", "SequentialPlan:\n"):
        (tmp_path / "bad.plan").write_text(text)
        with pytest.raises(UPException):
            read_plan(problem, tmp_path / "bad.plan")
    with pytest.raises(UPException):
        read_plan(problem, tmp_path / "missing.plan")


def test_mixin_checks() -> None:
    sys.path.insert(0, str(FIXTURES))
    from toy_engines import ToyPlanSet

    problem = PDDLReader().parse_problem(
        str(FIXTURES / "domain.pddl"), str(FIXTURES / "problem.pddl")
    )
    assert ToyPlanSet.is_plan_set_planner()
    with ToyPlanSet() as engine:
        assert isinstance(engine, PlanSetPlannerMixin)
        result = engine.get_plan_set(problem, None)
        assert isinstance(result, PlanSetGenerationResult)
        assert len(result.plans) == 2 and result.complete
        with pytest.raises(UPException):
            engine.get_plan_set(problem, 0)
