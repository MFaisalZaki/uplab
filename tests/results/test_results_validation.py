import json
from pathlib import Path

import pytest

import up_lab.validation as validation
from up_lab.validation import load_problem, plan_cost, validate_run
from up_lab_worker.plans import read_plan

LONG, SHORT, BAD = "move(a, b)\nmove(b, c)\n", "move(a, c)\n", "move(b, c)\n"


def test_valid_plan_cost_from_action_costs(make_run, tmp_path: Path):
    run_dir = make_run(tmp_path, "local/route/p2", "good", plans=[LONG, SHORT])
    result = validate_run(run_dir)
    assert result == json.loads((run_dir / "validation.json").read_text())
    assert result["schema"] == "uplab/validation/1"
    assert result["run_id"] == run_dir.name
    assert result["validator"] == "sequential_plan_validator"
    # p2: dist(a,b)=2, dist(b,c)=2, dist(a,c)=7; no reference cost
    assert result["plans"] == [
        {"index": 1, "valid": True, "cost": 4, "length": 2, "duplicate_of": None, "within_bound": None},
        {"index": 2, "valid": True, "cost": 7, "length": 1, "duplicate_of": None, "within_bound": None},
    ]
    assert result["summary"] == {
        "valid": 2, "invalid": 0, "unvalidated": 0, "distinct_valid": 2, "best_cost": 4,
    }
    assert result["parsed"] == {}


def test_invalid_plan(make_run, tmp_path: Path):
    result = validate_run(make_run(tmp_path, "local/route/p1", "bad", plans=[BAD]))
    assert result["plans"][0]["valid"] is False
    assert result["plans"][0]["cost"] == 1  # still recomputed
    assert result["summary"] == {
        "valid": 0, "invalid": 1, "unvalidated": 0, "distinct_valid": 0, "best_cost": None,
    }


def test_duplicates_and_bound(make_run, tmp_path: Path):
    # p1: reference cost 2, quality bound 2.0 -> plans may cost up to 4
    run_dir = make_run(tmp_path, "local/route/p1", "good", "set", plans=[LONG, SHORT, LONG])
    result = validate_run(run_dir)
    assert [p["cost"] for p in result["plans"]] == [2, 5, 2]
    assert [p["duplicate_of"] for p in result["plans"]] == [None, None, 1]
    assert [p["within_bound"] for p in result["plans"]] == [True, False, True]
    assert result["summary"]["valid"] == 3
    assert result["summary"]["distinct_valid"] == 2
    assert result["summary"]["best_cost"] == 2


def test_cost_is_length_without_metric(make_run, tmp_path: Path):
    result = validate_run(make_run(tmp_path, "local/plain/q1", "good", plans=[LONG]))
    assert result["plans"][0] == {
        "index": 1, "valid": True, "cost": 2, "length": 2, "duplicate_of": None, "within_bound": None,
    }


def test_unreadable_plan_is_unvalidated(make_run, tmp_path: Path):
    result = validate_run(make_run(tmp_path, "local/route/p1", "good", plans=["fly(a)\n"]))
    assert result["plans"][0]["valid"] is None and result["plans"][0]["cost"] is None
    assert result["summary"]["unvalidated"] == 1 and result["validator"] is None


def test_no_plans(make_run, tmp_path: Path):
    result = validate_run(make_run(tmp_path, "local/route/p1", "good", status="TIMEOUT"))
    assert result["plans"] == [] and result["summary"]["best_cost"] is None


def test_log_parser_keyed_by_engine(make_run, tmp_path: Path, monkeypatch):
    parsers = {"pyperplan": lambda out: {"expanded": int(out.split()[1])}, "other": dict}
    monkeypatch.setattr(validation, "hook", lambda name: parsers)
    result = validate_run(make_run(tmp_path, "local/route/p1", "good", plans=[LONG]))
    assert result["parsed"] == {"expanded": 7}


def test_plan_cost_python_problem(make_run, tmp_path: Path):
    run_dir = make_run(tmp_path, "local/route/p1", "good", plans=[SHORT])
    task = json.loads((run_dir / "run.json").read_text())["task"]
    problem = load_problem(task)
    assert plan_cost(problem, read_plan(problem, run_dir / "plans/00001.plan")) == 5
