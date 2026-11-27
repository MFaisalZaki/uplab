"""Validation of the plans of one run, outside the planner's process."""

import importlib
import json
import sys
import warnings
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Hashable, List, Optional, Tuple, Union

from unified_planning.engines import ValidationResultStatus
from unified_planning.exceptions import UPException
from unified_planning.io import ANMLReader, PDDLReader
from unified_planning.model import AbstractProblem, UPState
from unified_planning.model.walkers import StateEvaluator
from unified_planning.plans import ActionInstance, Plan, SequentialPlan
from unified_planning.plans import TimeTriggeredPlan
from unified_planning.shortcuts import PlanValidator, get_environment
from up_lab_worker.plans import canonical_plan, read_plan

from up_lab.plugins import hook

SCHEMA = "uplab/validation/1"


def load_problem(task: Dict[str, Any]) -> AbstractProblem:
    """
    Loads the original problem of a run.

    :param task: The `task` object of a `run.json`.
    :return: The problem.
    """
    if task["format"] == "python":
        if task.get("python_path") and task["python_path"] not in sys.path:
            sys.path.insert(0, task["python_path"])
        module, _, function = task["factory"].partition(":")
        factory = getattr(importlib.import_module(module), function)
        return factory(**(task.get("factory_args") or {}))
    if task["format"] == "anml":
        files = [task.get("domain_file"), task.get("problem_file")]
        return ANMLReader().parse_problem([f for f in files if f])
    return PDDLReader().parse_problem(task["domain_file"], task["problem_file"])


def _number(value: Union[int, float, Fraction]) -> Union[int, float]:
    return int(value) if value == int(value) else float(value)


def _actions(plan: Plan) -> Optional[List[ActionInstance]]:
    if isinstance(plan, SequentialPlan):
        return list(plan.actions)
    if isinstance(plan, TimeTriggeredPlan):
        return [action for _, action, _ in plan.timed_actions]
    return None


def plan_cost(problem: AbstractProblem, plan: Plan) -> Union[int, float, None]:
    """
    Recomputes the cost of a plan from the first quality metric of the problem.

    :param problem: The original problem.
    :param plan: A plan of the problem.
    :return: The sum of the action costs (evaluated in the initial state, so
        costs may depend on static fluents), the makespan, or the number of
        actions when the metric is plan length or absent; `None` for any
        other metric or plan kind.
    """
    actions = _actions(plan)
    metrics = getattr(problem, "quality_metrics", [])
    metric = metrics[0] if metrics else None
    if actions is None:
        return None
    if metric is None or metric.is_minimize_sequential_plan_length():
        return len(actions)
    if metric.is_minimize_makespan() and isinstance(plan, TimeTriggeredPlan):
        ends = [start + (dur or 0) for start, _, dur in plan.timed_actions]
        return _number(max(ends, default=Fraction(0)))
    if not metric.is_minimize_action_costs():
        return None
    evaluator = StateEvaluator(problem)
    state = UPState(problem.initial_values, problem)
    manager = problem.environment.expression_manager
    total = Fraction(0)
    for instance in actions:
        cost = metric.get_action_cost(instance.action)
        if cost is None:
            continue
        parameters = [manager.ParameterExp(p) for p in instance.action.parameters]
        cost = cost.substitute(dict(zip(parameters, instance.actual_parameters)))
        try:
            total += evaluator.evaluate(cost, state).constant_value()
        except UPException:  # the cost reads an undefined value
            return None
    return _number(total)


def _validate(problem: AbstractProblem, plan: Plan) -> Tuple[Optional[bool], Any]:
    # Benchmarks often leave numeric fluents undefined (e.g. distances of
    # unconnected places), which no UP validator declares to support: select
    # the validator without that feature. A plan that reads an undefined
    # value makes the validator fail and stays unvalidated.
    kind = problem.kind.clone()
    kind.unset_initial_state("UNDEFINED_INITIAL_NUMERIC")
    try:
        with PlanValidator(problem_kind=kind, plan_kind=plan.kind) as engine:
            engine.skip_checks = True
            result = engine.validate(problem, plan)
            return result.status == ValidationResultStatus.VALID, engine.name
    except Exception:  # no validator applies, or it failed: unvalidated
        return None, None


def _parsed(run: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    variant, log = run["variant"], run_dir / "stdout.log"
    if variant.get("compile_only") or not variant["pipeline"] or not log.exists():
        return {}
    parser = hook("uplab_log_parsers").get(variant["pipeline"][-1]["engine"])
    if parser is None:
        return {}
    return dict(parser(log.read_text(errors="replace")))


def validate_run(run_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Validates the plans of a done run and writes its `validation.json`.

    :param run_dir: The run directory.
    :return: The content of `validation.json`.
    """
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    plans_file = run_dir / "plans.jsonl"
    lines = plans_file.read_text().splitlines() if plans_file.exists() else []
    entries = [json.loads(line) for line in lines if line.strip()]
    get_environment().credits_stream = None
    problem = None
    if entries:
        try:
            problem = load_problem(run["task"])
        except Exception as error:  # the plans stay unvalidated
            warnings.warn(f"{run_dir.name}: cannot load the problem: {error}")
    bound, reference = run["request"]["quality_bound"], run["task"]["reference_cost"]
    validator = None
    first: Dict[Hashable, int] = {}
    plans = []
    for entry in entries:
        path = run_dir / entry["file"]
        plan = None
        if problem is not None:
            try:
                plan = read_plan(problem, path)
            except Exception:  # unreadable: unvalidated
                pass
        valid, cost, length = None, None, entry.get("length")
        key: Hashable = path.read_text() if path.exists() else None
        if problem is not None and plan is not None:
            valid, name = _validate(problem, plan)
            validator = name or validator
            cost = plan_cost(problem, plan)
            actions = _actions(plan)
            length = len(actions) if actions is not None else length
            key = canonical_plan(plan)
        duplicate_of = first.setdefault(key, entry["index"])
        within = None
        if cost is not None and bound is not None and reference is not None:
            within = cost <= bound * reference
        plans.append(
            {
                "index": entry["index"],
                "valid": valid,
                "cost": cost,
                "length": length,
                "duplicate_of": (
                    None if duplicate_of == entry["index"] else duplicate_of
                ),
                "within_bound": within,
            }
        )
    good = [p for p in plans if p["valid"]]
    costs = [p["cost"] for p in good if p["cost"] is not None]
    validation = {
        "schema": SCHEMA,
        "run_id": run["run_id"],
        "validator": validator,
        "plans": plans,
        "summary": {
            "valid": len(good),
            "invalid": sum(p["valid"] is False for p in plans),
            "unvalidated": sum(p["valid"] is None for p in plans),
            "distinct_valid": sum(p["duplicate_of"] is None for p in good),
            "best_cost": min(costs, default=None),
        },
        "parsed": _parsed(run, run_dir),
    }
    (run_dir / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    return validation
