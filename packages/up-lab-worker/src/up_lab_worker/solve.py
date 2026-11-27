"""The solving process of a run: ``python -m up_lab_worker.solve <run-dir>``."""

import importlib
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import IO, Any, Callable, Dict, List, Optional, Set, Tuple

_START = float(os.environ.get("UP_LAB_START") or time.time())

import unified_planning as up
from unified_planning.engines import CompilationKind
from unified_planning.engines.mixins import (
    AnytimePlannerMixin,
    CompilerMixin,
    OneshotPlannerMixin,
)
from unified_planning.engines.results import CompilerResult
from unified_planning.engines.results import PlanGenerationResultStatus as Status
from unified_planning.exceptions import UPException
from unified_planning.plans import Plan, SequentialPlan, TimeTriggeredPlan

from up_lab_worker.mixins import PlanSetPlannerMixin
from up_lab_worker.plans import canonical_plan, write_plan

STATUS_OF_UP_STATUS = {
    Status.TIMEOUT: "TIMEOUT",
    Status.MEMOUT: "MEMOUT",
    Status.UNSUPPORTED_PROBLEM: "UNSUPPORTED",
    Status.INTERNAL_ERROR: "ERROR",
    Status.UNSOLVABLE_PROVEN: "UNSOLVABLE",
    Status.UNSOLVABLE_INCOMPLETELY: "UNSOLVABLE",
}


class UPLabWorkerError(UPException):
    """Raised when a run cannot be executed as described by its `run.json`."""


class UPLabUnsupportedError(UPLabWorkerError):
    """Raised when an engine of the pipeline does not support the problem."""


def _clock() -> Tuple[float, float]:
    """Returns wall and CPU seconds (children included) since the process started."""
    t = os.times()
    cpu = t.user + t.system + t.children_user + t.children_system
    return time.time() - _START, cpu


def _since(start: Tuple[float, float]) -> Dict[str, float]:
    wall, cpu = _clock()
    return {"wall": round(wall - start[0], 6), "cpu": round(cpu - start[1], 6)}


def load_problem(task: Dict[str, Any]) -> "up.model.AbstractProblem":
    """
    Loads the problem of a task.

    :param task: The `task` object of a `run.json`.
    :return: The parsed or constructed problem.
    """
    task_format = task["format"]
    if task_format == "pddl":
        from unified_planning.io import PDDLReader

        return PDDLReader().parse_problem(task["domain_file"], task["problem_file"])
    if task_format == "anml":
        from unified_planning.io import ANMLReader

        files = [f for f in (task.get("domain_file"), task["problem_file"]) if f]
        return ANMLReader().parse_problem(files)
    if task_format == "python":
        if task.get("python_path"):
            sys.path.insert(0, task["python_path"])
        module_name, _, function_name = task["factory"].partition(":")
        factory = getattr(importlib.import_module(module_name), function_name)
        return factory(**(task.get("factory_args") or {}))
    raise UPLabWorkerError(f"Unknown task format '{task_format}'")


def problem_size(problem: "up.model.AbstractProblem") -> Dict[str, Optional[int]]:
    """Returns the number of actions, fluents and objects, `None` where undefined."""
    size: Dict[str, Optional[int]] = {}
    for key, attribute in (
        ("actions", "actions"),
        ("fluents", "fluents"),
        ("objects", "all_objects"),
    ):
        value = getattr(problem, attribute, None)
        size[key] = None if value is None else len(value)
    return size


def engine_class(spec: str) -> type:
    """
    Returns the engine class of a pipeline step.

    :param spec: A UP engine name, or `"module:Class"`.
    :return: The engine class, taken from the UP `Factory`.
    """
    factory = up.environment.get_environment().factory
    if ":" in spec and spec not in factory.engines:
        module_name, _, class_name = spec.partition(":")
        factory.add_engine(spec, module_name, class_name)
    return factory.engine(spec)


def _check_support(cls: type, spec: str, problem: "up.model.AbstractProblem") -> None:
    if not cls.supports(problem.kind):  # type: ignore[attr-defined]
        raise UPLabUnsupportedError(f"Engine '{spec}' does not support the problem")


class Run:
    """The state of the solving process of one run directory."""

    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        (run_dir / "plans").mkdir(exist_ok=True)
        self._events: IO[str] = open(run_dir / "events.jsonl", "a", encoding="utf-8")
        self._plans: IO[str] = open(run_dir / "plans.jsonl", "a", encoding="utf-8")
        self._seen: Set[Any] = set()
        self._compiled: List[CompilerResult] = []
        self.count = 0
        self.out: Dict[str, Any] = {
            "status": None,
            "up_status": None,
            "error": None,
            "phases": {"parse": None, "compile": [], "solve": None},
            "problem": None,
            "plans": {"claimed_optimal": None, "claimed_complete": None},
            "engine": {"name": None, "metrics": {}, "log_messages": []},
        }

    def event(self, name: str, **data: Any) -> None:
        """Appends an event to `events.jsonl`."""
        record = {"t": round(_clock()[0], 6), "event": name, **data}
        self._events.write(json.dumps(record) + "\n")
        self._events.flush()

    def checkpoint(self) -> None:
        """Writes `solve.json` atomically; `status` is `None` until the end."""
        tmp = self.dir / "solve.json.tmp"
        tmp.write_text(json.dumps(self.out, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.dir / "solve.json")

    def add_plan(self, plan: Optional[Plan]) -> None:
        """Maps a plan back to the original problem and records it, unless seen."""
        if plan is None:
            return
        wall, cpu = _clock()
        for result in reversed(self._compiled):
            if result.plan_back_conversion is not None:
                plan = result.plan_back_conversion(plan)
            else:
                assert result.map_back_action_instance is not None
                plan = plan.replace_action_instances(result.map_back_action_instance)
        key = canonical_plan(plan)
        if key in self._seen:
            return
        self._seen.add(key)
        self.count += 1
        length: Optional[int] = None
        if isinstance(plan, SequentialPlan):
            length = len(plan.actions)
        elif isinstance(plan, TimeTriggeredPlan):
            length = len(plan.timed_actions)
        record = {
            "index": self.count,
            "file": f"plans/{self.count:05d}.plan",
            "wall_time": round(wall, 6),
            "cpu_time": round(cpu, 6),
            "claimed_cost": None,
            "length": length,
            "kind": plan.kind.name,
        }
        write_plan(plan, self.dir / record["file"])
        self._plans.write(json.dumps(record) + "\n")
        self._plans.flush()
        self.event("plan", index=self.count)

    def _timeout(self) -> Optional[float]:
        limit = (self.run.get("limits") or {}).get("time")
        return None if limit is None else max(limit - _clock()[0], 0.001)

    def _compile(
        self, step: Dict[str, Any], problem: "up.model.AbstractProblem"
    ) -> "up.model.AbstractProblem":
        start = _clock()
        cls = engine_class(step["engine"])
        if not issubclass(cls, CompilerMixin):
            raise UPLabWorkerError(f"Engine '{step['engine']}' is not a compiler")
        _check_support(cls, step["engine"], problem)
        kind_name = step.get("compilation_kind")
        kind = None if kind_name is None else CompilationKind[kind_name]
        if kind is not None and not cls.supports_compilation(kind):  # type: ignore[attr-defined]
            raise UPLabUnsupportedError(
                f"Engine '{step['engine']}' does not support {kind_name}"
            )
        with cls(**(step.get("params") or {})) as compiler:
            result = compiler.compile(problem, kind)
        if result.problem is None:
            raise UPLabWorkerError(f"Engine '{step['engine']}' returned no problem")
        self._compiled.append(result)
        name = step["engine"]  # the configured name, not `Engine.name`
        phase = {
            "engine": name,
            **_since(start),
            "output": problem_size(result.problem),
        }
        self.out["phases"]["compile"].append(phase)
        self.out["engine"]["name"] = name
        self.event("compile_end", engine=name)
        self.checkpoint()
        return result.problem

    def _report(self, result: Any) -> Status:
        self.out["engine"]["metrics"] = dict(result.metrics or {})
        self.out["engine"]["log_messages"] = [
            {"level": m.level.name, "message": m.message}
            for m in result.log_messages or []
        ]
        return result.status

    def _anytime(self, engine: Any, problem: "up.model.AbstractProblem") -> Status:
        k = self.run["request"].get("k")
        status = Status.UNSOLVABLE_INCOMPLETELY
        for result in engine.get_solutions(problem, timeout=self._timeout()):
            self.add_plan(result.plan)
            status = self._report(result)
            if k is not None and self.count >= k:
                break
        return status

    def _plan(self, step: Dict[str, Any], problem: "up.model.AbstractProblem") -> None:
        request = self.run["request"]
        mode = request["mode"]
        cls = engine_class(step["engine"])
        _check_support(cls, step["engine"], problem)
        start = _clock()
        self.event("solve_start")
        with cls(**(step.get("params") or {})) as engine:
            self.out["engine"]["name"] = step["engine"]
            self.checkpoint()
            if mode == "planset" and isinstance(engine, PlanSetPlannerMixin):
                result = engine.get_plan_set(
                    problem,
                    request.get("k"),
                    quality_bound=request.get("quality_bound"),
                    diversity=request.get("diversity"),
                    timeout=self._timeout(),
                    on_plan=self.add_plan,
                )
                for plan in result.plans:
                    self.add_plan(plan)
                self.out["plans"]["claimed_complete"] = result.complete
                status = self._report(result)
            elif mode in ("anytime", "planset") and isinstance(
                engine, AnytimePlannerMixin
            ):
                status = self._anytime(engine, problem)
            elif mode in ("oneshot", "planset") and isinstance(
                engine, OneshotPlannerMixin
            ):
                result = engine.solve(problem, timeout=self._timeout())
                self.add_plan(result.plan)
                status = self._report(result)
            else:
                raise UPLabWorkerError(
                    f"Engine '{step['engine']}' cannot be used in mode '{mode}'"
                )
        self.out["phases"]["solve"] = _since(start)
        self.out["up_status"] = status.name
        self.out["plans"]["claimed_optimal"] = status == Status.SOLVED_OPTIMALLY
        self.event("solve_end", up_status=status.name)
        if status in STATUS_OF_UP_STATUS and not (
            self.count and STATUS_OF_UP_STATUS[status] == "UNSOLVABLE"
        ):
            self.out["status"] = STATUS_OF_UP_STATUS[status]
            if self.out["status"] in ("ERROR", "UNSUPPORTED"):
                messages = [m["message"] for m in self.out["engine"]["log_messages"]]
                self.out["error"] = "\n".join(messages)[-4000:] or status.name
        elif self.count:
            self.out["status"] = "SOLVED"
        else:
            self.out["status"] = "ERROR"
            self.out["error"] = f"The engine reported {status.name} without a plan"

    def execute(self) -> None:
        """Parses the problem and runs the pipeline of the variant."""
        self.event("start")
        random.seed(self.run.get("seed"))
        up.environment.get_environment().credits_stream = None
        start = _clock()
        problem = load_problem(self.run["task"])
        self.out["phases"]["parse"] = _since(start)
        kind = sorted(problem.kind.features)
        self.out["problem"] = {"kind": kind, **problem_size(problem)}
        self.event("parse_end")
        self.checkpoint()
        variant = self.run["variant"]
        steps = list(variant["pipeline"])
        planner = None if variant.get("compile_only") or not steps else steps.pop()
        for step in steps:
            problem = self._compile(step, problem)
        if planner is None:
            self.out["status"] = "COMPILED"
        else:
            self._plan(planner, problem)

    def main(self) -> None:
        """Runs :func:`execute`, turning exceptions into statuses, and writes `solve.json`."""
        try:
            self.execute()
        except MemoryError:
            self.out["status"], self.out["error"] = "MEMOUT", "MemoryError"
        except Exception as error:
            traceback.print_exc()
            unsupported = isinstance(error, UPLabUnsupportedError)
            self.out["status"] = "UNSUPPORTED" if unsupported else "ERROR"
            self.out["error"] = f"{type(error).__name__}: {error}"
        if self.out["status"] in ("ERROR", "MEMOUT", "UNSUPPORTED"):
            self.event("error", status=self.out["status"], message=self.out["error"])
        self.checkpoint()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m up_lab_worker.solve <run-dir>")
    Run(Path(sys.argv[1])).main()
