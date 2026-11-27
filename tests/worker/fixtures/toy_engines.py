"""Toy engines for the worker tests, referenced as ``toy_engines:<Class>``."""

import time
from typing import Iterator, List

import unified_planning as up
from unified_planning.engines import Engine, PlanGenerationResult
from unified_planning.engines import PlanGenerationResultStatus as Status
from unified_planning.engines.mixins import AnytimePlannerMixin, OneshotPlannerMixin
from unified_planning.engines.sequential_simulator import UPSequentialSimulator
from unified_planning.plans import ActionInstance, SequentialPlan

from up_lab_worker import PlanSetGenerationResult, PlanSetPlannerMixin


def enumerate_plans(
    problem: "up.model.Problem", depth: int
) -> Iterator[SequentialPlan]:
    """Yields every plan of at most `depth` actions, shortest first."""
    simulator = UPSequentialSimulator(problem, error_on_failed_checks=False)
    frontier = [(simulator.get_initial_state(), [])]
    for _ in range(depth + 1):
        successors = []
        for state, prefix in frontier:
            if simulator.is_goal(state):
                yield SequentialPlan(prefix)
                continue
            for action, params in simulator.get_applicable_actions(state):
                successor = simulator.apply(state, action, params)
                step = ActionInstance(action, params)
                successors.append((successor, prefix + [step]))
        frontier = successors


class ToyEngine(Engine):
    """Common part of the toy engines."""

    def __init__(self, depth: int = 3, sleep: float = 0, **options):
        Engine.__init__(self)
        self.optimality_metric_required = False
        self.depth, self.sleep = depth, sleep

    @property
    def name(self) -> str:
        return type(self).__name__

    @staticmethod
    def supported_kind() -> "up.model.ProblemKind":
        kind = up.model.ProblemKind()
        kind.set_problem_class("ACTION_BASED")
        kind.set_typing("FLAT_TYPING")
        return kind

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return problem_kind <= ToyEngine.supported_kind()


class Sleeper(ToyEngine, OneshotPlannerMixin):
    """Never returns in time."""

    def _solve(self, problem, heuristic=None, timeout=None, output_stream=None):
        time.sleep(600)
        return PlanGenerationResult(Status.TIMEOUT, None, self.name)


class Crasher(ToyEngine, OneshotPlannerMixin):
    """Raises while solving."""

    def _solve(self, problem, heuristic=None, timeout=None, output_stream=None):
        raise RuntimeError("the toy engine crashed")


class Hog(ToyEngine, OneshotPlannerMixin):
    """Allocates memory until it fails."""

    def _solve(self, problem, heuristic=None, timeout=None, output_stream=None):
        blocks: List[bytearray] = []
        while True:
            blocks.append(bytearray(64 * 2**20))


class Picky(ToyEngine, OneshotPlannerMixin):
    """Supports nothing."""

    @staticmethod
    def supports(problem_kind: "up.model.ProblemKind") -> bool:
        return False

    def _solve(self, problem, heuristic=None, timeout=None, output_stream=None):
        raise AssertionError("never called")


class ToyAnytime(ToyEngine, AnytimePlannerMixin):
    """Yields every plan twice, then sleeps before the final result."""

    def _get_solutions(self, problem, timeout=None, output_stream=None):
        last = None
        for plan in enumerate_plans(problem, self.depth):
            for _ in range(2):
                yield PlanGenerationResult(Status.INTERMEDIATE, plan, self.name)
            last = plan
        time.sleep(self.sleep)
        yield PlanGenerationResult(Status.SOLVED_SATISFICING, last, self.name)


class ToyPlanSet(ToyEngine, PlanSetPlannerMixin):
    """Reports the first plan through `on_plan` and returns the first `k` plans."""

    def _get_plan_set(
        self, problem, k, quality_bound, diversity, timeout, output_stream, on_plan
    ):
        plans = []
        for plan in enumerate_plans(problem, self.depth):
            if not plans and on_plan is not None:
                on_plan(plan)
                time.sleep(0.2)
            plans.append(plan)
            if k is not None and len(plans) >= k:
                break
        return PlanSetGenerationResult(
            Status.SOLVED_SATISFICING,
            plans,
            self.name,
            metrics={"diversity": str(diversity)},
            complete=k is None or len(plans) < k,
        )
