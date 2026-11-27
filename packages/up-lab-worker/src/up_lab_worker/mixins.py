"""Operation-mode mixin for engines that return a set of plans."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import IO, Any, Callable, Dict, List, Optional
from warnings import warn

import unified_planning as up
from unified_planning.engines.results import (
    LogMessage,
    PlanGenerationResultStatus,
    Result,
)
from unified_planning.plans import Plan


@dataclass
class PlanSetGenerationResult(Result):
    """Class that represents the result of a `get_plan_set` call."""

    status: PlanGenerationResultStatus
    plans: List[Plan]
    engine_name: str
    metrics: Optional[Dict[str, str]] = field(default=None)
    log_messages: Optional[List[LogMessage]] = field(default=None)
    complete: Optional[bool] = field(default=None)
    """`True` if the set contains every plan within the quality bound."""

    def is_definitive_result(self, *args) -> bool:
        return self.status != PlanGenerationResultStatus.INTERMEDIATE


class PlanSetPlannerMixin(ABC):
    """Base class that must be extended by an :class:`~unified_planning.engines.Engine` that is also a `PlanSetPlanner`."""

    @staticmethod
    def is_plan_set_planner() -> bool:
        return True

    def get_plan_set(
        self,
        problem: "up.model.AbstractProblem",
        k: Optional[int],
        quality_bound: Optional[float] = None,
        diversity: Any = None,
        timeout: Optional[float] = None,
        output_stream: Optional[IO[str]] = None,
        on_plan: Optional[Callable[[Plan], None]] = None,
    ) -> PlanSetGenerationResult:
        """
        This method takes an `AbstractProblem` and returns a set of plans for it.

        :param problem: is the `AbstractProblem` to solve.
        :param k: is the number of plans requested; `None` means as many as possible.
        :param quality_bound: plans may cost up to `quality_bound` times the cost
            of an optimal plan; defaults to `None` (no bound).
        :param diversity: is an opaque, JSON-like description of the diversity
            requested, interpreted by the planner; defaults to `None`.
        :param timeout: is the time in seconds that the planner has at max; defaults to `None`.
        :param output_stream: is a stream of strings where the planner writes its
            output while it is solving the problem; defaults to `None`.
        :param on_plan: is a function called by the planner with every plan as soon
            as it is found; defaults to `None`.
        :return: the `PlanSetGenerationResult` created by the planner.
        """
        assert isinstance(self, up.engines.engine.Engine)
        if not self.skip_checks and not self.supports(problem.kind):
            msg = f"We cannot establish whether {self.name} can solve this problem!"
            if self.error_on_failed_checks:
                raise up.exceptions.UPUsageError(msg)
            else:
                warn(msg)
        if k is not None and k < 1:
            raise up.exceptions.UPUsageError("The number of plans must be positive!")
        if quality_bound is not None and quality_bound < 1:
            raise up.exceptions.UPUsageError("The quality bound must be at least 1!")
        return self._get_plan_set(
            problem, k, quality_bound, diversity, timeout, output_stream, on_plan
        )

    @abstractmethod
    def _get_plan_set(
        self,
        problem: "up.model.AbstractProblem",
        k: Optional[int],
        quality_bound: Optional[float],
        diversity: Any,
        timeout: Optional[float],
        output_stream: Optional[IO[str]],
        on_plan: Optional[Callable[[Plan], None]],
    ) -> PlanSetGenerationResult:
        """Method called by the PlanSetPlannerMixin.get_plan_set method."""
        raise NotImplementedError
