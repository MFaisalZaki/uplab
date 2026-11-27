"""In-environment runner and plan-set planner mixin for up-lab."""

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

__all__ = ["PlanSetPlannerMixin", "PlanSetGenerationResult", "__version__"]

if TYPE_CHECKING:
    from up_lab_worker.mixins import PlanSetGenerationResult, PlanSetPlannerMixin


def __getattr__(name: str) -> Any:
    # Imported lazily so that the driver does not pay for importing unified_planning.
    if name in ("PlanSetPlannerMixin", "PlanSetGenerationResult"):
        from up_lab_worker import mixins

        return getattr(mixins, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
