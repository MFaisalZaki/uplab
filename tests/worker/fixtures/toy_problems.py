"""A Python factory task for the worker tests."""

from unified_planning.shortcuts import BoolType, Fluent, InstantaneousAction, Problem


def make_problem(steps: int = 2) -> Problem:
    """Returns a problem solved by `step_1(), ..., step_<steps>()` in order."""
    problem = Problem(f"chain-{steps}")
    done = [Fluent(f"done_{i}", BoolType()) for i in range(steps + 1)]
    for i, fluent in enumerate(done):
        problem.add_fluent(fluent, default_initial_value=(i == 0))
    for i in range(1, steps + 1):
        action = InstantaneousAction(f"step_{i}")
        action.add_precondition(done[i - 1])
        action.add_effect(done[i], True)
        problem.add_action(action)
    problem.add_goal(done[steps])
    return problem
