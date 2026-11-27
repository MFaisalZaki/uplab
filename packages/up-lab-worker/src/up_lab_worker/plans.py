"""Text plan files: one action per line, readable back against the problem."""

import re
from fractions import Fraction
from pathlib import Path
from typing import Hashable, List, Optional, Tuple, Union

import unified_planning as up
from unified_planning.exceptions import UPException
from unified_planning.plans import (
    ActionInstance,
    Plan,
    PlanKind,
    SequentialPlan,
    TimeTriggeredPlan,
)


class UPPlanFileError(UPException):
    """Raised when a plan file cannot be read back against a problem."""


_TIMED = re.compile(r"^(\S+)\s*:\s*(.+?)(?:\s*\[\s*(\S+)\s*\])?$")
_ACTION = re.compile(r"^([^\s()]+)\s*(?:\((.*)\))?$")


def _number(value: Fraction) -> str:
    text = str(float(value))
    return text if Fraction(text) == value else str(value)


def _action(instance: ActionInstance) -> str:
    args = []
    for param in instance.actual_parameters:
        if param.is_object_exp():
            args.append(param.object().name)
        elif param.is_bool_constant():
            args.append("true" if param.bool_constant_value() else "false")
        else:
            args.append(str(param))
    return f"{instance.action.name}({', '.join(args)})"


def _expressible(plan: Plan) -> bool:
    if plan.kind == PlanKind.SEQUENTIAL_PLAN:
        assert isinstance(plan, SequentialPlan)
        instances = plan.actions
    elif plan.kind == PlanKind.TIME_TRIGGERED_PLAN:
        assert isinstance(plan, TimeTriggeredPlan)
        instances = [instance for _, instance, _ in plan.timed_actions]
    else:
        return False
    return all(i.agent is None and i.motion_paths is None for i in instances)


def plan_lines(plan: Plan) -> List[str]:
    """
    Returns the lines of the plan file of the given plan.

    :param plan: The `Plan` to write.
    :return: One line per action, or the lines of `str(plan)` for plan kinds
        the format cannot express.
    """
    if not _expressible(plan):
        return str(plan).splitlines()
    if isinstance(plan, SequentialPlan):
        return [_action(instance) for instance in plan.actions]
    assert isinstance(plan, TimeTriggeredPlan)
    lines = []
    for start, instance, duration in sorted(
        plan.timed_actions, key=lambda x: (x[0], _action(x[1]))
    ):
        line = f"{_number(start)}: {_action(instance)}"
        lines.append(line if duration is None else f"{line} [{_number(duration)}]")
    return lines


def canonical_plan(plan: Plan) -> Hashable:
    """
    Returns a hashable canonical form of the given plan, used to detect duplicates.

    :param plan: The `Plan` to canonicalize.
    :return: A hashable value, equal for plans written to identical plan files.
    """
    return (plan.kind.name, tuple(plan_lines(plan)))


def write_plan(plan: Plan, path: Union[str, Path]) -> None:
    """
    Writes the given plan to a text file.

    :param plan: The `Plan` to write.
    :param path: The path of the file to write.
    """
    lines = plan_lines(plan)
    Path(path).write_text("".join(line + "\n" for line in lines), encoding="utf-8")


def _read_action(problem: "up.model.Problem", text: str, where: str) -> ActionInstance:
    match = _ACTION.match(text.strip())
    if match is None:
        raise UPPlanFileError(f"{where}: cannot parse '{text}'")
    name, arguments = match.group(1), (match.group(2) or "").strip()
    if not problem.has_action(name):
        raise UPPlanFileError(f"{where}: unknown action '{name}'")
    action = problem.action(name)
    names = [a.strip() for a in arguments.split(",")] if arguments else []
    if len(names) != len(action.parameters):
        raise UPPlanFileError(f"{where}: wrong number of arguments for '{name}'")
    manager = problem.environment.expression_manager
    params = []
    for parameter, arg in zip(action.parameters, names):
        try:
            if parameter.type.is_user_type():
                params.append(manager.ObjectExp(problem.object(arg)))
            elif parameter.type.is_bool_type():
                params.append(manager.Bool({"true": True, "false": False}[arg.lower()]))
            elif parameter.type.is_int_type():
                params.append(manager.Int(int(arg)))
            else:
                params.append(manager.Real(Fraction(arg)))
        except (UPException, KeyError, ValueError, ZeroDivisionError):
            raise UPPlanFileError(f"{where}: bad argument '{arg}' for '{name}'")
    try:
        return ActionInstance(action, tuple(params))
    except (UPException, AssertionError) as error:
        raise UPPlanFileError(f"{where}: {error}")


def read_plan(problem: "up.model.AbstractProblem", path: Union[str, Path]) -> Plan:
    """
    Reads a plan file written by :func:`write_plan`.

    :param problem: The `Problem` the plan refers to.
    :param path: The path of the plan file.
    :return: A `SequentialPlan`, or a `TimeTriggeredPlan` if the lines are timed.
    :raises: :exc:`UPPlanFileError` if the file cannot be read against the problem.
    """
    if not isinstance(problem, up.model.Problem):
        raise UPPlanFileError(
            f"{path}: plans of {type(problem).__name__} are not readable"
        )
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise UPPlanFileError(f"{path}: {error}")
    sequential: List[ActionInstance] = []
    timed: List[Tuple[Fraction, ActionInstance, Optional[Fraction]]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        line, where = line.strip(), f"{path}:{number}"
        if not line or line.startswith(";"):
            continue
        match = _TIMED.match(line)
        try:
            start = None if match is None else Fraction(match.group(1))
        except (ValueError, ZeroDivisionError):
            start = None
        if match is None or start is None:
            sequential.append(_read_action(problem, line, where))
            continue
        try:
            duration = None if match.group(3) is None else Fraction(match.group(3))
        except (ValueError, ZeroDivisionError):
            raise UPPlanFileError(f"{where}: bad duration '{match.group(3)}'")
        timed.append((start, _read_action(problem, match.group(2), where), duration))
    if timed and sequential:
        raise UPPlanFileError(f"{path}: mixes timed and untimed actions")
    if timed or (not sequential and problem.kind.has_continuous_time()):
        return TimeTriggeredPlan(timed, problem.environment)
    return SequentialPlan(sequential, problem.environment)
