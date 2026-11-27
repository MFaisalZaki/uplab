"""Plugin manager: built-ins, then entry points of the group ``up_lab``."""

import importlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict

import pluggy

from up_lab import hookspecs
from up_lab.hookspecs import PROJECT, hookimpl


class _CorePlugin:
    """Built-in benchmark layouts and executors."""

    @hookimpl
    def uplab_benchmark_layouts(self) -> Dict[str, Callable]:
        from up_lab import benchmarks

        return {"pddl": benchmarks.pddl_layout, "anml": benchmarks.anml_layout}

    @hookimpl
    def uplab_executors(self) -> Dict[str, Callable]:
        def local(exp_dir: Path, config: Any) -> None:
            from up_lab.executors.local import run_local

            run_local(exp_dir, config)

        def slurm(exp_dir: Path, config: Any) -> None:
            from up_lab.executors.slurm import run_slurm

            run_slurm(exp_dir, config)

        return {"local": local, "slurm": slurm}


@lru_cache(maxsize=None)
def manager() -> pluggy.PluginManager:
    """
    :return: the plugin manager with built-in and installed plugins.
    """
    pm = pluggy.PluginManager(PROJECT)
    pm.add_hookspecs(hookspecs)
    pm.register(_CorePlugin(), name="up_lab.core")
    try:
        pm.register(importlib.import_module("up_lab.results_plugin"))
    except ModuleNotFoundError as error:
        if error.name != "up_lab.results_plugin":
            raise
    pm.load_setuptools_entrypoints(PROJECT)
    return pm


def hook(name: str) -> Dict[str, Any]:
    """
    Merges what every plugin returns for a hook; plugins registered later
    (installed ones) override built-ins of the same name.

    :param name: a hook name, e.g. ``uplab_executors``.
    :return: the merged mapping from names to callables (query texts for
        ``uplab_queries``).
    """
    merged: Dict[str, Any] = {}
    # pluggy returns results last-registered first.
    for result in reversed(getattr(manager().hook, name)()):
        merged.update(result or {})
    return merged
