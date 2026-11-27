from pathlib import Path
from typing import Any, Callable, Dict

import pytest

from up_lab import plugins
from up_lab.config import UPLabError
from up_lab.experiment import Experiment
from up_lab.hookspecs import hookimpl


class Plugin:
    def __init__(self) -> None:
        self.calls: list = []

    @hookimpl
    def uplab_executors(self) -> Dict[str, Callable]:
        return {"recording": lambda exp_dir, config: self.calls.append(exp_dir)}

    @hookimpl
    def uplab_benchmark_layouts(self) -> Dict[str, Callable]:
        return {"anml": lambda root: None, "flat": lambda root: []}


def test_hook_merges_plugins(make_exp: Callable[..., Any]) -> None:
    assert {"local", "slurm"} <= set(plugins.hook("uplab_executors"))
    builtin = plugins.hook("uplab_benchmark_layouts")
    assert list(builtin)[:2] == ["pddl", "anml"]
    assert isinstance(plugins.hook("uplab_log_parsers"), dict)
    plugin = Plugin()
    plugins.manager().register(plugin)
    try:
        layouts = plugins.hook("uplab_benchmark_layouts")
        assert {"pddl", "anml", "flat"} <= set(layouts)
        assert layouts["anml"] is not builtin["anml"]  # later plugins override
        exp_dir, config = make_exp(executor={"kind": "recording"})
        Experiment(config, exp_dir).run()
        assert plugin.calls == [exp_dir]
    finally:
        plugins.manager().unregister(plugin)
    with pytest.raises(UPLabError, match="unknown executor"):
        Experiment(config, exp_dir).run()
