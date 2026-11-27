"""Python API: an experiment and the steps of its pipeline."""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from up_lab import benchmarks, build, envs
from up_lab.config import ExperimentConfig, UPLabError, dump_config, load_config
from up_lab.plugins import hook


class Experiment:
    """
    An experiment configuration bound to an experiment directory.

    :param config: the configuration, or the path of an ``experiment.yaml``.
    :param exp_dir: the experiment directory; defaults to
        ``experiments/<name>`` next to the configuration file (or in the
        working directory for a configuration object).
    """

    def __init__(
        self,
        config: Union[ExperimentConfig, str, Path],
        exp_dir: Union[str, Path, None] = None,
    ):
        base = Path.cwd()
        if not isinstance(config, ExperimentConfig):
            base = Path(config).resolve().parent
            config = load_config(config)
        self.config: ExperimentConfig = config
        self.exp_dir: Path = Path(exp_dir or base / "experiments" / config.name)
        self.exp_dir = self.exp_dir.resolve()
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, self.exp_dir / "experiment.yaml")

    @classmethod
    def open(
        cls, path: Union[str, Path], exp_dir: Union[str, Path, None] = None
    ) -> "Experiment":
        """
        :param path: an experiment directory or an ``experiment.yaml``.
        :param exp_dir: the experiment directory, when ``path`` is a YAML
            file outside of it.
        :return: the experiment.
        """
        path = Path(path)
        if path.is_dir():
            return cls(path / "experiment.yaml", path)
        if exp_dir is None and (path.parent / "lock.json").exists():
            exp_dir = path.parent  # the resolved copy inside an experiment
        return cls(path, exp_dir)

    def resolve(self, force: bool = False) -> Dict[str, Any]:
        """
        Fetches the benchmark sources, builds the planner environments and
        writes ``lock.json``.

        :param force: rebuild environments whose inputs are unchanged.
        :return: the content of ``lock.json``.
        """
        lock = {
            "schema": build.LOCK_SCHEMA,
            "benchmarks": {
                b.name: benchmarks.fetch(b, self.exp_dir)
                for b in self.config.benchmarks
            },
            "planners": {
                p.name: envs.build_env(p, self.exp_dir, force)
                for p in self.config.planners
            },
        }
        (self.exp_dir / "lock.json").write_text(json.dumps(lock, indent=2) + "\n")
        return lock

    def build(self) -> List[str]:
        """
        Writes the run directories of the run matrix.

        :return: the run ids.
        """
        return build.build(self.exp_dir, self.config)

    def run(self) -> None:
        """Executes the pending runs through the configured executor."""
        executors = hook("uplab_executors")
        kind = self.config.executor.kind
        if kind not in executors:
            raise UPLabError(f"unknown executor {kind!r} (known: {sorted(executors)})")
        executors[kind](self.exp_dir, self.config)

    def status(self) -> Dict[str, int]:
        """
        :return: the number of runs per state: the ``status`` of the result,
            or ``PENDING`` for runs without one.
        """
        counts: Counter = Counter()
        for line in build.read_index(self.exp_dir):
            result = self.exp_dir / "runs" / line["run_id"] / "result.json"
            if result.exists():
                counts[json.loads(result.read_text()).get("status", "ERROR")] += 1
            else:
                counts["PENDING"] += 1
        return dict(counts)

    def collect(self, jobs: int = 1) -> Any:
        """
        Validates the plans and loads the store.

        :param jobs: number of validation processes.
        """
        from up_lab.collect import collect

        return collect(self.exp_dir, jobs=jobs)

    def analyze(self, names: Optional[List[str]] = None) -> Any:
        """
        :param names: the analyses to run; all the configured ones if ``None``.
        """
        from up_lab.analysis import analyze

        return analyze(self.exp_dir, names=names)

    def report(self, names: Optional[List[str]] = None, html: bool = False) -> Any:
        """
        :param names: the reports to write; all the configured ones if ``None``.
        :param html: also write an HTML overview.
        """
        from up_lab.reports import report

        return report(self.exp_dir, names=names, html=html)

    def serve(self, host: str = "127.0.0.1", port: int = 7878) -> Any:
        """
        Serves the store as a SPARQL endpoint.

        :param host: the address to bind.
        :param port: the port to listen on.
        """
        from up_lab.serve import serve

        return serve(self.exp_dir, host, port)

    def notebook(self) -> Path:
        """
        :return: the path of the marimo notebook written for the store.
        """
        from up_lab.notebook import write_notebook

        path: Path = write_notebook(self.exp_dir)
        return path
