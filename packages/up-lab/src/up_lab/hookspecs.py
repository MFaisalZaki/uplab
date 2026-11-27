"""Plugin hooks of up-lab; each returns a ``dict`` mapping names to callables (or texts)."""

from typing import Callable, Dict

import pluggy

PROJECT = "up_lab"

hookspec = pluggy.HookspecMarker(PROJECT)
hookimpl = pluggy.HookimplMarker(PROJECT)


@hookspec
def uplab_benchmark_layouts() -> Dict[str, Callable]:  # type: ignore[empty-body]
    """
    :return: layout detectors ``(root: Path) -> list[dict] | None`` giving
        task dicts as in the benchmark manifest, or ``None`` on no match.
    """


@hookspec
def uplab_executors() -> Dict[str, Callable]:  # type: ignore[empty-body]
    """
    :return: executors ``(exp_dir: Path, config: ExperimentConfig) -> None``
        that run the pending runs.
    """


@hookspec
def uplab_analyses() -> Dict[str, Callable]:  # type: ignore[empty-body]
    """
    :return: analyses ``(store, exp_dir: Path, options: dict) ->
        Iterable[pyoxigraph.Triple]``.
    """


@hookspec
def uplab_renderers() -> Dict[str, Callable]:  # type: ignore[empty-body]
    """
    :return: renderers ``(df: DataFrame, stem: Path, options: dict) ->
        list[Path]``.
    """


@hookspec
def uplab_queries() -> Dict[str, str]:  # type: ignore[empty-body]
    """
    :return: SPARQL queries by name, usable in reports as ``builtin:<name>``
        (up-lab's own queries take precedence) or ``plugin:<name>``.
    """


@hookspec
def uplab_log_parsers() -> Dict[str, Callable]:  # type: ignore[empty-body]
    """
    :return: log parsers ``(stdout: str) -> dict``, keyed by engine name.
    """
