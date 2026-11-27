"""Hook implementations; loaded through the entry-point group ``up_lab``."""

from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pyoxigraph as ox

from up_lab.hookspecs import hookimpl
from up_lab_stats import analyses, render

QUERIES = Path(__file__).parent / "queries"


@hookimpl
def uplab_analyses() -> Dict[str, Callable[..., Iterable[ox.Triple]]]:
    """
    :return: The analyses of this plugin; all need the graphs of the
        built-in `reference-cost` and `scores`, which `analyze` runs first.
    """
    return {
        "optimality-claims": analyses.optimality_claims,
        "pairwise-tests": analyses.pairwise_tests,
        "double-hits": analyses.double_hits,
        "domain-scores": analyses.domain_scores,
        "portfolio": analyses.portfolio,
        "lab-scores": analyses.lab_scores,
    }


@hookimpl
def uplab_queries() -> Dict[str, str]:
    """
    :return: The queries of `up_lab_stats/queries`, by file stem.
    """
    return {path.stem: path.read_text() for path in sorted(QUERIES.glob("*.rq"))}


@hookimpl
def uplab_renderers() -> Dict[str, Callable[..., List[Path]]]:
    """
    :return: The `anytime` renderer.
    """
    return {"anytime": render.render_anytime}
