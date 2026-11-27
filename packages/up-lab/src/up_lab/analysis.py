"""Analyses: derived triples, each analysis in its own named graph."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pyoxigraph as ox

from up_lab import results_plugin
from up_lab.config import UPLabError
from up_lab.plugins import hook
from up_lab.store import experiment_config, graph_iri, load_query, open_store
from up_lab.store import run_query


def analyze(
    exp_dir: Union[str, Path], names: Optional[List[str]] = None
) -> Dict[str, int]:
    """
    Runs the built-in analyses, then the configured ones; the triples of each
    replace its named graph `urn:uplab:graph:<experiment>:analysis:<name>`.

    An analysis whose name ends in `.rq` is a file with a `CONSTRUCT` query
    (its graph is named after the file stem); any other name is a callable
    of the `uplab_analyses` hook.

    :param exp_dir: The experiment directory.
    :param names: Run only the analyses with these names.
    :return: The number of triples stored per analysis.
    """
    exp_dir = Path(exp_dir)
    config = experiment_config(exp_dir)
    todo: Dict[str, Dict[str, Any]] = {n: {} for n in results_plugin.uplab_analyses()}
    todo.update({a["name"]: a.get("options") or {} for a in config["analyses"]})
    unknown = set(names or []) - set(todo)
    if unknown:
        raise UPLabError(f"Unknown analyses: {sorted(unknown)}")
    analyses = hook("uplab_analyses")
    store = open_store(exp_dir)
    counts = {}
    for name, options in todo.items():
        if names is not None and name not in names:
            continue
        if name.endswith(".rq"):
            result = run_query(store, load_query(name, exp_dir))
            if not isinstance(result, ox.QueryTriples):
                raise UPLabError(f"Analysis {name} is not a CONSTRUCT query.")
            triples, label = list(result), Path(name).stem
        elif name in analyses:
            triples, label = list(analyses[name](store, exp_dir, options)), name
        else:
            raise UPLabError(f"Unknown analysis: {name}")
        graph = graph_iri(config["name"], label)
        store.remove_graph(graph)
        store.extend(ox.Quad(t.subject, t.predicate, t.object, graph) for t in triples)
        counts[name] = len(triples)
    store.flush()
    return counts
