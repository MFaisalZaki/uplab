"""Oxigraph store of an experiment: opening, querying, shared names."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import pyoxigraph as ox

from up_lab.config import UPLabError, load_config

UPLAB = "urn:uplab:ns#"
PROV = "http://www.w3.org/ns/prov#"
XSD = "http://www.w3.org/2001/XMLSchema#"
PREFIXES = {"uplab": UPLAB, "prov": PROV, "xsd": XSD}
BUILTIN_QUERIES = ("coverage", "scores", "runtimes", "errors", "plansets", "anytime")

_INTEGERS = {XSD + t for t in ("integer", "int", "long", "nonNegativeInteger")}
_FLOATS = {XSD + t for t in ("double", "float", "decimal")}


def experiment_config(exp_dir: Path) -> Dict[str, Any]:
    """
    Reads the experiment configuration as plain data.

    :param exp_dir: The experiment directory.
    :return: The configuration; the results modules use `name`, `analyses`
        and `reports`.
    """
    return load_config(Path(exp_dir) / "experiment.yaml").model_dump()


def graph_iri(experiment: str, analysis: Optional[str] = None) -> ox.NamedNode:
    """
    Returns the named graph of the records or of one analysis of an experiment.

    :param experiment: The experiment name.
    :param analysis: The analysis name, or `None` for the records graph.
    :return: The graph name.
    """
    suffix = f":analysis:{analysis}" if analysis else ""
    return ox.NamedNode(f"urn:uplab:graph:{experiment}{suffix}")


def open_store(exp_dir: Path, read_only: bool = False) -> ox.Store:
    """
    Opens the on-disk store of an experiment.

    :param exp_dir: The experiment directory.
    :param read_only: Open without taking the write lock; the store must exist.
    :return: The store.
    """
    path = Path(exp_dir) / "store"
    if not read_only:
        return ox.Store(str(path))
    if not path.exists():
        raise UPLabError(f"No store in {exp_dir}: run 'uplab collect' first.")
    return ox.Store.read_only(str(path))


def load_query(ref: str, exp_dir: Optional[Path] = None) -> str:
    """
    Returns the text of a query given as `builtin:<name>` or as a `.rq` path.

    :param ref: The query reference.
    :param exp_dir: The directory relative paths are resolved against.
    :return: The SPARQL text.
    """
    if ref.startswith("builtin:"):
        path = Path(__file__).parent / "queries" / f"{ref[len('builtin:'):]}.rq"
    else:
        path = Path(ref)
        if exp_dir is not None and (Path(exp_dir) / ref).exists():
            path = Path(exp_dir) / ref
    if not path.exists():
        raise UPLabError(f"Query not found: {ref}")
    return path.read_text()


def with_prefixes(sparql: str) -> str:
    """
    Prepends the common PREFIX declarations a query does not declare itself.

    :param sparql: The query.
    :return: The query with the missing prefixes declared.
    """
    missing = [
        f"PREFIX {prefix}: <{iri}>\n"
        for prefix, iri in PREFIXES.items()
        if not re.search(rf"PREFIX\s+{prefix}:", sparql, re.IGNORECASE)
    ]
    return "".join(missing) + sparql


def run_query(store: ox.Store, sparql: str) -> Any:
    """
    Runs a query with the union of all named graphs as the default graph.

    :param store: The store.
    :param sparql: The query; common prefixes are added when missing.
    :return: The pyoxigraph result.
    """
    try:
        return store.query(with_prefixes(sparql), use_default_graph_as_union=True)
    except (SyntaxError, ValueError, OSError) as error:
        raise UPLabError(f"SPARQL query failed: {error}") from error


def to_python(term: Any) -> Any:
    """
    Converts an RDF term to a plain Python value.

    :param term: A pyoxigraph term or `None`.
    :return: An int, float, bool or str, or `None`.
    """
    if term is None:
        return None
    if isinstance(term, ox.Literal):
        datatype = term.datatype.value
        if datatype in _INTEGERS:
            return int(term.value)
        if datatype in _FLOATS:
            return float(term.value)
        if datatype == XSD + "boolean":
            return term.value in ("true", "1")
        return term.value
    return term.value if hasattr(term, "value") else str(term)


def query(source: Union[Path, str, ox.Store], sparql: str) -> pd.DataFrame:
    """
    Runs a SELECT (or CONSTRUCT) query and returns its result as a table.

    :param source: The experiment directory, or an open store.
    :param sparql: The query; no GRAPH clause is needed.
    :return: One row per solution, columns in SELECT order (`s`, `p`, `o` for
        CONSTRUCT), with plain Python values.
    """
    store = source if isinstance(source, ox.Store) else open_store(source, True)
    result = run_query(store, sparql)
    if isinstance(result, ox.QuerySolutions):
        columns = [variable.value for variable in result.variables]
        rows = [[to_python(solution[c]) for c in columns] for solution in result]
    elif isinstance(result, ox.QueryTriples):
        columns = ["s", "p", "o"]
        rows = [[to_python(term) for term in triple] for triple in result]
    else:
        columns, rows = ["result"], [[bool(result)]]
    return pd.DataFrame(rows, columns=columns)
