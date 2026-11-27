"""Built-in analyses, renderers and log parsers, registered as a plugin."""

import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List

import pandas as pd
import pluggy
import pyoxigraph as ox

from up_lab.config import UPLabError
from up_lab.store import UPLAB, XSD, query

hookimpl = pluggy.HookimplMarker("up_lab")

# Categorical colours in fixed order (colour-blind validated), then line styles.
PALETTE = ["#257de0", "#f47230", "#00b894", "#f2a600"]
PALETTE += ["#ee7fac", "#008a17", "#6a4ce2", "#ec4647"]


def triple(subject: str, name: str, value: Any) -> ox.Triple:
    """
    Builds a triple with a `uplab:` property and a typed literal.

    :param subject: The subject IRI.
    :param name: The local name of the property.
    :param value: A bool, int, float or str.
    :return: The triple.
    """
    if isinstance(value, bool):
        literal = ox.Literal(str(value).lower(), datatype=ox.NamedNode(XSD + "boolean"))
    elif isinstance(value, int):
        literal = ox.Literal(str(value), datatype=ox.NamedNode(XSD + "integer"))
    elif isinstance(value, float):
        literal = ox.Literal(repr(value), datatype=ox.NamedNode(XSD + "double"))
    else:
        literal = ox.Literal(str(value))
    return ox.Triple(ox.NamedNode(subject), ox.NamedNode(UPLAB + name), literal)


def _rows(store: ox.Store, sparql: str) -> List[Dict[str, Any]]:
    """Query solutions as dictionaries, unbound variables as `None`."""
    rows = query(store, sparql).astype(object).to_dict("records")
    return [{k: None if pd.isna(v) else v for k, v in row.items()} for row in rows]


def reference_cost(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Reference cost of every task: the one of the benchmark manifest if
    present, else the best valid cost over all runs of the experiment.

    :param store: The store.
    :param exp_dir: The experiment directory.
    :param options: Unused.
    :return: `uplab:effectiveReferenceCost` and `uplab:referenceCostSource`
        (`manifest` or `best-known`) for each task with a known cost.
    """
    sparql = """
        SELECT ?task (SAMPLE(?manifest) AS ?given) (MIN(?cost) AS ?best) WHERE {
          ?task a uplab:Task .
          OPTIONAL { ?task uplab:referenceCost ?manifest }
          OPTIONAL { ?run uplab:task ?task ; uplab:bestCost ?cost }
        } GROUP BY ?task"""
    for row in _rows(store, sparql):
        source = "manifest" if row["given"] is not None else "best-known"
        cost = row["given"] if row["given"] is not None else row["best"]
        if cost is not None:
            yield triple(row["task"], "effectiveReferenceCost", float(cost))
            yield triple(row["task"], "referenceCostSource", source)


def scores(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Per-run scores; needs the graph of the `reference-cost` analysis.

    - `uplab:coverage`: 1 if the outcome is `SOLVED`, else 0; with
      `uplab:coverageUnvalidated true` when no plan could be validated.
    - `uplab:qualityScore`: IPC satisficing score, reference cost / best
      valid cost (at most 1), 0 if not covered, absent if a cost is unknown.
    - `uplab:agileScore`: IPC-2018 agile score on the CPU time `t` to the
      first plan with time limit `T`: 1 if `t <= 1`, else `1 - log(t)/log(T)`;
      0 if not covered.
    - `uplab:par10`: CPU time, or 10 x the time limit if not covered.

    :param store: The store.
    :param exp_dir: The experiment directory.
    :param options: Unused.
    :return: The score triples of every run.
    """
    sparql = """
        SELECT ?run ?outcome ?valid ?best ?ref ?first ?cpu ?limit WHERE {
          ?run a uplab:Run ; uplab:outcome ?outcome ; uplab:task ?task .
          OPTIONAL { ?run uplab:plansValid ?valid }
          OPTIONAL { ?run uplab:bestCost ?best }
          OPTIONAL { ?task uplab:effectiveReferenceCost ?ref }
          OPTIONAL { ?run uplab:firstPlanCpuTime ?first }
          OPTIONAL { ?run uplab:cpuTime ?cpu }
          OPTIONAL { ?run uplab:timeLimit ?limit }
        }"""
    for row in _rows(store, sparql):
        run, limit = row["run"], row["limit"]
        covered = row["outcome"] == "SOLVED"
        yield triple(run, "coverage", int(covered))
        if covered and not row["valid"]:
            yield triple(run, "coverageUnvalidated", True)
        time = row["first"] if row["first"] is not None else row["cpu"]
        if not covered:
            yield triple(run, "qualityScore", 0.0)
            yield triple(run, "agileScore", 0.0)
        else:
            if row["best"] is not None and row["ref"] is not None:
                quality = row["ref"] / row["best"] if row["best"] > 0 else 1.0
                yield triple(run, "qualityScore", min(1.0, quality))
            if time is not None and time <= 1:
                yield triple(run, "agileScore", 1.0)
            elif time is not None and limit is not None and limit > 1:
                agile = 1 - math.log(time) / math.log(limit)
                yield triple(run, "agileScore", max(0.0, agile))
        if covered and row["cpu"] is not None:
            yield triple(run, "par10", float(row["cpu"]))
        elif not covered and limit is not None:
            yield triple(run, "par10", 10.0 * limit)


def render_csv(df: pd.DataFrame, stem: Path, options: Dict[str, Any]) -> List[Path]:
    """
    Writes the table as CSV.

    :param df: The query result.
    :param stem: The output path without extension.
    :param options: Unused.
    :return: The written file.
    """
    path = stem.with_name(stem.name + ".csv")
    df.to_csv(path, index=False)
    return [path]


def render_latex(df: pd.DataFrame, stem: Path, options: Dict[str, Any]) -> List[Path]:
    """
    Writes the table as a LaTeX `tabular`.

    :param df: The query result.
    :param stem: The output path without extension.
    :param options: `pivot: {index, columns, values}` reshapes the table
        first; `float_format` (default `%.2f`).
    :return: The written file.
    """
    pivot = options.get("pivot")
    if pivot:
        df = df.pivot(**pivot)
    path = stem.with_name(stem.name + ".tex")
    float_format = options.get("float_format", "%.2f")
    df.to_latex(path, index=bool(pivot), escape=True, float_format=float_format)
    return [path]


def _column(df: pd.DataFrame, options: Dict[str, Any], key: str, default: Any) -> Any:
    column = options.get(key, default)
    if column is not None and column not in df.columns:
        raise UPLabError(f"Option {key}={column!r}: no such column in {list(df)}")
    return column


def figure(kind: str, df: pd.DataFrame, options: Dict[str, Any]) -> Any:
    """
    Draws one of the built-in figures.

    :param kind: `cactus`, `scatter`, `bar` or `line`.
    :param df: The query result.
    :param options: The options of the renderer of that name.
    :return: The matplotlib figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from cycler import cycler

    styles = ["-", "--", ":", "-."]
    cycle = cycler(linestyle=styles) * cycler(color=PALETTE)
    rc = {"axes.prop_cycle": cycle, "lines.linewidth": 2, "grid.linewidth": 0.5}
    with sns.axes_style("whitegrid", rc=rc):
        fig, ax = plt.subplots(figsize=options.get("figsize", (6, 4)))
        _DRAW[kind](ax, df, options)
        sns.despine(fig)
    if options.get("title"):
        ax.set_title(options["title"])
    fig.tight_layout()
    return fig


def _cactus(ax: Any, df: pd.DataFrame, options: Dict[str, Any]) -> None:
    time = _column(df, options, "time", "cpu_time")
    group = _column(df, options, "group", "variant")
    if "outcome" in df.columns:
        df = df[df["outcome"] == "SOLVED"]
    for name, part in df.dropna(subset=[time]).groupby(group):
        times = sorted(part[time])
        ax.step(times, range(1, len(times) + 1), where="post", label=str(name))
    ax.set_xscale(options.get("xscale", "log"))
    ax.set_xlabel(options.get("xlabel", f"{time} (s)"))
    ax.set_ylabel(options.get("ylabel", "solved runs"))
    if ax.get_legend_handles_labels()[0]:
        ax.legend(title=group, frameon=False)


def _scatter(ax: Any, df: pd.DataFrame, options: Dict[str, Any]) -> None:
    value = _column(df, options, "value", "cpu_time")
    group = _column(df, options, "group", "variant")
    key = _column(df, options, "key", "task")
    if "x" not in options or "y" not in options:
        raise UPLabError("The scatter renderer needs the options x and y (variants).")
    x, y = options["x"], options["y"]
    wide = df.pivot_table(index=key, columns=group, values=value, aggfunc="mean")
    missing = [v for v in (x, y) if v not in wide.columns]
    if missing:
        raise UPLabError(f"No {value} for {missing}; found {list(wide.columns)}")
    wide = wide[[x, y]].dropna()
    ax.scatter(wide[x], wide[y], s=36, color=PALETTE[0], edgecolor="white")
    ax.axline((1, 1), slope=1, color="#666666", linewidth=1, linestyle="--")
    ax.set_xscale(options.get("scale", "log"))
    ax.set_yscale(options.get("scale", "log"))
    ax.set_xlabel(f"{x}: {value}")
    ax.set_ylabel(f"{y}: {value}")


def _seaborn(plot: str) -> Callable[[Any, pd.DataFrame, Dict[str, Any]], None]:
    def draw(ax: Any, df: pd.DataFrame, options: Dict[str, Any]) -> None:
        import seaborn as sns

        x = _column(df, options, "x", df.columns[0])
        y = _column(df, options, "y", df.columns[-1])
        hue = _column(df, options, "hue", None)
        palette = (PALETTE * df[hue].nunique())[: df[hue].nunique()] if hue else None
        color = None if hue else PALETTE[0]
        getattr(sns, plot)(
            data=df, x=x, y=y, hue=hue, palette=palette, color=color, ax=ax
        )
        if plot == "barplot":
            ax.tick_params(axis="x", rotation=options.get("rotation", 0))
        if hue:
            ax.legend(title=hue, frameon=False)

    return draw


_DRAW = {
    "cactus": _cactus,
    "scatter": _scatter,
    "bar": _seaborn("barplot"),
    "line": _seaborn("lineplot"),
}


def _figure_renderer(kind: str) -> Callable[..., List[Path]]:
    def render(df: pd.DataFrame, stem: Path, options: Dict[str, Any]) -> List[Path]:
        import matplotlib.pyplot as plt

        fig = figure(kind, df, options)
        paths = [stem.with_name(f"{stem.name}.{ext}") for ext in ("pdf", "png")]
        for path in paths:
            fig.savefig(path, dpi=200)
        plt.close(fig)
        return paths

    render.__doc__ = f"Saves the `{kind}` figure of the table as PDF and PNG."
    return render


@hookimpl
def uplab_analyses() -> Dict[str, Callable[..., Iterable[ox.Triple]]]:
    """
    :return: The built-in analyses; `analyze` always runs them first.
    """
    return {"reference-cost": reference_cost, "scores": scores}


@hookimpl
def uplab_renderers() -> Dict[str, Callable[..., List[Path]]]:
    """
    :return: The built-in renderers.
    """
    renderers = {"csv": render_csv, "latex": render_latex}
    renderers.update({kind: _figure_renderer(kind) for kind in _DRAW})
    return renderers


@hookimpl
def uplab_log_parsers() -> Dict[str, Callable[[str], Dict[str, Any]]]:
    """
    :return: No log parser is built in; plugins add them per engine name.
    """
    return {}
