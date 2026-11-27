"""Marimo notebook on the store of an experiment."""

from pathlib import Path
from typing import Union

from up_lab.store import BUILTIN_QUERIES

NOTEBOOK = "uplab_notebook.py"

_HEADER = '''"""Results of a up-lab experiment; run with: marimo edit {name}"""

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    from up_lab.results_plugin import figure
    from up_lab.store import load_query, open_store, query

    exp_dir = Path(__file__).resolve().parent
    store = open_store(exp_dir, read_only=True)
    return exp_dir, figure, load_query, mo, query, store


@app.cell
def _(exp_dir, mo):
    mo.md(f"# up-lab results: `{{exp_dir.name}}`")
    return
'''

_TABLE = '''

@app.cell
def _(load_query, mo, query, store):
    {name} = query(store, load_query("builtin:{name}"))
    mo.vstack([mo.md("## {title}"), mo.ui.table({name}, selection=None)])
    return ({name},)
'''

_FOOTER = '''

@app.cell
def _(figure, mo, runtimes):
    mo.vstack([mo.md("## Solved runs over time"), figure("cactus", runtimes, {})])
    return


@app.cell
def _(mo):
    sparql = mo.ui.text_area(
        value="""SELECT ?variant ?task ?outcome ?cpu WHERE {
  ?run a uplab:Run ;
       uplab:variant/uplab:name ?variant ;
       uplab:task/uplab:name ?task ;
       uplab:outcome ?outcome .
  OPTIONAL { ?run uplab:cpuTime ?cpu }
}
ORDER BY ?variant ?task""",
        rows=12,
        full_width=True,
        label="SPARQL (the prefixes uplab:, prov: and xsd: are predefined)",
    )
    mo.vstack([mo.md("## Query"), sparql])
    return (sparql,)


@app.cell
def _(mo, query, sparql, store):
    mo.ui.table(query(store, sparql.value), selection=None)
    return


if __name__ == "__main__":
    app.run()
'''


def write_notebook(exp_dir: Union[str, Path]) -> Path:
    """
    Writes a marimo notebook that opens the store read-only, shows the
    built-in queries and offers an editable SPARQL query.

    :param exp_dir: The experiment directory.
    :return: The notebook, `<exp-dir>/uplab_notebook.py`.
    """
    path = Path(exp_dir) / NOTEBOOK
    cells = [
        _TABLE.format(name=name, title=name.capitalize()) for name in BUILTIN_QUERIES
    ]
    path.write_text(_HEADER.format(name=NOTEBOOK) + "".join(cells) + _FOOTER)
    return path
