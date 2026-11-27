import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

from up_lab import serve as serve_module
from up_lab.config import UPLabError
from up_lab.notebook import write_notebook
from up_lab.plugins import hook
from up_lab.reports import report
from up_lab.serve import serve_command
from up_lab.store import load_query, query


def test_default_reports(exp_copy: Path):
    written = report(exp_copy)
    names = sorted(p.name for p in written)
    assert names == [
        "coverage.csv", "coverage.tex", "errors.csv", "runtimes.pdf",
        "runtimes.png", "scores.csv", "scores.tex",
    ]
    assert all(p.parent == exp_copy / "reports" and p.stat().st_size > 0 for p in written)
    coverage = pd.read_csv(exp_copy / "reports" / "coverage.csv")
    assert coverage.values.tolist()[-1] == ["good", "TOTAL", 4, 4]
    tex = (exp_copy / "reports" / "scores.tex").read_text()
    assert "\\begin{tabular}" in tex and "good & TOTAL & 4 & 4.00 & 3.00" in tex
    assert (exp_copy / "reports" / "runtimes.pdf").read_bytes().startswith(b"%PDF")
    assert [p.name for p in report(exp_copy, names=["errors"])] == ["errors.csv"]
    with pytest.raises(UPLabError):
        report(exp_copy, names=["nope"])


def test_configured_reports(exp_copy: Path):
    (exp_copy / "mine.rq").write_text(
        "SELECT ?variant (COUNT(?r) AS ?runs) WHERE "
        "{ ?r uplab:variant/uplab:name ?variant } GROUP BY ?variant ORDER BY ?variant"
    )
    config = yaml.safe_load((exp_copy / "experiment.yaml").read_text())
    pivot = {"index": "variant", "columns": "domain", "values": "solved"}
    config["reports"] = [
        {"name": "mine", "query": "mine.rq", "render": "csv"},
        {"name": "pivot", "query": "builtin:coverage", "render": "latex", "options": {"pivot": pivot}},
        {"name": "vs", "query": "builtin:runtimes", "render": "scatter",
         "options": {"x": "good", "y": "bad", "value": "cpu_time"}},
        {"name": "bars", "query": "builtin:coverage", "render": "bar",
         "options": {"x": "domain", "y": "solved", "hue": "variant"}},
        {"name": "lines", "query": "builtin:anytime", "render": "line",
         "options": {"x": "cpu_time", "y": "cost", "hue": "variant"}},
    ]
    (exp_copy / "experiment.yaml").write_text(yaml.safe_dump(config))
    written = {p.name for p in report(exp_copy)}
    assert written == {
        "mine.csv", "pivot.tex", "vs.pdf", "vs.png", "bars.pdf", "bars.png",
        "lines.pdf", "lines.png",
    }
    assert (exp_copy / "reports" / "mine.csv").read_text().split() == ["variant,runs", "bad,4", "good,4"]
    tex = (exp_copy / "reports" / "pivot.tex").read_text()
    assert "TOTAL & plain & route" in tex and "good & 4 & 1 & 3" in tex


def test_renderer_errors(exp: Path, tmp_path: Path):
    df = query(exp, load_query("builtin:runtimes"))
    renderers = hook("uplab_renderers")
    assert set(renderers) >= {"csv", "latex", "cactus", "scatter", "bar", "line"}
    with pytest.raises(UPLabError):
        renderers["scatter"](df, tmp_path / "s", {})
    with pytest.raises(UPLabError):
        renderers["scatter"](df, tmp_path / "s", {"x": "good", "y": "nobody"})
    with pytest.raises(UPLabError):
        renderers["bar"](df, tmp_path / "b", {"y": "nothing"})
    # an empty table still gives a figure
    assert len(renderers["cactus"](df.iloc[:0], tmp_path / "c", {})) == 2


def test_notebook_exports_to_html(exp_copy: Path):
    notebook = write_notebook(exp_copy)
    assert notebook.parent == exp_copy and "marimo.App" in notebook.read_text()
    subprocess.run([sys.executable, "-m", "marimo", "check", str(notebook)], check=True)
    written = report(exp_copy, names=["errors"], html=True)
    html = exp_copy / "reports" / "report.html"
    assert written[-1] == html
    text = html.read_text()
    assert "Coverage" in text and "boom" in text and "SPARQL" in text
    assert "Traceback" not in text and "marimo-error" not in text


def test_serve_command(exp: Path, tmp_path: Path, monkeypatch):
    command = serve_command(exp, "0.0.0.0", 9999)
    assert Path(command[0]).name == "oxigraph"
    assert command[1:] == [
        "serve-read-only", "--location", str(exp / "store"),
        "--bind", "0.0.0.0:9999", "--union-default-graph",
    ]
    called = []
    monkeypatch.setattr(serve_module.os, "execv", lambda path, argv: called.append(argv))
    serve_module.serve(exp)
    assert called[0][-3:] == ["--bind", "localhost:7878", "--union-default-graph"]
    with pytest.raises(UPLabError):
        serve_command(tmp_path)
    monkeypatch.setattr(serve_module.shutil, "which", lambda *a, **k: None)
    with pytest.raises(UPLabError, match="serve"):
        serve_command(exp)
