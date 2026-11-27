"""Reports: saved queries rendered to tables and figures."""

import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from up_lab.config import UPLabError
from up_lab.notebook import write_notebook
from up_lab.plugins import hook
from up_lab.store import experiment_config, load_query, open_store, query

DEFAULT_REPORTS: List[Dict[str, Any]] = [
    {"name": "coverage", "query": "builtin:coverage", "render": "csv"},
    {"name": "coverage", "query": "builtin:coverage", "render": "latex"},
    {"name": "scores", "query": "builtin:scores", "render": "csv"},
    {"name": "scores", "query": "builtin:scores", "render": "latex"},
    {"name": "runtimes", "query": "builtin:runtimes", "render": "cactus"},
    {"name": "errors", "query": "builtin:errors", "render": "csv"},
]


def export_html(exp_dir: Path) -> Path:
    """
    Exports the notebook of the experiment to `reports/report.html`.

    :param exp_dir: The experiment directory.
    :return: The HTML file.
    """
    output = exp_dir / "reports" / "report.html"
    command = [sys.executable, "-m", "marimo", "export", "html"]
    command += [str(write_notebook(exp_dir)), "-o", str(output), "--force"]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0 or not output.exists():
        raise UPLabError(
            "marimo export failed (is the extra up-lab[notebook] installed?):\n"
            + done.stderr[-2000:]
        )
    return output


def report(
    exp_dir: Union[str, Path], names: Optional[List[str]] = None, html: bool = False
) -> List[Path]:
    """
    Runs the query of every configured report (the default ones when none is
    configured) and renders it under `<exp-dir>/reports/<name>.<ext>`.

    :param exp_dir: The experiment directory.
    :param names: Produce only the reports with these names.
    :param html: Also export the notebook to `reports/report.html`.
    :return: The written files.
    """
    exp_dir = Path(exp_dir)
    reports = experiment_config(exp_dir)["reports"] or DEFAULT_REPORTS
    unknown = set(names or []) - {r["name"] for r in reports}
    if unknown:
        raise UPLabError(f"Unknown reports: {sorted(unknown)}")
    (exp_dir / "reports").mkdir(exist_ok=True)
    renderers = hook("uplab_renderers")
    store = open_store(exp_dir, read_only=True)
    written: List[Path] = []
    for entry in reports:
        if names is not None and entry["name"] not in names:
            continue
        if entry["render"] not in renderers:
            raise UPLabError(
                f"Report {entry['name']}: unknown renderer {entry['render']!r}; "
                f"available: {sorted(renderers)}"
            )
        df = query(store, load_query(entry["query"], exp_dir))
        stem = exp_dir / "reports" / entry["name"]
        written += renderers[entry["render"]](df, stem, entry.get("options") or {})
    del store
    if html:
        written.append(export_html(exp_dir))
    return written
