import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

from conftest import raw_config
from test_local import STUB
from up_lab import envs
from up_lab.cli import app
from up_lab.experiment import Experiment

runner = CliRunner()


@pytest.fixture
def yaml_file(tmp_path: Path, bench: Path, monkeypatch: Any) -> Path:
    env = {"python": sys.executable, "source": [], "freeze_sha256": "", "packages": {}}
    monkeypatch.setattr(envs, "build_env", lambda planner, exp_dir, force=False: env)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_config(Path("bench"))))
    return path


def test_resolve_build_run_status(yaml_file: Path, tmp_path: Path) -> None:
    result = runner.invoke(app, ["resolve", str(yaml_file)])
    assert result.exit_code == 0, result.output
    exp_dir = tmp_path / "experiments" / "demo"
    lock = json.loads((exp_dir / "lock.json").read_text())
    assert lock["benchmarks"]["b"]["path"] == str(tmp_path / "bench")
    assert lock["planners"]["pp"]["python"] == sys.executable
    assert (exp_dir / "experiment.yaml").exists()

    # Later steps accept the YAML file or the experiment directory.
    assert "6 runs" in runner.invoke(app, ["build", str(yaml_file)]).output
    for launcher in exp_dir.glob("runs/*/run.sh"):
        launcher.write_text(STUB)
    assert runner.invoke(app, ["run", str(exp_dir)]).exit_code == 0
    status = runner.invoke(app, ["status", str(exp_dir / "experiment.yaml")])
    assert status.output.split() == ["SOLVED", "6"]


def test_dir_option_and_errors(yaml_file: Path, tmp_path: Path) -> None:
    other = tmp_path / "other"
    assert (
        runner.invoke(app, ["resolve", str(yaml_file), "--dir", str(other)]).exit_code
        == 0
    )
    assert (other / "lock.json").exists()
    assert Experiment.open(other).exp_dir == other

    yaml_file.write_text("name: demo\nunknown: 1\n")
    with pytest.raises(SystemExit):
        app(["build", str(yaml_file)])


def test_results_steps_are_imported_lazily(yaml_file: Path, monkeypatch: Any) -> None:
    calls = []
    for module, function in (
        ("collect", "collect"),
        ("analysis", "analyze"),
        ("reports", "report"),
        ("notebook", "write_notebook"),
        ("serve", "serve"),
    ):
        fake = type(sys)(f"up_lab.{module}")
        setattr(fake, function, lambda *a, _n=function, **k: calls.append((_n, a, k)))
        monkeypatch.setitem(sys.modules, f"up_lab.{module}", fake)
    exp = Experiment.open(yaml_file)
    exp.collect(jobs=2), exp.analyze(), exp.report(html=True), exp.notebook()
    exp.serve("0.0.0.0", 1234)
    assert calls == [
        ("collect", (exp.exp_dir,), {"jobs": 2}),
        ("analyze", (exp.exp_dir,), {"names": None}),
        ("report", (exp.exp_dir,), {"names": None, "html": True}),
        ("write_notebook", (exp.exp_dir,), {}),
        ("serve", (exp.exp_dir, "0.0.0.0", 1234), {}),
    ]
