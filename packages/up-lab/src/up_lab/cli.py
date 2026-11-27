"""Command line ``uplab``."""

from pathlib import Path
from typing import Any, List, Optional

import typer
from unified_planning.exceptions import UPException

from up_lab.experiment import Experiment


class _App(typer.Typer):
    """A typer application printing up-lab errors without a traceback."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().__call__(*args, **kwargs)
        except UPException as error:
            typer.secho(str(error), fg="red", err=True)
            raise SystemExit(1)


app = _App(
    help="Benchmarking experiments for Unified Planning engines.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

EXPERIMENT = typer.Argument(..., help="experiment.yaml or experiment directory")
DIR = typer.Option(None, "--dir", help="experiment directory")
NAMES = typer.Option(None, "--name", help="restrict to these names (repeatable)")


def _open(path: Path, exp_dir: Optional[Path] = None) -> Experiment:
    return Experiment.open(path, exp_dir)


def _status(exp: Experiment) -> None:
    for state, count in sorted(exp.status().items()):
        typer.echo(f"{state:12} {count}")


@app.command()
def resolve(
    experiment: Path = typer.Argument(..., help="experiment.yaml"),
    exp_dir: Optional[Path] = DIR,
    force: bool = typer.Option(False, help="rebuild the planner environments"),
) -> None:
    """Clone benchmark sources, build planner environments, write lock.json."""
    exp = _open(experiment, exp_dir)
    lock = exp.resolve(force=force)
    for name, entry in lock["benchmarks"].items():
        typer.echo(f"benchmark {name}: {entry['path']} @ {entry['commit']}")
    for name, entry in lock["planners"].items():
        typer.echo(f"planner {name}: {entry['python']}")
    typer.echo(f"experiment directory: {exp.exp_dir}")


@app.command()
def build(experiment: Path = EXPERIMENT, exp_dir: Optional[Path] = DIR) -> None:
    """Expand the run matrix and write the run directories."""
    exp = _open(experiment, exp_dir)
    typer.echo(f"{len(exp.build())} runs in {exp.exp_dir / 'runs'}")


@app.command()
def run(experiment: Path = EXPERIMENT, exp_dir: Optional[Path] = DIR) -> None:
    """Execute the pending runs through the configured executor."""
    _open(experiment, exp_dir).run()


@app.command()
def status(experiment: Path = EXPERIMENT, exp_dir: Optional[Path] = DIR) -> None:
    """Print the number of runs per state."""
    _status(_open(experiment, exp_dir))


@app.command()
def collect(
    experiment: Path = EXPERIMENT,
    exp_dir: Optional[Path] = DIR,
    jobs: int = typer.Option(1, "--jobs", "-j", help="validation processes"),
) -> None:
    """Validate plans, write the records and load the store."""
    _open(experiment, exp_dir).collect(jobs=jobs)


@app.command()
def analyze(
    experiment: Path = EXPERIMENT,
    exp_dir: Optional[Path] = DIR,
    names: Optional[List[str]] = NAMES,
) -> None:
    """Run the analyses, each into its own named graph."""
    _open(experiment, exp_dir).analyze(names=names or None)


@app.command()
def report(
    experiment: Path = EXPERIMENT,
    exp_dir: Optional[Path] = DIR,
    names: Optional[List[str]] = NAMES,
    html: bool = typer.Option(False, help="also write an HTML overview"),
) -> None:
    """Write the reports: saved queries rendered to CSV, LaTeX, PDF."""
    _open(experiment, exp_dir).report(names=names or None, html=html)


@app.command()
def notebook(experiment: Path = EXPERIMENT, exp_dir: Optional[Path] = DIR) -> None:
    """Write a marimo notebook on the store."""
    typer.echo(_open(experiment, exp_dir).notebook())


@app.command()
def serve(
    experiment: Path = EXPERIMENT,
    exp_dir: Optional[Path] = DIR,
    host: str = "127.0.0.1",
    port: int = 7878,
) -> None:
    """Serve the store as a SPARQL endpoint."""
    _open(experiment, exp_dir).serve(host, port)


@app.command(name="all")
def all_steps(experiment: Path = EXPERIMENT, exp_dir: Optional[Path] = DIR) -> None:
    """resolve, build, run, collect, analyze and report (local executor)."""
    exp = _open(experiment, exp_dir)
    if exp.config.executor.kind != "local":
        typer.secho("'uplab all' needs the local executor", fg="red", err=True)
        raise typer.Exit(1)
    exp.resolve()
    exp.build()
    exp.run()
    _status(exp)
    exp.collect()
    exp.analyze()
    exp.report()


if __name__ == "__main__":
    app()
