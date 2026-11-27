"""Local executor: a pool of subprocesses on this machine."""

import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from up_lab.build import pending
from up_lab.config import ExperimentConfig


def _run(run_dir: Path) -> bool:
    done = subprocess.run(
        [str(run_dir / "run.sh")],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    finished = (run_dir / "result.json").exists()
    if not finished:  # the worker could not even start: keep what it said
        (run_dir / "driver.log").write_bytes(done.stdout)
    return finished


def run_local(exp_dir: Path, config: ExperimentConfig) -> None:
    """
    Runs every pending run's ``run.sh``, ``executor.workers`` at a time.

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    """
    run_dirs = [exp_dir / "runs" / run_id for run_id in pending(exp_dir)]
    with ThreadPoolExecutor(max_workers=config.executor.workers) as pool:
        finished = list(pool.map(_run, run_dirs))
    failed = finished.count(False)
    print(f"{len(run_dirs)} runs executed, {failed} without a result")
    if failed:
        print("see driver.log in the run directories of the runs without a result")
