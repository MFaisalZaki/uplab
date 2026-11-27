"""
Slurm submitter: a controller job submits job arrays, one run per array
index, and refills the queue until every run has a result.
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import submitit
from submitit.core.utils import FailedJobError

from up_lab.build import pending
from up_lab.config import ExperimentConfig, UPLabError

DEFAULT_MAX_ARRAY_SIZE = 1000
CONTROLLER_ENV = "UPLAB_SLURM_CONTROLLER"


def max_array_size() -> int:
    """
    :return: the largest job array the cluster accepts (``MaxArraySize - 1``
        from ``scontrol show config``), 1000 when it cannot be asked.
    """
    try:
        out = subprocess.run(
            ["scontrol", "show", "config"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return DEFAULT_MAX_ARRAY_SIZE
    match = re.search(r"^MaxArraySize\s*=\s*(\d+)", out, re.MULTILINE)
    return max(int(match.group(1)) - 1, 1) if match else DEFAULT_MAX_ARRAY_SIZE


def live_jobs() -> Set[str]:
    """
    :return: ids of the user's jobs and job arrays that ``squeue`` lists.
    """
    try:
        out = subprocess.run(
            ["squeue", "--me", "--noheader", "--format=%F"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise UPLabError(f"cannot list queued jobs with squeue: {error}") from error
    return set(out.split())


def read_ledger(exp_dir: Path) -> Dict[str, Any]:
    """
    :param exp_dir: the experiment directory.
    :return: ``slurm/ledger.json``: the accepted ``arrays`` (``job_id``,
        ``run_ids``, ``submitted_at``) and the last ``controller`` job.
    """
    path = exp_dir / "slurm" / "ledger.json"
    if path.exists():
        ledger: Dict[str, Any] = json.loads(path.read_text())
        return ledger
    return {"arrays": [], "controller": None}


def _write_ledger(exp_dir: Path, ledger: Dict[str, Any]) -> None:
    path = exp_dir / "slurm" / "ledger.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _executor(
    exp_dir: Path, config: ExperimentConfig, **more: Any
) -> submitit.SlurmExecutor:
    slurm = config.executor.slurm
    executor = submitit.SlurmExecutor(folder=exp_dir / "slurm" / "submitit")
    parameters: Dict[str, Any] = {
        "job_name": f"uplab-{config.name}",
        "partition": slurm.partition,
        "qos": slurm.qos,
        "account": slurm.account,
        "additional_parameters": slurm.extra or None,
        **more,
    }
    executor.update_parameters(**{k: v for k, v in parameters.items() if v is not None})
    return executor


def _submit_arrays(
    exp_dir: Path, config: ExperimentConfig, ledger: Dict[str, Any], live: Set[str]
) -> Tuple[int, List[str], List[str]]:
    """
    Submits job arrays of pending runs until all are queued, `max_queued` is
    reached or `sbatch` refuses.

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    :param ledger: the ledger, updated and written after each accepted array.
    :param live: ids of the jobs `squeue` lists.
    :return: the number of queued runs, the runs still to submit, and the
        runs given up on (submitted `max_submissions` times, never a result).
    """
    slurm, limits = config.executor.slurm, config.limits
    in_live = {r for a in ledger["arrays"] if a["job_id"] in live for r in a["run_ids"]}
    attempts = Counter(r for a in ledger["arrays"] for r in a["run_ids"])
    unfinished = pending(exp_dir)
    waiting = [r for r in unfinished if r not in in_live]
    todo = [r for r in waiting if attempts[r] < slurm.max_submissions]
    lost = [r for r in waiting if attempts[r] >= slurm.max_submissions]
    queued = len(in_live.intersection(unfinished))
    size = slurm.max_array_size or max_array_size()
    while todo:
        room = size if slurm.max_queued is None else slurm.max_queued - queued
        chunk = todo[: min(size, room)]
        if not chunk:
            break
        executor = _executor(
            exp_dir,
            config,
            time=math.ceil((limits.time + slurm.time_headroom) / 60),
            mem=f"{limits.memory_mb + slurm.memory_headroom_mb}M",
            cpus_per_task=limits.cores,
            array_parallelism=slurm.array_parallelism,
        )
        launchers = [str(exp_dir / "runs" / run_id / "run.sh") for run_id in chunk]
        try:
            jobs = executor.map_array(
                submitit.helpers.CommandFunction(["bash"]), launchers
            )
        except FailedJobError as error:
            print(f"sbatch refused an array of {len(chunk)} runs: {error}")
            break
        ledger["arrays"].append(
            {
                "job_id": jobs[0].job_id.split("_")[0],
                "run_ids": chunk,
                "submitted_at": _now(),
            }
        )
        _write_ledger(exp_dir, ledger)
        todo = todo[len(chunk) :]
        queued += len(chunk)
    return queued, todo, lost


def _submit_controller(exp_dir: Path, config: ExperimentConfig) -> bool:
    """
    Queues the controller job: `uplab run` with `UPLAB_SLURM_CONTROLLER` set.

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    :return: whether `sbatch` accepted it.
    """
    try:
        # No srun: a job step cannot submit jobs on every cluster.
        executor = _executor(
            exp_dir, config, time=config.executor.slurm.controller_time, use_srun=False
        )
        command = ["env", f"{CONTROLLER_ENV}=1", sys.executable, "-m", "up_lab.cli"]
        job = executor.submit(
            submitit.helpers.CommandFunction([*command, "run", str(exp_dir)])
        )
    except FailedJobError as error:
        print(
            f"Could not queue the controller job ({error}).\n"
            f"Run it yourself, e.g. in tmux on the login node:\n"
            f"  {CONTROLLER_ENV}=1 uplab run {exp_dir}"
        )
        return False
    ledger = read_ledger(exp_dir)
    ledger["controller"] = job.job_id
    _write_ledger(exp_dir, ledger)
    print(f"controller job {job.job_id} submits the runs and then collects")
    return True


def control(exp_dir: Path, config: ExperimentConfig) -> None:
    """
    The controller loop: submit arrays, wait, refill; collect when no run is
    queued or left to submit. Inside a Slurm job it resubmits itself shortly
    before its own time limit.

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    """
    slurm = config.executor.slurm
    budget = slurm.controller_time * 60
    margin = min(max(300, 2 * slurm.poll_interval), budget / 2)
    in_job = "SLURM_JOB_ID" in os.environ
    deadline = time.monotonic() + budget - margin
    while True:
        try:
            live = live_jobs()
        except UPLabError as error:
            print(f"{error}; trying again")
            time.sleep(slurm.poll_interval)
            continue
        queued, todo, lost = _submit_arrays(exp_dir, config, read_ledger(exp_dir), live)
        print(f"{_now()} {queued} runs queued, {len(todo)} not yet queued", flush=True)
        if not queued and not todo:
            if lost:
                print(
                    f"{len(lost)} runs left no result after "
                    f"{slurm.max_submissions} submissions: {' '.join(lost)}"
                )
            from up_lab.collect import collect

            collect(exp_dir)
            return
        if in_job and time.monotonic() >= deadline:
            _submit_controller(exp_dir, config)
            return
        time.sleep(slurm.poll_interval)


def run_slurm(exp_dir: Path, config: ExperimentConfig) -> None:
    """
    Queues the controller job, a small Slurm job that submits the pending
    runs as job arrays and keeps the queue filled. With the environment
    variable `UPLAB_SLURM_CONTROLLER` set, this call is the controller.

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    """
    exp_dir = exp_dir.resolve()
    (exp_dir / "slurm").mkdir(parents=True, exist_ok=True)
    if os.environ.get(CONTROLLER_ENV):
        control(exp_dir, config)
        return
    controller = read_ledger(exp_dir).get("controller")
    if controller in live_jobs():
        print(f"controller job {controller} is already queued")
    elif not pending(exp_dir):
        print("every run has a result")
    else:
        _submit_controller(exp_dir, config)
