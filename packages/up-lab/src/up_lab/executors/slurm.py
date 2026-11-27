"""Slurm submitter: chained job arrays, one run per array index."""

import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import submitit
from submitit.core.utils import FailedJobError

from up_lab.build import pending
from up_lab.config import ExperimentConfig, UPLabError

DEFAULT_MAX_ARRAY_SIZE = 1000


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
        ``run_ids``, ``submitted_at``) and the last ``continuation`` job.
    """
    path = exp_dir / "slurm" / "ledger.json"
    if path.exists():
        ledger: Dict[str, Any] = json.loads(path.read_text())
        return ledger
    return {"arrays": [], "continuation": None}


def _write_ledger(exp_dir: Path, ledger: Dict[str, Any]) -> None:
    path = exp_dir / "slurm" / "ledger.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ledger, indent=2))
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _executor(
    exp_dir: Path, config: ExperimentConfig, dependency: Optional[str], **more: Any
) -> submitit.SlurmExecutor:
    slurm = config.executor.slurm
    executor = submitit.SlurmExecutor(folder=exp_dir / "slurm" / "submitit")
    parameters: Dict[str, Any] = {
        "job_name": f"uplab-{config.name}",
        "partition": slurm.partition,
        "qos": slurm.qos,
        "account": slurm.account,
        "dependency": dependency,
        "additional_parameters": slurm.extra or None,
        **more,
    }
    executor.update_parameters(**{k: v for k, v in parameters.items() if v is not None})
    return executor


def run_slurm(exp_dir: Path, config: ExperimentConfig) -> None:
    """
    Queues the pending runs as chained job arrays, then a continuation job
    that calls ``uplab run`` again (``uplab collect`` once all is queued).

    :param exp_dir: the experiment directory.
    :param config: the experiment.
    """
    exp_dir = exp_dir.resolve()
    slurm, limits = config.executor.slurm, config.limits
    (exp_dir / "slurm").mkdir(parents=True, exist_ok=True)
    ledger = read_ledger(exp_dir)
    live = live_jobs()
    live_arrays = [a for a in ledger["arrays"] if a["job_id"] in live]
    in_live = {run_id for a in live_arrays for run_id in a["run_ids"]}
    unfinished = pending(exp_dir)
    todo = [run_id for run_id in unfinished if run_id not in in_live]
    queued = len(in_live.intersection(unfinished))
    last: Optional[str] = live_arrays[-1]["job_id"] if live_arrays else None
    size = slurm.max_array_size or max_array_size()
    submitted = 0
    while todo:
        room = size if slurm.max_queued is None else slurm.max_queued - queued
        chunk = todo[: min(size, room)]
        if not chunk:
            print(f"max_queued={slurm.max_queued} reached")
            break
        chained = last is not None and slurm.dependency != "none"
        executor = _executor(
            exp_dir,
            config,
            f"{slurm.dependency}:{last}" if chained else None,
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
        last = jobs[0].job_id.split("_")[0]
        ledger["arrays"].append(
            {"job_id": last, "run_ids": chunk, "submitted_at": _now()}
        )
        _write_ledger(exp_dir, ledger)
        todo = todo[len(chunk) :]
        queued += len(chunk)
        submitted += len(chunk)
    print(f"{submitted} runs submitted, {queued} queued, {len(todo)} not yet queued")
    if not submitted and not todo:
        return
    previous = ledger.get("continuation")
    if previous in live and previous != os.environ.get("SLURM_JOB_ID"):
        print(f"continuation job {previous} is already queued")
        return
    command = "run" if todo else "collect"
    try:
        if last is None:
            raise FailedJobError("no accepted job array to depend on")
        # No srun: a job step cannot submit jobs on every cluster.
        executor = _executor(
            exp_dir, config, f"afterany:{last}", time=60, use_srun=False
        )
        job = executor.submit(
            submitit.helpers.CommandFunction(
                [sys.executable, "-m", "up_lab.cli", command, str(exp_dir)]
            )
        )
    except FailedJobError as error:
        print(
            f"Could not queue the continuation job ({error}).\n"
            f"Re-run 'uplab {command} {exp_dir}' yourself"
            + (" once the queue has room." if todo else " once the jobs have ended.")
        )
        return
    ledger["continuation"] = job.job_id
    _write_ledger(exp_dir, ledger)
    print(f"continuation job {job.job_id} will run 'uplab {command}'")
