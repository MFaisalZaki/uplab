"""The driver of a run: ``python -m up_lab_worker <run-dir>``."""

import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

from up_lab_worker import __version__

KILL_GRACE = 2.0
"""Seconds given to an engine after the time limit to return by itself."""

MEMOUT_MARKERS = ("MemoryError", "std::bad_alloc", "Cannot allocate memory")


def machine_info() -> Dict[str, Any]:
    """Returns the `machine` object of `result.json`."""
    cpu_model: Optional[str] = platform.processor() or None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    try:
        memory_mb: Optional[int] = (
            os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2**20
        )
    except (ValueError, OSError):
        memory_mb = None
    try:
        up_version: Optional[str] = metadata.version("unified-planning")
    except metadata.PackageNotFoundError:
        up_version = None
    return {
        "hostname": platform.node(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "memory_mb": memory_mb,
        "os": f"{platform.system()}-{platform.release()}",
        "python": platform.python_version(),
        "unified_planning": up_version,
        "up_lab_worker": __version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _load(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _plan_records(run_dir: Path) -> List[Dict[str, Any]]:
    records = []
    try:
        for line in (run_dir / "plans.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:  # a line cut by a kill
                break
    except OSError:
        pass
    return records


def _tail(path: Path, size: int = 4000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-size:].strip()
    except OSError:
        return ""


def _kill(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def empty_result(run: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a `result.json` object with every key present and unknown values."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema": "uplab/result/1",
        "run_id": run.get("run_id"),
        "status": "ERROR",
        "up_status": None,
        "exit_code": None,
        "signal": None,
        "error": None,
        "started_at": now,
        "ended_at": now,
        "resources": {"wall_time": None, "cpu_time": None, "peak_memory_mb": None},
        "phases": {
            "parse": None,
            "compile": [],
            "solve": None,
            "first_plan": None,
            "last_plan": None,
        },
        "problem": None,
        "plans": {"count": 0, "claimed_optimal": None, "claimed_complete": None},
        "engine": {"name": None, "metrics": {}, "log_messages": []},
        "machine": machine_info(),
    }


def drive(run_dir: Path) -> Dict[str, Any]:
    """
    Runs the solving process of a run under its limits.

    :param run_dir: The run directory, containing `run.json`.
    :return: The content of `result.json`.
    """
    run = _load(run_dir / "run.json")
    limits = run.get("limits") or {}
    time_limit, memory_mb = limits.get("time"), limits.get("memory_mb")
    for name in ("solve.json", "plans.jsonl", "events.jsonl", "result.json"):
        (run_dir / name).unlink(missing_ok=True)
    if (run_dir / "plans").is_dir():
        for old in (run_dir / "plans").glob("*.plan"):
            old.unlink()

    def set_limits() -> None:
        if memory_mb is not None:
            size = int(memory_mb * 2**20)
            resource.setrlimit(resource.RLIMIT_AS, (size, size))
        if time_limit is not None:
            cpu = math.ceil(time_limit + KILL_GRACE)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 5))

    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_at = datetime.now(timezone.utc)
    start = time.time()
    limit_hit: Optional[str] = None
    with (
        open(run_dir / "stdout.log", "wb") as out,
        open(run_dir / "stderr.log", "wb") as err,
    ):
        process = subprocess.Popen(
            [sys.executable, "-m", "up_lab_worker.solve", str(run_dir)],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            preexec_fn=set_limits,
            env={**os.environ, "UP_LAB_START": repr(start)},
        )
        try:
            process.wait(None if time_limit is None else time_limit + KILL_GRACE)
        except subprocess.TimeoutExpired:
            limit_hit = "wall_time"
        _kill(process.pid)  # also removes processes the solver left behind
        process.wait()
    wall_time = time.time() - start
    ended_at = datetime.now(timezone.utc)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_time = after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
    peak_memory_mb = after.ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
    code = process.returncode
    if limit_hit is None and code == -signal.SIGXCPU:
        limit_hit = "cpu_time"

    solve = _load(run_dir / "solve.json")
    plans = _plan_records(run_dir)
    status, error = solve.get("status"), solve.get("error")
    stderr = _tail(run_dir / "stderr.log")
    if limit_hit is not None:
        status, error = "TIMEOUT", None
    elif status is None:
        near_limit = memory_mb is not None and peak_memory_mb >= 0.9 * memory_mb
        if near_limit or any(marker in stderr for marker in MEMOUT_MARKERS):
            status, limit_hit = "MEMOUT", "memory"
        else:
            status = "ERROR"
            error = stderr or f"The solving process ended with return code {code}"
    elif status in ("ERROR", "UNSOLVABLE") and time_limit is not None:
        # an engine whose own subprocess was stopped by the CPU limit
        if cpu_time >= time_limit + KILL_GRACE:
            status, error, limit_hit = "TIMEOUT", None, "cpu_time"
    if status == "MEMOUT":
        limit_hit = "memory"
    if limit_hit is not None:
        with open(run_dir / "events.jsonl", "a", encoding="utf-8") as events:
            record = {"t": round(wall_time, 6), "event": "limit", "kind": limit_hit}
            events.write(json.dumps(record) + "\n")

    def stamp(record: Dict[str, Any]) -> Dict[str, Any]:
        return {"wall": record.get("wall_time"), "cpu": record.get("cpu_time")}

    result = empty_result(run)
    phases, engine = solve.get("phases") or {}, solve.get("engine") or {}
    result.update(
        status=status,
        up_status=solve.get("up_status"),
        exit_code=code if code >= 0 else None,
        signal=-code if code < 0 else None,
        error=error,
        started_at=started_at.isoformat(timespec="seconds"),
        ended_at=ended_at.isoformat(timespec="seconds"),
        resources={
            "wall_time": round(wall_time, 3),
            "cpu_time": round(cpu_time, 3),
            "peak_memory_mb": round(peak_memory_mb, 1),
        },
        problem=solve.get("problem"),
    )
    result["phases"].update(
        parse=phases.get("parse"),
        compile=phases.get("compile") or [],
        solve=phases.get("solve"),
        first_plan=stamp(plans[0]) if plans else None,
        last_plan=stamp(plans[-1]) if plans else None,
    )
    result["plans"].update({**(solve.get("plans") or {}), "count": len(plans)})
    result["engine"].update({k: v for k, v in engine.items() if v is not None})
    return result


def main(argv: Optional[List[str]] = None) -> int:
    """
    Runs a run directory and writes its `result.json`.

    :param argv: The command-line arguments, defaults to `sys.argv[1:]`.
    :return: The exit code: 0 whenever `result.json` was written.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m up_lab_worker <run-dir>", file=sys.stderr)
        return 2
    run_dir = Path(argv[0]).resolve()
    try:
        result = drive(run_dir)
    except Exception as error:  # result.json is written whatever happens
        result = empty_result(_load(run_dir / "run.json"))
        result["error"] = f"{type(error).__name__}: {error}"
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, run_dir / "result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
