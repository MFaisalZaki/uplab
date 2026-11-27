"""Planner environments: one uv virtual environment per planner."""

import hashlib
import json
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse
from urllib.request import url2pathname

from up_lab.config import PlannerConfig, UPLabError

WORKER = "up-lab-worker"
STAMP = "uplab-env.json"


def worker_requirement() -> List[str]:
    """
    The pip arguments installing the same ``up-lab-worker`` as the one the
    orchestrator runs with: its local directory or VCS commit when it was
    installed that way, its exact version otherwise.

    :return: arguments for ``uv pip install``.
    """
    dist = metadata.distribution(WORKER)
    direct = json.loads(dist.read_text("direct_url.json") or "{}")
    url = direct.get("url", "")
    if "dir_info" in direct:
        path = url2pathname(urlparse(url).path)
        return ["--editable", path] if direct["dir_info"].get("editable") else [path]
    if "vcs_info" in direct:
        vcs = direct["vcs_info"]
        requirement = f"{WORKER} @ {vcs['vcs']}+{url}@{vcs['commit_id']}"
        if direct.get("subdirectory"):
            requirement += f"#subdirectory={direct['subdirectory']}"
        return [requirement]
    return [f"{WORKER}=={dist.version}"]


def _uv(*args: str) -> str:
    try:
        done = subprocess.run(["uv", *args], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise UPLabError(f"uv {' '.join(args)} failed:\n{detail.strip()}") from error
    return done.stdout


def _packages(freeze: str) -> Dict[str, str]:
    packages = {}
    for line in freeze.splitlines():
        for separator in ("==", " @ "):
            if separator in line:
                name, version = line.split(separator, 1)
                packages[name.strip()] = version.strip()
                break
    return packages


def build_env(planner: PlannerConfig, exp_dir: Path, force: bool = False) -> Dict:
    """
    Creates ``<exp-dir>/envs/<planner>`` and installs the planner and the
    worker into it. An environment built from the same inputs is kept.

    :param planner: the planner to install.
    :param exp_dir: the experiment directory.
    :param force: rebuild even when the inputs are unchanged.
    :return: the ``lock.json`` entry: ``python``, ``source``,
        ``freeze_sha256``, ``packages``, ``inputs_sha256``.
    """
    env_dir = (exp_dir / "envs" / planner.name).resolve()
    python = env_dir / "bin" / "python"
    worker = worker_requirement()
    inputs = hashlib.sha256(
        json.dumps([planner.source, planner.python, worker]).encode()
    ).hexdigest()
    stamp = env_dir / STAMP
    if not force and stamp.is_file() and python.exists():
        entry: Dict[str, Any] = json.loads(stamp.read_text())
        if entry.get("inputs_sha256") == inputs:
            return entry
    if env_dir.exists():
        shutil.rmtree(env_dir)
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    version = ["--python", planner.python] if planner.python else []
    _uv("venv", *version, str(env_dir))
    _uv("pip", "install", "--python", str(python), *planner.source, *worker)
    freeze = _uv("pip", "freeze", "--python", str(python))
    entry = {
        "python": str(python),
        "source": planner.source,
        "freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
        "packages": _packages(freeze),
        "inputs_sha256": inputs,
    }
    stamp.write_text(json.dumps(entry, indent=2))
    return entry
