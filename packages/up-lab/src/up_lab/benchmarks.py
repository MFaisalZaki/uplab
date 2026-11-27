"""Benchmark sources: fetch and pin, discover tasks, select them."""

import hashlib
import os
import re
import subprocess
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from up_lab.config import BenchmarkConfig, UPLabError, is_url
from up_lab.plugins import hook

MANIFEST = "uplab-benchmark.yaml"
PDDL_SUFFIXES = (".pddl", ".hddl")


def _git(*args: str, cwd: Optional[Path] = None) -> str:
    try:
        done = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise UPLabError(f"git {' '.join(args)} failed: {detail.strip()}") from error
    return done.stdout.strip()


def fetch(benchmark: BenchmarkConfig, exp_dir: Path) -> Dict[str, Any]:
    """
    Makes a benchmark source available locally. A local path without ``rev``
    is used in place; anything else is cloned into
    ``<exp-dir>/benchmarks/<name>`` and checked out at ``rev``.

    :param benchmark: the benchmark source.
    :param exp_dir: the experiment directory.
    :return: the ``lock.json`` entry: ``source``, ``rev``, ``path``, ``commit``.
    """
    entry: Dict[str, Any] = {"source": benchmark.source, "rev": benchmark.rev}
    if benchmark.rev is None and not is_url(benchmark.source):
        path = Path(benchmark.source)
        if not path.is_dir():
            raise UPLabError(f"benchmark {benchmark.name}: {path} is not a directory")
        commit: Optional[str] = None  # stays so outside git or with changes
        try:
            if not _git("status", "--porcelain", "--", ".", cwd=path):
                commit = _git("rev-parse", "HEAD", cwd=path)
        except UPLabError:
            pass
        return {**entry, "path": str(path), "commit": commit}
    path = (exp_dir / "benchmarks" / benchmark.name).resolve()
    if not (path / ".git").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", benchmark.source, str(path))
    elif benchmark.rev is not None:
        _git("fetch", "--tags", "origin", cwd=path)
    if benchmark.rev is not None:
        try:  # a branch name means the remote branch, not a stale local one
            _git("checkout", "--detach", f"origin/{benchmark.rev}", cwd=path)
        except UPLabError:
            _git("checkout", "--detach", benchmark.rev, cwd=path)
    return {**entry, "path": str(path), "commit": _git("rev-parse", "HEAD", cwd=path)}


def _natural(text: str) -> List[Any]:
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", text)]


def _task(domain: str, problem: str, fmt: str, **files: Optional[str]) -> Dict:
    return {"domain": domain, "problem": problem, "format": fmt, **files}


def _domain_name(root: Path, directory: Path) -> str:
    return directory.relative_to(root).as_posix() if directory != root else root.name


def pddl_layout(root: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Directories holding PDDL/HDDL domain and problem files: either one domain
    file for all problems, or one per problem (``pNN-domain``, ``domain_pNN``).

    :param root: the benchmark root.
    :return: task dicts, or ``None`` when no directory matches.
    """
    tasks = []
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        directory = Path(current)
        names = {Path(f).stem: f for f in files if f.endswith(PDDL_SUFFIXES)}
        domains = {stem: f for stem, f in names.items() if "domain" in stem.lower()}
        shared = names.get("domain") or (
            next(iter(domains.values())) if len(domains) == 1 else None
        )
        for stem in sorted(set(names) - set(domains), key=_natural):
            candidates = (f"{stem}-domain", f"{stem}_domain", f"domain-{stem}")
            own = next(
                (domains[c] for c in (*candidates, f"domain_{stem}") if c in domains),
                None,
            )
            domain_file = own or shared
            if domain_file is not None:
                rel = directory.relative_to(root)
                tasks.append(
                    _task(
                        _domain_name(root, directory),
                        stem,
                        "pddl",
                        domain_file=(rel / domain_file).as_posix(),
                        problem_file=(rel / names[stem]).as_posix(),
                    )
                )
    return tasks or None


def anml_layout(root: Path) -> Optional[List[Dict[str, Any]]]:
    """
    Every ``*.anml`` file is a task; its directory is the domain.

    :param root: the benchmark root.
    :return: task dicts, or ``None`` when there is no ANML file.
    """
    tasks = [
        _task(
            _domain_name(root, path.parent),
            path.stem,
            "anml",
            problem_file=path.relative_to(root).as_posix(),
        )
        for path in sorted(root.rglob("*.anml"), key=lambda p: _natural(str(p)))
    ]
    return tasks or None


def _raw_tasks(benchmark: BenchmarkConfig, root: Path) -> List[Dict[str, Any]]:
    manifest = Path(benchmark.manifest) if benchmark.manifest else root / MANIFEST
    if benchmark.manifest or manifest.is_file():
        try:
            return list(yaml.safe_load(manifest.read_text())["tasks"])
        except (OSError, yaml.YAMLError, KeyError, TypeError) as error:
            raise UPLabError(f"invalid manifest {manifest}: {error!r}") from error
    layouts = hook("uplab_benchmark_layouts")
    if benchmark.layout != "auto":
        if benchmark.layout not in layouts:
            raise UPLabError(
                f"benchmark {benchmark.name}: unknown layout {benchmark.layout!r}"
                f" (known: {sorted(layouts)})"
            )
        layouts = {benchmark.layout: layouts[benchmark.layout]}
    for detect in layouts.values():
        tasks = detect(root)
        if tasks:
            return list(tasks)
    raise UPLabError(f"benchmark {benchmark.name}: no tasks found in {root}")


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise UPLabError(f"cannot read task file {path}: {error}") from error


def find_tasks(benchmark: BenchmarkConfig, lock: Dict[str, Any]) -> List[Dict]:
    """
    Lists the selected tasks of a fetched benchmark source.

    :param benchmark: the benchmark source.
    :param lock: its ``lock.json`` entry, as returned by :func:`fetch`.
    :return: task dicts exactly as in ``run.json``.
    """
    root = Path(lock["path"]) / (benchmark.subdir or "")
    if not root.is_dir():
        raise UPLabError(f"benchmark {benchmark.name}: {root} is not a directory")
    tasks: List[Dict[str, Any]] = []
    per_domain: Dict[str, int] = {}
    for raw in _raw_tasks(benchmark, root):
        try:
            domain, problem = str(raw["domain"]), str(raw["problem"])
            fmt = raw.get("format", "pddl")
        except (KeyError, TypeError, AttributeError) as error:
            raise UPLabError(f"benchmark {benchmark.name}: bad task {raw!r}") from error
        key = f"{domain}/{problem}"
        if benchmark.include and not any(fnmatch(key, p) for p in benchmark.include):
            continue
        if any(fnmatch(key, p) for p in benchmark.exclude):
            continue
        per_domain[domain] = per_domain.get(domain, 0) + 1
        if benchmark.max_per_domain and per_domain[domain] > benchmark.max_per_domain:
            continue
        files = {
            k: str((root / raw[k]).resolve()) if raw.get(k) else None
            for k in ("domain_file", "problem_file")
        }
        tasks.append(
            {
                "id": f"{benchmark.name}/{key}",
                "source": benchmark.name,
                "source_rev": lock.get("commit"),
                "domain": domain,
                "problem": problem,
                "format": fmt,
                **files,
                "files_sha256": {k: _sha256(Path(f)) for k, f in files.items() if f},
                "factory": raw.get("factory"),
                "factory_args": raw.get("factory_args") or {},
                "python_path": str(root.resolve()) if fmt == "python" else None,
                "reference_cost": raw.get("reference_cost"),
                "tags": list(raw.get("tags") or []),
            }
        )
    return tasks
