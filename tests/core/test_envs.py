import json
import os
import subprocess
from pathlib import Path
from typing import Any, List

import pytest

from up_lab import envs
from up_lab.config import PlannerConfig


class Dist:
    version = "0.1.0"

    def __init__(self, direct_url: Any) -> None:
        self.direct_url = direct_url

    def read_text(self, name: str) -> Any:
        return None if self.direct_url is None else json.dumps(self.direct_url)


@pytest.mark.parametrize(
    "direct_url, expected",
    [
        (None, ["up-lab-worker==0.1.0"]),
        ({"url": "file:///src/worker", "dir_info": {}}, ["/src/worker"]),
        (
            {"url": "file:///src/worker", "dir_info": {"editable": True}},
            ["--editable", "/src/worker"],
        ),
        (
            {
                "url": "https://example.org/repo.git",
                "vcs_info": {"vcs": "git", "commit_id": "abc123"},
                "subdirectory": "packages/up-lab-worker",
            },
            [
                "up-lab-worker @ git+https://example.org/repo.git@abc123"
                "#subdirectory=packages/up-lab-worker"
            ],
        ),
    ],
)
def test_worker_requirement(
    monkeypatch: Any, direct_url: Any, expected: List[str]
) -> None:
    monkeypatch.setattr(envs.metadata, "distribution", lambda name: Dist(direct_url))
    assert envs.worker_requirement() == expected


def test_build_env_commands_and_reuse(tmp_path: Path, monkeypatch: Any) -> None:
    calls: List[tuple] = []

    def fake_uv(*args: str) -> str:
        calls.append(args)
        if args[0] == "venv":
            (Path(args[-1]) / "bin").mkdir(parents=True)
            (Path(args[-1]) / "bin" / "python").write_text("")
        return "up-pyperplan==1.1.0\nlocal-pkg @ file:///src/pkg\n"

    monkeypatch.setattr(envs, "_uv", fake_uv)
    monkeypatch.setattr(envs, "worker_requirement", lambda: ["up-lab-worker==0.1.0"])
    planner = PlannerConfig(
        name="pp", source=["up-pyperplan", "/src/pkg"], python="3.11"
    )
    entry = envs.build_env(planner, tmp_path)
    python = str(tmp_path / "envs" / "pp" / "bin" / "python")
    assert calls == [
        ("venv", "--python", "3.11", str(tmp_path / "envs" / "pp")),
        (
            "pip",
            "install",
            "--python",
            python,
            "up-pyperplan",
            "/src/pkg",
            "up-lab-worker==0.1.0",
        ),
        ("pip", "freeze", "--python", python),
    ]
    assert entry["python"] == python and entry["source"] == planner.source
    assert entry["packages"] == {
        "up-pyperplan": "1.1.0",
        "local-pkg": "file:///src/pkg",
    }
    assert len(entry["freeze_sha256"]) == 64

    assert envs.build_env(planner, tmp_path) == entry and len(calls) == 3  # reused
    envs.build_env(planner, tmp_path, force=True)
    assert len(calls) == 6
    changed = PlannerConfig(name="pp", source=["up-pyperplan"])
    assert envs.build_env(changed, tmp_path)["source"] == ["up-pyperplan"]
    assert len(calls) == 9 and calls[6] == ("venv", str(tmp_path / "envs" / "pp"))


@pytest.mark.skipif(
    not os.environ.get("UPLAB_NETWORK_TESTS"),
    reason="builds a real uv environment (needs a package index); "
    "set UPLAB_NETWORK_TESTS=1",
)
@pytest.mark.parametrize("with_uv", [True, False])
def test_build_real_env(tmp_path: Path, monkeypatch: Any, with_uv: bool) -> None:
    if not with_uv:  # the standard venv + pip path
        which = envs.shutil.which
        monkeypatch.setattr(
            envs.shutil, "which", lambda name: None if name == "uv" else which(name)
        )
    entry = envs.build_env(PlannerConfig(name="pp", source=["up-pyperplan"]), tmp_path)
    assert "up-pyperplan" in entry["packages"]
    code = "import up_lab_worker, up_pyperplan"
    subprocess.run([entry["python"], "-c", code], check=True)
