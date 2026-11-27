"""The Slurm submitter against fake ``sbatch``/``squeue``/``scontrol``."""

import json
import re
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pytest
from submitit.core.utils import DelayedSubmission

from up_lab import build
from up_lab.config import UPLabError
from up_lab.executors.slurm import max_array_size, read_ledger, run_slurm

SBATCH = """#!/bin/bash
# Accepts "limit" submissions, then refuses like a full queue.
n=$(cat "$FAKE_SLURM/count" 2>/dev/null || echo 0)
limit=$(cat "$FAKE_SLURM/limit" 2>/dev/null || echo 999)
if [ "$n" -ge "$limit" ]; then
  echo "sbatch: error: QOSMaxSubmitJobPerUserLimit" >&2
  exit 1
fi
n=$((n + 1))
echo "$n" > "$FAKE_SLURM/count"
id=$((100 + n))
cp "$1" "$FAKE_SLURM/job_$id.sh"
echo "$id" >> "$FAKE_SLURM/live"
echo "Submitted batch job $id"
"""
STUBS = {
    "sbatch": SBATCH,
    "squeue": '#!/bin/bash\necho "$@" >> "$FAKE_SLURM/squeue_args"\n'
    'cat "$FAKE_SLURM/live" 2>/dev/null\nexit 0\n',
    "scontrol": '#!/bin/bash\necho "MaxArraySize            = 4"\n',
    "srun": "#!/bin/bash\nexit 0\n",
}


class Cluster:
    """What the fake Slurm commands recorded."""

    def __init__(self, root: Path, exp_dir: Path) -> None:
        self.root, self.exp_dir = root, exp_dir

    def script(self, job_id: int) -> str:
        return (self.root / f"job_{job_id}.sh").read_text()

    def flag(self, job_id: int, name: str) -> Any:
        match = re.search(
            rf"^#SBATCH --{name}=(.*)$", self.script(job_id), re.MULTILINE
        )
        return match.group(1) if match else None

    def jobs(self) -> List[int]:
        return sorted(int(p.stem[4:]) for p in self.root.glob("job_*.sh"))

    def submission(self, job: str) -> DelayedSubmission:
        folder = self.exp_dir / "slurm" / "submitit"
        return DelayedSubmission.load(folder / f"{job}_submitted.pkl")

    def set(self, name: str, value: Any) -> None:
        (self.root / name).write_text(f"{value}\n")

    def arrays(self) -> List[Tuple[str, int]]:
        return [
            (a["job_id"], len(a["run_ids"]))
            for a in read_ledger(self.exp_dir)["arrays"]
        ]


@pytest.fixture
def cluster(tmp_path: Path, monkeypatch: Any, make_exp: Callable[..., Any]) -> Any:
    """Returns ``start(**slurm options) -> (Cluster, exp_dir, config, run ids)``."""
    bin_dir, root = tmp_path / "bin", tmp_path / "fake-slurm"
    bin_dir.mkdir()
    root.mkdir()
    for name, text in STUBS.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir), prepend=":")
    monkeypatch.setenv("FAKE_SLURM", str(root))
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    def start(**slurm: Any) -> Any:
        exp_dir, config = make_exp(  # 6 runs, the last one already done
            executor={"kind": "slurm", "slurm": {"array_parallelism": 2, **slurm}},
            benchmarks=[{"name": "b", "source": str(tmp_path / "bench")}],
            requests=[{"name": "sat"}],
        )
        ids = build.build(exp_dir, config)
        (exp_dir / "runs" / ids[-1] / "result.json").write_text("{}")
        return Cluster(root, exp_dir), exp_dir, config, ids[:-1]

    return start


def test_max_array_size_from_scontrol(cluster: Any, monkeypatch: Any) -> None:
    assert max_array_size() == 3
    monkeypatch.setenv("PATH", "/nonexistent")
    assert max_array_size() == 1000


def test_arrays_are_chained_one_element_per_run(cluster: Any, capsys: Any) -> None:
    fake, exp_dir, config, ids = cluster(partition="cpu", extra={"nice": 5})
    assert len(ids) == 5
    run_slurm(exp_dir, config)

    assert fake.jobs() == [101, 102, 103]  # arrays of 3 and 2 runs, continuation
    assert fake.arrays() == [("101", 3), ("102", 2)]
    ledger = read_ledger(exp_dir)
    assert [r for a in ledger["arrays"] for r in a["run_ids"]] == ids
    assert all(a["submitted_at"] for a in ledger["arrays"])
    assert fake.flag(101, "array") == "0-2%2" and fake.flag(102, "array") == "0-1%2"
    assert fake.flag(101, "dependency") is None
    assert fake.flag(102, "dependency") == "afterany:101"
    # Slurm limits are the run limits (300 s, 4096 MB, 1 core) plus headroom.
    assert fake.flag(101, "time") == "15" and fake.flag(101, "mem") == "5120M"
    assert fake.flag(101, "cpus-per-task") == "1"
    assert fake.flag(101, "partition") == "cpu" and fake.flag(101, "nice") == "5"
    for job, run_id in zip(("101_0", "101_1", "101_2", "102_0", "102_1"), ids):
        submission = fake.submission(job)
        assert submission.function.command == ["bash"]
        assert submission.args == (str(exp_dir / "runs" / run_id / "run.sh"),)
    # Everything is queued: the continuation collects after the last array.
    assert fake.flag(103, "dependency") == "afterany:102"
    assert fake.flag(103, "array") is None and "srun" not in fake.script(103)
    assert fake.submission("103").function.command[1:] == [
        "-m",
        "up_lab.cli",
        "collect",
        str(exp_dir),
    ]
    assert ledger["continuation"] == "103"
    assert "5 runs submitted" in capsys.readouterr().out

    run_slurm(exp_dir, config)  # all runs are in live jobs: nothing to do
    assert fake.jobs() == [101, 102, 103]


def test_stop_when_sbatch_refuses_then_resume(cluster: Any, capsys: Any) -> None:
    fake, exp_dir, config, ids = cluster()
    fake.set("limit", 1)
    run_slurm(exp_dir, config)

    assert fake.jobs() == [101] and fake.arrays() == [("101", 3)]
    out = capsys.readouterr().out
    assert "sbatch refused" in out and "Could not queue the continuation job" in out
    assert f"Re-run 'uplab run {exp_dir}'" in out
    assert read_ledger(exp_dir)["continuation"] is None

    # Resume while job 101 is still queued: only the two remaining runs go out.
    fake.set("limit", 999)
    run_slurm(exp_dir, config)
    assert fake.arrays() == [("101", 3), ("102", 2)]
    assert read_ledger(exp_dir)["arrays"][1]["run_ids"] == ids[3:]
    assert fake.flag(102, "dependency") == "afterany:101"
    assert fake.flag(103, "dependency") == "afterany:102"
    assert fake.submission("103").function.command[2:4] == ["up_lab.cli", "collect"]

    # The jobs ended; one run of the first array left no result: it is resubmitted.
    for run_id in ids[1:]:
        (exp_dir / "runs" / run_id / "result.json").write_text("{}")
    fake.set("live", "")
    run_slurm(exp_dir, config)
    assert fake.arrays()[2:] == [("104", 1)]
    assert read_ledger(exp_dir)["arrays"][2]["run_ids"] == ids[:1]
    assert fake.flag(104, "array") is None  # a single run is a plain job
    assert fake.flag(104, "dependency") is None
    assert fake.flag(105, "dependency") == "afterany:104"


def test_max_queued_and_continuation(cluster: Any, monkeypatch: Any) -> None:
    fake, exp_dir, config, ids = cluster(
        max_queued=4, max_array_size=3, dependency="none"
    )
    run_slurm(exp_dir, config)

    assert fake.arrays() == [("101", 3), ("102", 1)]
    assert fake.flag(102, "dependency") is None  # dependency: none
    assert fake.flag(103, "dependency") == "afterany:102"
    assert fake.submission("103").function.command[2:] == [
        "up_lab.cli",
        "run",
        str(exp_dir),
    ]

    run_slurm(exp_dir, config)  # by hand, while continuation 103 is queued
    assert fake.jobs() == [101, 102, 103]

    # Inside the continuation job: the arrays ended, their runs have results.
    monkeypatch.setenv("SLURM_JOB_ID", "103")
    for run_id in ids[:4]:
        (exp_dir / "runs" / run_id / "result.json").write_text("{}")
    fake.set("live", "103")
    run_slurm(exp_dir, config)
    assert fake.arrays()[2:] == [("104", 1)]
    assert read_ledger(exp_dir)["arrays"][2]["run_ids"] == ids[4:]
    assert fake.submission("105").function.command[2:4] == ["up_lab.cli", "collect"]
    assert read_ledger(exp_dir)["continuation"] == "105"
    assert not list((exp_dir / "slurm").glob("*.tmp"))


def test_without_slurm(cluster: Any, monkeypatch: Any) -> None:
    fake, exp_dir, config, ids = cluster()
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(UPLabError, match="squeue"):
        run_slurm(exp_dir, config)
