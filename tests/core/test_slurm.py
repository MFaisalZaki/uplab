"""The Slurm submitter against fake ``sbatch``/``squeue``/``scontrol``."""

import json
import os
import re
from pathlib import Path
from typing import Any, Callable, List, Tuple

import pytest
from submitit.core.utils import DelayedSubmission

from up_lab import build
from up_lab.config import UPLabError
from up_lab.executors import slurm as slurm_module
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
    monkeypatch.delenv("UPLAB_SLURM_CONTROLLER", raising=False)

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


def finish(exp_dir: Path, run_ids: List[str]) -> None:
    for run_id in run_ids:
        (exp_dir / "runs" / run_id / "result.json").write_text("{}")


@pytest.fixture
def controller(monkeypatch: Any) -> Any:
    """Makes `run_slurm` the controller; returns `(steps, collected)`.

    Each `time.sleep` of the loop pops and calls the next step.
    """
    steps: List[Callable[[], None]] = []
    collected: List[Path] = []
    clock = [0.0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds
        assert steps, "the controller loop did not end"
        steps.pop(0)()

    monkeypatch.setenv("UPLAB_SLURM_CONTROLLER", "1")
    monkeypatch.setattr(slurm_module.time, "sleep", sleep)
    monkeypatch.setattr(slurm_module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr("up_lab.collect.collect", collected.append)
    return steps, collected


def test_run_queues_one_controller_job(cluster: Any, capsys: Any) -> None:
    fake, exp_dir, config, ids = cluster(partition="cpu", controller_time=90)
    run_slurm(exp_dir, config)

    assert fake.jobs() == [101] and fake.arrays() == []
    assert fake.flag(101, "array") is None and "srun" not in fake.script(101)
    assert fake.flag(101, "time") == "90" and fake.flag(101, "partition") == "cpu"
    assert fake.flag(101, "dependency") is None
    command = fake.submission("101").function.command
    assert command[:2] == ["env", "UPLAB_SLURM_CONTROLLER=1"]
    assert command[3:] == ["-m", "up_lab.cli", "run", str(exp_dir)]
    assert read_ledger(exp_dir)["controller"] == "101"

    run_slurm(exp_dir, config)  # the controller is queued: nothing to do
    assert fake.jobs() == [101]
    assert "already queued" in capsys.readouterr().out

    fake.set("live", "")
    finish(exp_dir, ids)
    run_slurm(exp_dir, config)  # no pending run: nothing to do
    assert fake.jobs() == [101]


def test_controller_refused(cluster: Any, capsys: Any) -> None:
    fake, exp_dir, config, ids = cluster()
    fake.set("limit", 0)
    run_slurm(exp_dir, config)
    out = capsys.readouterr().out
    assert "Could not queue the controller job" in out
    assert f"UPLAB_SLURM_CONTROLLER=1 uplab run {exp_dir}" in out
    assert read_ledger(exp_dir)["controller"] is None


def test_controller_submits_independent_arrays(cluster: Any, controller: Any) -> None:
    steps, collected = controller
    fake, exp_dir, config, ids = cluster(extra={"nice": 5})

    def arrays_end() -> None:
        assert collected == []
        finish(exp_dir, ids)
        fake.set("live", "")

    steps.append(arrays_end)
    run_slurm(exp_dir, config)

    assert fake.jobs() == [101, 102]  # arrays of 3 and 2 runs, nothing else
    assert fake.arrays() == [("101", 3), ("102", 2)]
    ledger = read_ledger(exp_dir)
    assert [r for a in ledger["arrays"] for r in a["run_ids"]] == ids
    assert fake.flag(101, "array") == "0-2%2" and fake.flag(102, "array") == "0-1%2"
    assert fake.flag(101, "dependency") is None
    assert fake.flag(102, "dependency") is None
    # Slurm limits are the run limits (300 s, 4096 MB, 1 core) plus headroom.
    assert fake.flag(101, "time") == "15" and fake.flag(101, "mem") == "5120M"
    assert fake.flag(101, "cpus-per-task") == "1" and fake.flag(101, "nice") == "5"
    for job, run_id in zip(("101_0", "101_1", "101_2", "102_0", "102_1"), ids):
        submission = fake.submission(job)
        assert submission.function.command == ["bash"]
        assert submission.args == (str(exp_dir / "runs" / run_id / "run.sh"),)
    assert collected == [exp_dir] and steps == []
    assert not list((exp_dir / "slurm").glob("*.tmp"))


def test_controller_refills_when_sbatch_accepts_again(
    cluster: Any, controller: Any, capsys: Any
) -> None:
    steps, collected = controller
    fake, exp_dir, config, ids = cluster()
    fake.set("limit", 1)

    def still_full() -> None:
        assert fake.arrays() == [("101", 3)]

    def room_again() -> None:
        fake.set("limit", 999)

    def all_end() -> None:
        assert fake.arrays() == [("101", 3), ("102", 2)]
        assert read_ledger(exp_dir)["arrays"][1]["run_ids"] == ids[3:]
        finish(exp_dir, ids)
        fake.set("live", "")

    steps.extend([still_full, room_again, all_end])
    run_slurm(exp_dir, config)
    assert "sbatch refused" in capsys.readouterr().out
    assert fake.jobs() == [101, 102] and collected == [exp_dir]


def test_max_queued_and_lost_runs(cluster: Any, controller: Any, capsys: Any) -> None:
    steps, collected = controller
    fake, exp_dir, config, ids = cluster(max_queued=4, max_array_size=3)

    def first_batch_ends() -> None:
        assert fake.arrays() == [("101", 3), ("102", 1)]  # max_queued = 4
        finish(exp_dir, ids[1:4])  # ids[0] was killed by the scheduler
        fake.set("live", "")

    def second_batch_ends() -> None:
        # The last run and the lost one go out together; a single run would
        # be a plain job.
        assert fake.arrays()[2:] == [("103", 2)]
        assert read_ledger(exp_dir)["arrays"][2]["run_ids"] == [ids[0], ids[4]]
        finish(exp_dir, ids[4:])
        fake.set("live", "")

    steps.extend([first_batch_ends, second_batch_ends])
    run_slurm(exp_dir, config)

    # ids[0] was submitted twice (max_submissions) and is given up.
    assert fake.jobs() == [101, 102, 103] and collected == [exp_dir]
    out = capsys.readouterr().out
    assert f"1 runs left no result after 2 submissions: {ids[0]}" in out


def test_controller_resubmits_itself_before_its_time_limit(
    cluster: Any, controller: Any, monkeypatch: Any
) -> None:
    steps, collected = controller
    fake, exp_dir, config, ids = cluster(controller_time=20, poll_interval=600)
    monkeypatch.setenv("SLURM_JOB_ID", "100")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "4000")
    steps.append(lambda: None)  # one sleep of 600 s reaches the deadline
    run_slurm(exp_dir, config)

    # The controller's own Slurm settings must not reach the jobs it submits.
    assert not [n for n in os.environ if n.startswith("SLURM_")]
    assert "srun" not in fake.script(101) and "srun" not in fake.script(102)

    assert fake.jobs() == [101, 102, 103] and collected == []
    command = fake.submission("103").function.command
    assert command[:2] == ["env", "UPLAB_SLURM_CONTROLLER=1"]
    assert read_ledger(exp_dir)["controller"] == "103"


def test_without_slurm(cluster: Any, monkeypatch: Any) -> None:
    fake, exp_dir, config, ids = cluster()
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(UPLabError, match="squeue"):
        run_slurm(exp_dir, config)


def test_partition_qos_and_account_on_every_job(cluster: Any, controller: Any) -> None:
    steps, collected = controller
    options = {"partition": "cpu", "qos": "normal", "account": "proj"}
    fake, exp_dir, config, ids = cluster(**options)

    def arrays_end() -> None:
        finish(exp_dir, ids)
        fake.set("live", "")

    steps.append(arrays_end)
    run_slurm(exp_dir, config)  # as the controller: arrays 101 and 102
    (exp_dir / "runs" / ids[0] / "result.json").unlink()
    os.environ.pop("UPLAB_SLURM_CONTROLLER")
    run_slurm(exp_dir, config)  # from the login node: controller job 103
    assert fake.jobs() == [101, 102, 103]
    for job in fake.jobs():
        assert {name: fake.flag(job, name) for name in options} == options
