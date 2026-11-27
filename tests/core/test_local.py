import json
from pathlib import Path
from typing import Any, Callable

from up_lab import build
from up_lab.experiment import Experiment
from up_lab.plugins import hook

STUB = """#!/bin/bash
dir="$(dirname "$(readlink -f "$0")")"
echo '{"status": "SOLVED"}' > "$dir/result.json"
"""
BROKEN = "#!/bin/bash\necho no worker here >&2\nexit 3\n"


def test_local_executor_runs_pending_runs(
    make_exp: Callable[..., Any], capsys: Any
) -> None:
    exp_dir, config = make_exp(executor={"kind": "local", "workers": 3})
    ids = build.build(exp_dir, config)
    runs = exp_dir / "runs"
    for run_id in ids:
        (runs / run_id / "run.sh").write_text(STUB)
    (runs / ids[0] / "result.json").write_text('{"status": "TIMEOUT"}')  # done: skipped
    (runs / ids[1] / "run.sh").write_text(BROKEN)

    hook("uplab_executors")["local"](exp_dir, config)

    assert "5 runs executed, 1 without a result" in capsys.readouterr().out
    assert (
        json.loads((runs / ids[0] / "result.json").read_text())["status"] == "TIMEOUT"
    )
    assert (runs / ids[1] / "driver.log").read_text() == "no worker here\n"
    assert build.pending(exp_dir) == [ids[1]]
    experiment = Experiment(config, exp_dir)
    assert experiment.status() == {"TIMEOUT": 1, "PENDING": 1, "SOLVED": 4}
