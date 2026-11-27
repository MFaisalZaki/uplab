from pathlib import Path

import pytest
import yaml
from unified_planning.exceptions import UPException

from conftest import raw_config
from up_lab.config import UPLabError, dump_config, load_config, parse_config


def test_defaults_and_shorthand(bench: Path) -> None:
    config = parse_config(raw_config(bench))
    step = config.variants[0].pipeline[0]
    assert (step.engine, step.params, step.compilation_kind) == ("pyperplan", {}, None)
    assert config.seeds == [0]
    assert config.limits.model_dump() == {"time": 300, "memory_mb": 4096, "cores": 1}
    assert config.executor.kind == "local"
    assert config.executor.slurm.max_submissions == 2
    assert config.requests[0].mode == "oneshot" and config.requests[0].k == 1


def test_contract_example_is_valid() -> None:
    text = (Path(__file__).parents[2] / "docs" / "architecture.md").read_text()
    section = text.split("## Configuration")[1]
    data = yaml.safe_load(section.split("```yaml")[1].split("```")[0])
    config = parse_config(data)
    assert config.benchmarks[0].source.startswith("https://")
    assert config.executor.slurm.array_parallelism == 100


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"colour": "red"}, "colour"),
        ({"limits": {"time": 10, "disk": 1}}, "disk"),
        ({"variants": [{"name": "v", "planner": "nope", "engine": "e"}]}, "nope"),
        ({"requests": [{"name": "r"}, {"name": "r"}]}, "duplicate"),
        ({"requests": [{"name": "r", "mode": "fast"}]}, "mode"),
        ({"variants": [{"name": "v", "planner": "pp"}]}, "pipeline"),
        ({"name": "has space"}, "name"),
        ({"seeds": [1, 1]}, "duplicate"),
    ],
)
def test_invalid(bench: Path, overrides: dict, message: str) -> None:
    with pytest.raises(UPLabError, match=message) as info:
        parse_config(raw_config(bench, **overrides))
    assert isinstance(info.value, UPException)


def test_engine_and_pipeline_are_exclusive(bench: Path) -> None:
    variant = {
        "name": "v",
        "planner": "pp",
        "engine": "e",
        "pipeline": [{"engine": "e"}],
    }
    with pytest.raises(UPLabError, match="not both"):
        parse_config(raw_config(bench, variants=[variant]))


def test_load_resolves_local_paths_and_round_trips(tmp_path: Path, bench: Path) -> None:
    data = raw_config(bench)
    data["benchmarks"] = [
        {"name": "local", "source": "bench", "manifest": "m.yaml"},
        {"name": "remote", "source": "https://example.org/repo.git"},
    ]
    data["planners"] = [{"name": "pp", "source": ["./planner", "up-pyperplan"]}]
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(data))
    config = load_config(path)
    assert config.benchmarks[0].source == str(bench)
    assert config.benchmarks[0].manifest == str(tmp_path / "m.yaml")
    assert config.benchmarks[1].source == "https://example.org/repo.git"
    assert config.planners[0].source == [str(tmp_path / "planner"), "up-pyperplan"]
    copy = tmp_path / "elsewhere" / "experiment.yaml"
    copy.parent.mkdir()
    dump_config(config, copy)
    assert load_config(copy) == config


def test_unreadable_file(tmp_path: Path) -> None:
    with pytest.raises(UPLabError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
