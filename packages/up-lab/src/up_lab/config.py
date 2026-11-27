"""Experiment configuration: the pydantic models behind ``experiment.yaml``."""

from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from unified_planning.exceptions import UPException


class UPLabError(UPException):
    """Raised for invalid configurations and failed experiment steps."""


Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]*$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BenchmarkConfig(_Model):
    """A benchmark source: git URL or local path, optionally pinned."""

    name: Name
    source: str
    rev: Optional[str] = None
    subdir: Optional[str] = None
    layout: str = "auto"
    manifest: Optional[str] = None
    include: List[str] = []
    exclude: List[str] = []
    max_per_domain: Optional[int] = Field(default=None, ge=1)


class PlannerConfig(_Model):
    """Pip requirement(s) installed into the planner's own environment."""

    name: Name
    source: List[str] = Field(min_length=1)
    python: Optional[str] = None

    @field_validator("source", mode="before")
    @classmethod
    def _single_source(cls, value: Any) -> Any:
        return [value] if isinstance(value, str) else value


class StepConfig(_Model):
    """One UP engine of a pipeline."""

    engine: str
    params: Dict[str, Any] = {}
    compilation_kind: Optional[str] = None


class VariantConfig(_Model):
    """A planner with a pipeline of engines and their parameters."""

    name: Name
    planner: str
    pipeline: List[StepConfig] = Field(min_length=1)
    compile_only: bool = False

    @model_validator(mode="before")
    @classmethod
    def _shorthand(cls, data: Any) -> Any:
        if isinstance(data, dict) and "engine" in data:
            if "pipeline" in data:
                raise ValueError("give either 'pipeline' or 'engine', not both")
            data = dict(data)
            step = {"engine": data.pop("engine"), "params": data.pop("params", {})}
            data["pipeline"] = [step]
        return data


class RequestConfig(_Model):
    """What is asked of the planner."""

    name: Name
    mode: Literal["oneshot", "anytime", "planset"] = "oneshot"
    k: Optional[int] = Field(default=1, ge=1)
    quality_bound: Optional[float] = None
    diversity: Any = None


class LimitsConfig(_Model):
    """Per-run limits: seconds, megabytes, cores."""

    time: int = Field(default=300, gt=0)
    memory_mb: int = Field(default=4096, gt=0)
    cores: int = Field(default=1, gt=0)


class SlurmConfig(_Model):
    """Options of the Slurm submitter."""

    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    array_parallelism: int = Field(default=100, ge=1)
    max_array_size: Optional[int] = Field(default=None, ge=1)
    max_queued: Optional[int] = Field(default=None, ge=1)
    dependency: Literal["afterany", "afterok", "none"] = "afterany"
    time_headroom: int = Field(default=600, ge=0)
    memory_headroom_mb: int = Field(default=1024, ge=0)
    extra: Dict[str, Any] = {}


class ExecutorConfig(_Model):
    """How runs are executed."""

    kind: str = "local"
    workers: int = Field(default=4, ge=1)
    slurm: SlurmConfig = SlurmConfig()


class AnalysisConfig(_Model):
    """An analysis plugin name or a path to a ``CONSTRUCT`` query."""

    name: str
    options: Dict[str, Any] = {}


class ReportConfig(_Model):
    """A saved query and the renderer of its result."""

    name: Name
    query: str
    render: str
    options: Dict[str, Any] = {}


class ExperimentConfig(_Model):
    """The whole experiment; mirrors ``experiment.yaml``."""

    name: Name
    benchmarks: List[BenchmarkConfig]
    planners: List[PlannerConfig]
    variants: List[VariantConfig]
    requests: List[RequestConfig]
    seeds: List[int] = Field(default=[0], min_length=1)
    limits: LimitsConfig = LimitsConfig()
    executor: ExecutorConfig = ExecutorConfig()
    analyses: List[AnalysisConfig] = []
    reports: List[ReportConfig] = []

    @model_validator(mode="after")
    def _check_names(self) -> "ExperimentConfig":
        for section in ("benchmarks", "planners", "variants", "requests", "reports"):
            names = [item.name for item in getattr(self, section)]
            duplicates = sorted({n for n in names if names.count(n) > 1})
            if duplicates:
                raise ValueError(f"duplicate names in {section}: {duplicates}")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("duplicate seeds")
        planners = {planner.name for planner in self.planners}
        for variant in self.variants:
            if variant.planner not in planners:
                raise ValueError(
                    f"variant {variant.name!r}: unknown planner {variant.planner!r}"
                )
        return self


def is_url(source: str) -> bool:
    """
    :param source: a benchmark source.
    :return: whether it is a git URL rather than a local path.
    """
    return "://" in source or source.startswith("git@")


def _absolute(value: str, base: Path) -> str:
    return str((base / Path(value).expanduser()).resolve())


def parse_config(data: Any, base: Union[str, Path, None] = None) -> ExperimentConfig:
    """
    Validates a configuration and makes its local paths absolute.

    :param data: the content of an ``experiment.yaml``.
    :param base: directory against which relative paths are resolved.
    :return: the validated configuration.
    """
    try:
        config = ExperimentConfig.model_validate(data)
    except ValidationError as error:
        raise UPLabError(f"invalid experiment configuration: {error}") from error
    root = Path(base) if base is not None else Path.cwd()
    for benchmark in config.benchmarks:
        if not is_url(benchmark.source):
            benchmark.source = _absolute(benchmark.source, root)
        if benchmark.manifest is not None:
            benchmark.manifest = _absolute(benchmark.manifest, root)
    for planner in config.planners:
        # A requirement starting with ".", "/" or "~" is a local path.
        planner.source = [
            _absolute(s, root) if s.startswith((".", "/", "~")) else s
            for s in planner.source
        ]
    return config


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    """
    :param path: an ``experiment.yaml``; relative local paths in it are
        resolved against its directory.
    :return: the validated configuration.
    """
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise UPLabError(f"cannot read {path}: {error}") from error
    return parse_config(data, path.resolve().parent)


def dump_config(config: ExperimentConfig, path: Path) -> None:
    """
    Writes the resolved copy of a configuration.

    :param config: the configuration.
    :param path: the destination YAML file.
    """
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
