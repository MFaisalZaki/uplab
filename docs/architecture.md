# up-lab architecture and contracts

up-lab is to the Unified Planning library (UP) what Lab is to Fast Downward:
it runs planners over benchmark tasks under limits, records everything about
every run, and turns the records into tables and figures.

This file is the contract between the modules. Change it before changing a
schema.

## Principles

1. **Files are the source of truth.** A run writes only into its own
   directory. The RDF store, tables and figures are derived and rebuildable.
2. **Planners are isolated.** Each planner lives in its own `uv` virtual
   environment; a run is a subprocess of that environment's Python.
3. **Planners cannot certify themselves.** Costs and validity are recomputed
   by `collect`, outside the planner's process.
4. **Little code, maintained libraries.** Glue only; no speculative
   abstraction.
5. **Unified Planning style** (see the last section).

## Distributions

| Distribution | Module | Depends on | Role |
|---|---|---|---|
| `up-lab-worker` | `up_lab_worker` | `unified-planning` only (otherwise stdlib) | Installed into every planner environment. Plan-set mixin, in-environment runner, record schema constants. |
| `up-lab` | `up_lab` | `up-lab-worker`, pydantic, PyYAML, typer, pluggy, submitit, pyoxigraph, pandas, seaborn | Orchestrator and CLI `uplab`. |
| `up-lab-stats` | `up_lab_stats` | `up-lab`, scipy | Plugin (entry point only, never imported by `up_lab`): planner-comparison analyses, their queries and the `anytime` renderer. Its terms are in `up_lab_stats/vocab.ttl`. |

## Vocabulary

| Term | Meaning |
|---|---|
| Benchmark source | Git URL or local path, pinned to a commit, yielding tasks |
| Task | One problem: PDDL/HDDL/ANML files, or a Python factory returning a UP `Problem` |
| Planner | Installable distribution(s) exposing UP engines |
| Variant | Planner + pipeline of UP engines + parameters |
| Request | What is asked: `mode`, `k`, `quality_bound`, `diversity` |
| Run | task x variant x request x seed |

## Pipeline

```
uplab resolve   clone benchmark sources, build planner environments, write lock.json
uplab build     expand the run matrix, write runs/<run-id>/{run.json,run.sh}
uplab run       execute through the configured executor (local pool or Slurm submitter)
uplab status    counts by state
uplab collect   validate plans, write validation.json + record.jsonld, load the store
uplab analyze   run analysis plugins, each into its own named graph
uplab report    saved SPARQL queries -> CSV / LaTeX / PDF, optional HTML
uplab notebook  marimo notebook on the store
uplab serve     SPARQL endpoint (Oxigraph server)
```

## Experiment directory

Defaults to `experiments/<name>` next to the experiment YAML.

```
<exp-dir>/
  experiment.yaml     resolved copy of the configuration
  lock.json           benchmark commits, planner environments
  benchmarks/<name>/  checkouts
  envs/<planner>/     uv virtual environments
  runs/index.jsonl    one line per run: run_id + coordinates
  runs/<run-id>/      see "Run directory"
  slurm/              submitit folder and ledger.json
  store/              Oxigraph store (rebuildable)
  reports/            report outputs
```

## Configuration (`experiment.yaml`)

Validated by pydantic models in `up_lab.config`; the same models are the
Python API. Unknown keys are errors.

```yaml
name: demo
benchmarks:
  - name: classical                 # unique
    source: https://github.com/AI-Planning/classical-domains  # git URL or local path
    rev: null                       # branch, tag or commit; null = default branch / working tree
    subdir: classical               # optional
    layout: auto                    # auto | <layout plugin name>
    manifest: null                  # optional path to a manifest outside the repo
    include: ["gripper/*"]          # fnmatch on "<domain>/<problem>"; empty = all
    exclude: []
    max_per_domain: null
planners:
  - name: pyperplan                 # unique
    source: [up-pyperplan]          # pip requirement(s): PyPI spec, git URL, or local path
    python: null                    # optional interpreter version for uv
variants:
  - name: pyperplan
    planner: pyperplan
    pipeline:                       # all steps but the last are compilers
      - engine: pyperplan           # UP engine name, or "module:Class"
        params: {}                  # constructor kwargs; "${k}", "${quality_bound}", "${seed}" are substituted
        compilation_kind: null      # compilers only, a CompilationKind name
    compile_only: false             # true: every step is a compiler, no planning
requests:
  - name: sat
    mode: oneshot                   # oneshot | anytime | planset
    k: 1                            # null = as many as the engine yields (anytime)
    quality_bound: null             # plans may cost up to quality_bound x reference cost
    diversity: null                 # opaque JSON, interpreted by the planner
seeds: [0]
limits: {time: 300, memory_mb: 4096, cores: 1}
executor:
  kind: local                       # local | slurm | <executor plugin name>
  workers: 4                        # local
  slurm:
    partition: null
    qos: null
    account: null
    array_parallelism: 100          # %N
    max_array_size: null            # null = ask scontrol, fall back to 1000
    max_queued: null                # optional cap on queued array elements
    max_submissions: 2              # give up on a run after this many submissions without a result
    poll_interval: 60               # seconds between the controller's checks
    controller_time: 1440           # minutes; the controller resubmits itself before this
    time_headroom: 600              # seconds added to the Slurm limit
    memory_headroom_mb: 1024
    extra: {}                       # passed to sbatch verbatim
analyses: []                        # [{name: <analysis plugin or path.rq>, options: {}}]
reports: []                         # [{name, query: builtin:<name> | plugin:<name> | path.rq, render: <renderer>, options: {}}]
```

A variant may use the shorthand `engine:` + `params:` instead of a one-step
`pipeline`.

### Benchmark manifest (`uplab-benchmark.yaml`)

Optional; in the repository root (or `subdir`), or given by `manifest:`.
Overrides layout discovery.

```yaml
tasks:
  - domain: gripper
    problem: prob01
    format: pddl                    # pddl | anml | python
    domain_file: gripper/domain.pddl
    problem_file: gripper/prob01.pddl
    factory: null                   # python: "module:function" returning a Problem
    factory_args: {}
    reference_cost: 11
    tags: [ipc-1998]
```

HDDL is read by UP's `PDDLReader`, so it uses `format: pddl`.

## Run directory

```
runs/<run-id>/
  run.json         inputs; written by build, never modified
  run.sh           standalone launcher: exec <env python> -m up_lab_worker <run-dir>
  result.json      outcome; written atomically (tmp + rename) when the run ends
  plans.jsonl      one line per plan, appended as plans arrive
  plans/00001.plan one file per plan
  events.jsonl     timeline
  stdout.log stderr.log
  validation.json  written by collect
  record.jsonld    merged JSON-LD record written by collect; this is what the store loads
```

`solve.json` (the solving process's checkpoint) and `driver.log` (local
executor, only when a launcher failed) may also appear.

A run is **done** iff `result.json` exists. `run_id` is the first 12 hex
digits of sha1 over `task.id|variant.name|request.name|seed`.

### `run.json`

```json
{
  "schema": "uplab/run/1",
  "run_id": "3f2a9c01d4e7",
  "experiment": "demo",
  "seed": 0,
  "task": {
    "id": "classical/gripper/prob01",
    "source": "classical", "source_rev": "<commit or null>",
    "domain": "gripper", "problem": "prob01",
    "format": "pddl",
    "domain_file": "/abs/domain.pddl", "problem_file": "/abs/prob01.pddl",
    "files_sha256": {"domain_file": "...", "problem_file": "..."},
    "factory": null, "factory_args": {}, "python_path": null,
    "reference_cost": null, "tags": []
  },
  "variant": {
    "name": "pyperplan", "planner": "pyperplan", "compile_only": false,
    "pipeline": [{"engine": "pyperplan", "params": {}, "compilation_kind": null}]
  },
  "request": {"name": "sat", "mode": "oneshot", "k": 1, "quality_bound": null, "diversity": null},
  "limits": {"time": 300, "memory_mb": 4096, "cores": 1},
  "planner_env": {"python": "/abs/envs/pyperplan/bin/python", "source": ["up-pyperplan"],
                  "freeze_sha256": "...", "packages": {"up-pyperplan": "1.1.0"}}
}
```

For `format: python`, `python_path` is prepended to `sys.path` and
`factory(**factory_args)` returns the problem. Parameters in `pipeline` are
already substituted.

### `result.json`

```json
{
  "schema": "uplab/result/1",
  "run_id": "3f2a9c01d4e7",
  "status": "SOLVED",
  "up_status": "SOLVED_SATISFICING",
  "exit_code": 0, "signal": null, "error": null,
  "started_at": "2026-11-27T10:00:00+00:00", "ended_at": "2026-11-27T10:00:02+00:00",
  "resources": {"wall_time": 1.93, "cpu_time": 1.80, "peak_memory_mb": 91.2},
  "phases": {
    "parse": {"wall": 0.21, "cpu": 0.20},
    "compile": [{"engine": "up_quantifiers_remover", "wall": 0.01, "cpu": 0.01,
                 "output": {"actions": 12, "fluents": 9, "objects": 6}}],
    "solve": {"wall": 1.5, "cpu": 1.4},
    "first_plan": {"wall": 1.5, "cpu": 1.4},
    "last_plan": {"wall": 1.5, "cpu": 1.4}
  },
  "problem": {"kind": ["ACTION_BASED", "FLAT_TYPING"], "actions": 3, "fluents": 4, "objects": 6},
  "plans": {"count": 1, "claimed_optimal": false, "claimed_complete": null},
  "engine": {"name": "pyperplan", "metrics": {}, "log_messages": []},
  "machine": {"hostname": "n01", "cpu_model": "...", "cpu_count": 8, "memory_mb": 32000,
              "os": "Linux-6.1", "python": "3.12.3", "unified_planning": "1.3.0",
              "up_lab_worker": "0.1.0", "slurm_job_id": null,
              "slurm_array_job_id": null, "slurm_array_task_id": null}
}
```

`status` (set by the worker):

| Status | Meaning |
|---|---|
| `SOLVED` | at least one plan was returned |
| `UNSOLVABLE` | the engine proved or reported unsolvability |
| `COMPILED` | `compile_only` variant finished |
| `TIMEOUT` | time limit hit (plans found before it are kept) |
| `MEMOUT` | memory limit hit |
| `UNSUPPORTED` | an engine does not support the problem kind |
| `ERROR` | anything else; `error` holds the message |

The engine is given the remaining time as its timeout; the driver kills the
process group two seconds after the limit, so a planner that honours its
timeout can still return its plans.

Phase times are measured inside the solving process; `resources` are measured
by the parent for the whole process tree. Unknown values are `null`, never
omitted.

### `plans.jsonl`

```json
{"index": 1, "file": "plans/00001.plan", "wall_time": 1.5, "cpu_time": 1.4, "claimed_cost": null, "length": 11, "kind": "SEQUENTIAL_PLAN"}
```

Times are seconds since the solving process started. Plans are expressed on
the **original** problem (mapped back through compiler steps).

### Plan file format

Text, one action per line, parseable back against the original problem:

```
pick(ball1, rooma, left)            sequential
0.0: move(a, b) [5.0]               time-triggered: start, action, optional duration
```

Arguments are object names (or literal constants). `up_lab_worker.plans`
provides `write_plan(plan, path)` and `read_plan(problem, path)`; plan kinds
it cannot express are written with `str(plan)` and marked unreadable at
validation.

### `events.jsonl`

`{"t": <wall seconds>, "event": "<name>", ...}` with events `start`,
`parse_end`, `compile_end`, `solve_start`, `plan`, `solve_end`, `limit`,
`error`.

### `validation.json` (written by `collect`)

```json
{
  "schema": "uplab/validation/1",
  "run_id": "3f2a9c01d4e7",
  "validator": "sequential_plan_validator",
  "plans": [{"index": 1, "valid": true, "cost": 11, "length": 11, "duplicate_of": null, "within_bound": null}],
  "summary": {"valid": 1, "invalid": 0, "unvalidated": 0, "distinct_valid": 1, "best_cost": 11},
  "parsed": {}
}
```

`valid` is `null` when no validator applies. `cost` is recomputed from the
problem's quality metric (plan length when there is none). `parsed` holds
the output of log-parser plugins.

## Worker (`up_lab_worker`)

- `python -m up_lab_worker <run-dir>`: the **driver**. Starts
  `python -m up_lab_worker.solve <run-dir>` in a new session with
  `RLIMIT_AS` set from `memory_mb`, redirects stdout/stderr to the log
  files, kills the process group when the wall-clock or CPU time limit is
  reached, measures resources with `resource.getrusage(RUSAGE_CHILDREN)`,
  and always writes `result.json`.
- `up_lab_worker.solve`: loads the problem, checks `supports(problem.kind)`,
  runs the pipeline, appends plans and events as they arrive, writes
  `solve.json` (its part of the result) at the end.
- Engines: `"module:Class"` is registered with `factory.add_engine`; the
  class comes from `factory.engine(name)` and is instantiated with the
  parameters and used as a context manager.
- Modes: `oneshot` -> `OneshotPlannerMixin.solve`; `anytime` ->
  `AnytimePlannerMixin.get_solutions`, distinct plans, stop at `k`;
  `planset` -> `PlanSetPlannerMixin.get_plan_set` if the engine has it,
  otherwise the anytime adapter, otherwise oneshot.

### Plan-set mixin

Mirrors `unified_planning.engines.mixins`:

```python
class PlanSetPlannerMixin(ABC):
    @staticmethod
    def is_plan_set_planner() -> bool: ...
    def get_plan_set(self, problem, k, quality_bound=None, diversity=None,
                     timeout=None, output_stream=None, on_plan=None) -> PlanSetGenerationResult: ...
    @abstractmethod
    def _get_plan_set(self, problem, k, quality_bound, diversity,
                      timeout, output_stream, on_plan) -> PlanSetGenerationResult: ...

@dataclass
class PlanSetGenerationResult:
    status: PlanGenerationResultStatus
    plans: List[Plan]
    engine_name: str
    metrics: Optional[Dict[str, str]] = None
    log_messages: Optional[List[LogMessage]] = None
    complete: Optional[bool] = None   # the set contains every plan within the bound
```

`on_plan(plan)` lets a planner report plans as it finds them so they are
timestamped; plans not reported are stamped when the call returns.

## Orchestrator modules (`up_lab`)

| Module | Owner | Contract |
|---|---|---|
| `config` | core | pydantic models; `load_config(path) -> ExperimentConfig` |
| `plugins`, `hookspecs` | core | `hook(name) -> dict[str, Callable]` merging all plugins |
| `benchmarks` | core | clone/pin sources, layout detectors, manifest, selection |
| `envs` | core | uv environments, `lock.json` |
| `build` | core | run matrix, `run.json`, `run.sh`, `runs/index.jsonl` |
| `executors/local`, `executors/slurm` | core | run pending runs |
| `experiment` | core | `Experiment` Python API wrapping the steps |
| `cli` | core | typer app; imports results modules lazily |
| `validation` | results | validate one run -> `validation.json` |
| `collect` | results | `collect(exp_dir, jobs=1)` |
| `store` | results | `open_store(exp_dir, read_only=False)`, `query(exp_dir, sparql) -> DataFrame` |
| `vocab/` | results | `context.jsonld`, `uplab.ttl` |
| `queries/` | results | built-in `.rq` files |
| `analysis` | results | `analyze(exp_dir, names=None)` |
| `reports` | results | `report(exp_dir, names=None, html=False)` |
| `notebook` | results | `write_notebook(exp_dir) -> Path` |
| `serve` | results | `serve(exp_dir, host, port)` |

### Plugins

`pluggy`, project name and entry-point group `up_lab`. Every hook returns a
`dict` mapping a name to a callable (a text for `uplab_queries`);
`up_lab.plugins.hook(name)` merges them.

| Hook | Callable |
|---|---|
| `uplab_benchmark_layouts` | `(root: Path) -> list[dict] | None` task dicts as in the manifest, or `None` if the layout does not match |
| `uplab_executors` | `(exp_dir: Path, config: ExperimentConfig) -> None` |
| `uplab_analyses` | `(store, exp_dir: Path, options: dict) -> Iterable[pyoxigraph.Triple]` |
| `uplab_renderers` | `(df: DataFrame, stem: Path, options: dict) -> list[Path]` |
| `uplab_queries` | SPARQL text, keyed by query name |
| `uplab_log_parsers` | `(stdout: str) -> dict`, keyed by engine name |

An analysis may also be a `.rq` file with a `CONSTRUCT` query: no code.

`analyze` runs the built-in analyses `reference-cost` and `scores` first, then
the configured ones in the order of `analyses:`; listing a built-in there only
sets its options. A query reference `builtin:<name>` is looked up among
up-lab's own `queries/<name>.rq`, then among the `uplab_queries` of plugins;
`plugin:<name>` only among the latter.

### Slurm submitter

`uplab run` with `executor.kind: slurm` queues one small **controller job**
and returns. The arrays do not depend on each other; Slurm schedules them
freely. The controller:

1. Reads `slurm/ledger.json` and `squeue`; pending = runs without
   `result.json` that are not in a live array.
2. Submits job arrays through submitit, **one run per array index**, sized to
   `max_array_size` and throttled with `array_parallelism`, until everything
   is queued, `max_queued` is reached or `sbatch` refuses. The ledger is
   updated after each accepted array.
3. Sleeps `poll_interval` seconds and repeats, refilling the queue as slots
   free up. A run whose job ended without a result is submitted again, at
   most `max_submissions` times in total.
4. When no run is queued or left to submit, runs `collect` and ends.
5. Shortly before `controller_time`, queues a new controller and ends.

`UPLAB_SLURM_CONTROLLER=1 uplab run <exp-dir>` runs the same loop in the
foreground, for clusters where a job may not call `sbatch`.

Each array element executes its run's `run.sh`. The Slurm limits are the run
limits plus headroom so the worker's own limits fire first.

### Store

Oxigraph (pyoxigraph), on disk in `store/`. Namespace
`PREFIX uplab: <urn:uplab:ns#>`, reusing PROV-O (`uplab:Run` is a
`prov:Activity`, `uplab:Variant` a `prov:SoftwareAgent`, `uplab:Plan` a
`prov:Entity` with `prov:wasGeneratedBy`). Named graphs:
`urn:uplab:graph:<experiment>` for records and
`urn:uplab:graph:<experiment>:analysis:<name>` per analysis, so an analysis
can be dropped and recomputed. Queries see the union of the graphs, in which
a triple stored in two graphs matches twice: an analysis describes only
resources of its own, or adds properties to those of the records. Per-plan
facts are loaded by default.

## Unified Planning style

- Operation-mode mixins with a public method doing the checks and an
  abstract `_method`, as in `unified_planning.engines.mixins`.
- Engines are obtained from the UP `Factory`, used as context managers, and
  filtered with `supports(problem.kind)`.
- Results are dataclasses carrying a `PlanGenerationResultStatus`.
- Exceptions derive from `unified_planning.exceptions.UPException`.
- Typed code, `black` formatting, reStructuredText docstrings with
  `:param:` / `:return:`, tests with `pytest`.
- Distribution names `up-...`, module names `up_...`.
