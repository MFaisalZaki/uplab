# up-lab

up-lab is to the [Unified Planning](https://github.com/aiplan4eu/unified-planning)
library what [Lab](https://lab.readthedocs.io) is to Fast Downward: it runs
planners over benchmark tasks under time and memory limits, records everything
about every run, and turns the records into tables and figures.

- Every planner lives in its own `uv` virtual environment.
- Every run is a directory of plain files; the RDF store, tables and figures
  are derived from them and can be rebuilt.
- Plan validity and cost are recomputed outside the planner's process.
- Runs execute on a local process pool or on Slurm, as job arrays submitted by a controller job.

The schemas and module contracts are in
[docs/architecture.md](docs/architecture.md).

## Quick start

Requirements: Python >= 3.10, [`uv`](https://docs.astral.sh/uv/) and `git`.

```sh
uv sync                                   # installs up-lab and up-lab-worker
uv run uplab all examples/experiment.yaml
```

`examples/experiment.yaml` runs pyperplan on two small domains. `uplab all`
is the following steps, which can also be run one at a time; after `resolve`
they accept the experiment directory (here `examples/experiments/demo`) in
place of the YAML file.

| Command | Effect |
|---|---|
| `uplab resolve <yaml> [--dir <exp-dir>]` | fetch benchmark sources, build planner environments, write `lock.json` |
| `uplab build <exp>` | expand the run matrix into `runs/<run-id>/{run.json,run.sh}` |
| `uplab run <exp>` | execute the pending runs (local pool or Slurm) |
| `uplab status <exp>` | number of runs per state |
| `uplab collect <exp>` | validate plans, write the records, load the store |
| `uplab analyze <exp>` | run the analyses |
| `uplab report <exp>` | write the reports |
| `uplab notebook <exp>` | write a marimo notebook on the store |
| `uplab serve <exp>` | SPARQL endpoint on the store |

The same steps are methods of `up_lab.experiment.Experiment`:

```python
from up_lab.experiment import Experiment

exp = Experiment("examples/experiment.yaml")
exp.resolve(); exp.build(); exp.run()
print(exp.status())
```

A run that has a `result.json` is done and is never run again: `uplab build`
and `uplab run` can be repeated after adding tasks, variants or seeds.

## Concepts

| Term | Meaning |
|---|---|
| Benchmark source | Git URL or local path, pinned to a commit, yielding tasks |
| Task | One problem: PDDL/HDDL/ANML files, or a Python factory returning a UP `Problem` |
| Planner | Installable distribution(s) exposing UP engines |
| Variant | Planner + pipeline of UP engines + parameters |
| Request | What is asked: `mode`, `k`, `quality_bound`, `diversity` |
| Run | task x variant x request x seed |

## Adding a planner

A planner is anything `uv pip install` accepts that registers a UP engine:

```yaml
planners:
  - name: mine
    source: [./path/to/up-mine]     # PyPI spec, git URL, or local path
    python: "3.11"                  # optional
variants:
  - name: mine-k
    planner: mine
    pipeline:
      - engine: up_quantifiers_remover          # compilers first
        compilation_kind: QUANTIFIERS_REMOVING
      - engine: "up_mine.engine:MyEngine"       # UP engine name or module:Class
        params: {k: "${k}", seed: "${seed}"}
```

`${k}`, `${quality_bound}` and `${seed}` are replaced per run. A planner that
returns several plans implements `up_lab_worker.mixins.PlanSetPlannerMixin`
and is asked with a `mode: planset` request; anytime engines work with
`mode: anytime`.

## Adding a benchmark

```yaml
benchmarks:
  - name: classical
    source: https://github.com/AI-Planning/classical-domains
    rev: main                       # branch, tag or commit; the commit is recorded
    subdir: classical
    include: ["gripper/*"]          # fnmatch on "<domain>/<problem>"
    max_per_domain: 5
```

Tasks are found by layout: `pddl` (a directory with one domain file and
problem files, or `pNN.pddl` with `pNN-domain.pddl` / `domain_pNN.pddl`;
`.pddl` and `.hddl`) or `anml` (`*.anml` files). Anything else, including
tasks built by a Python function, is listed in a manifest
`uplab-benchmark.yaml` in the source (or given by `manifest:`); see
[docs/architecture.md](docs/architecture.md#benchmark-manifest-uplab-benchmarkyaml).

## Slurm

```yaml
executor:
  kind: slurm
  slurm: {partition: cpu, array_parallelism: 100}
```

`uplab run` queues one small controller job and returns. The controller
submits job arrays with one run per array index until everything is queued
or the scheduler refuses, records what was accepted in `slurm/ledger.json`,
and refills the queue as slots free up. When every run has ended it runs
`uplab collect`. Before its own time limit (`controller_time`, minutes) it
queues a new controller.

If your cluster does not let a job call `sbatch`, run the controller
yourself, for example in `tmux` on the login node:

```bash
UPLAB_SLURM_CONTROLLER=1 uplab run <exp-dir>
```

## Plugins

Plugins use [pluggy](https://pluggy.readthedocs.io); the entry-point group is
`up_lab`. Each hook returns a `dict` from names to callables:

| Hook | Adds |
|---|---|
| `uplab_benchmark_layouts` | benchmark layouts |
| `uplab_executors` | executors |
| `uplab_analyses` | analyses writing triples into the store |
| `uplab_renderers` | report renderers |
| `uplab_log_parsers` | parsers of an engine's output |

```python
# up_mine/uplab_plugin.py
from up_lab.hookspecs import hookimpl

@hookimpl
def uplab_benchmark_layouts():
    return {"mine": lambda root: [{"domain": "d", "problem": "p", "format": "pddl",
                                   "domain_file": "d.pddl", "problem_file": "p.pddl"}]}
```

```toml
[project.entry-points.up_lab]
mine = "up_mine.uplab_plugin"
```

## Development

```sh
uv sync
uv run pytest
```
