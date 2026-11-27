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

`./setup.sh` does the installation below in one step (`--example` also runs
the example experiment, `--help` lists the options).

Requirements: Python >= 3.10 and `git`. [`uv`](https://docs.astral.sh/uv/)
is optional.

```sh
uv sync                                   # installs up-lab, up-lab-worker, up-lab-stats
uv run uplab all examples/local.yaml
```

Without `uv`:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -e packages/up-lab-worker -e packages/up-lab -e packages/up-lab-stats
uplab all examples/local.yaml
```

Planner environments are built with `uv` when it is on `PATH`, and with the
standard `venv` and `pip` otherwise. Only `uv` can fetch a Python version
that is not installed (a planner's `python:` option).

`examples/local.yaml` runs pyperplan on two small domains. `uplab all`
is the following steps, which can also be run one at a time; after `resolve`
they accept the experiment directory (here `examples/experiments/demo-local`) in
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

exp = Experiment("examples/local.yaml")
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

`examples/slurm.yaml` is the local example with the Slurm executor; its
header lists the commands to run. Set the partition, QOS and account your
cluster requires (null uses the cluster's default):

```yaml
executor:
  kind: slurm
  slurm: {partition: compute, qos: normal, account: my-project, array_parallelism: 20}
```

Every job the experiment submits, the controller included, carries these
options. The experiment directory, with the planner environments in it, must
be on a filesystem the compute nodes can see.

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
| `uplab_queries` | saved queries, used in reports as `builtin:<name>` or `plugin:<name>` |
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

`packages/up-lab-stats` is such a plugin: up-lab never imports it, its
`pyproject.toml` registers it with

```toml
[project.entry-points.up_lab]
stats = "up_lab_stats.plugin"
```

## Statistics plugin

`up-lab-stats` adds the planner comparisons of the benchmarking literature.
List the analyses you want under `analyses:` and render their queries (same
names unless noted) under `reports:`:

```yaml
analyses:
  - name: scores                    # built-in; here only to set an option
    options: {agile_base: 300}      # T of the agile score; default: the time limit (the IPC uses 300)
  - name: pairwise-tests
    options: {censor_factor: 1.0}
  - name: portfolio
reports:
  - {name: tests, query: builtin:pairwise-tests, render: csv}
  - {name: portfolio, query: builtin:portfolio, render: latex}
  - {name: anytime, query: builtin:anytime-quality, render: anytime}
```

Variants are compared per request and over tasks. With several seeds, the
runs of a variant on a task are aggregated first: the task is covered if at
least half of the runs are, and its CPU time and cost are the means over the
covered runs. All analyses build on the built-in `reference-cost` and
`scores`, which always run first.

| Analysis | Computes | Options | Follows |
|---|---|---|---|
| `optimality-claims` | `uplab:wrongOptimalityClaim` on every run that claimed optimality: true if its best valid cost exceeds the task's reference cost. Query `wrong-claims` lists them. | | |
| `pairwise-tests` | For every pair of variants, two-sided tests: sign (binomial) test on coverage over the tasks only one side covers; Wilcoxon signed-rank on CPU time over the tasks at least one side covers, the other side counting as `censor_factor` x time limit; Wilcoxon signed-rank on best cost over the double hits. Sample size, wins of each side, statistic and p-value; no p-value when there is no difference. | `censor_factor` (1.0) | IPC 2011 analysis: Linares López, Jiménez Celorrio & García Olaya, AIJ 2015 (after Gerevini et al., AIJ 2009); `scipy.stats.binomtest`, `scipy.stats.wilcoxon` |
| `double-hits` | For every pair of variants: the number of tasks both cover and each side's mean CPU time, cost and quality score on these tasks. | | as above |
| `domain-scores` | Per variant and domain: coverage, domain-normalised coverage (fraction of the domain's tasks covered; overall the mean over domains) and Borda count (in each domain the best of `n` variants by coverage gets `n` points, the worst 1, ties share the mean of their points; summed over domains). | | Vallati, Chrpa & McCluskey, KER 2018 |
| `portfolio` | Virtual best solver per request (a task is covered if any variant covers it, with the best CPU time and cost); each variant's marginal contribution to its coverage (all variants minus all but this one) and Shapley value for the coverage game, exact up to 10 variants, else estimated from `samples` random orders. | `samples` (2000), `seed` (0) | Fréchette et al., AAAI 2016 |
| `lab-scores` | Per run `uplab:labTimeScore` and `uplab:labMemoryScore`: `(log v - log U) / (log L - log U)` with `v` clamped to `[L, U]`, 0 if not covered. Time: total CPU time, `L` = 1 s, `U` = time limit. Memory: peak memory, `L` = 2000 KB, `U` = memory limit. Query `lab-scores` sums them per variant and domain. | | Downward Lab (`score_total_time`, `score_memory`) |

The query `anytime-quality` returns every valid plan with the CPU time at
which it was found, its cost and the reference cost; the renderer `anytime`
plots from it, per variant, the sum over runs of the IPC quality score of the
best plan found so far (reference cost / cost, at most 1) as a step function
of time on a log axis, as PDF and PNG (options `time_limit`, `xscale`,
`xlabel`, `ylabel`, `title`, `figsize`). The terms the analyses add to the
vocabulary are in `up_lab_stats/vocab.ttl`.

## Development

```sh
uv sync
uv run pytest
```
