"""
Planner-comparison analyses from the benchmarking literature.

Comparisons are made per request and over tasks, between variants. With
several seeds, the runs of a variant on a task are aggregated first: the task
is covered if at least half of the runs are, and its CPU time and cost are
the means over the covered runs. A variant without a run on a task of the
request does not cover it.
"""

import math
import random
import warnings
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence
from typing import Tuple

import pandas as pd
import pyoxigraph as ox
from scipy import stats

from up_lab.collect import iri
from up_lab.config import UPLabError
from up_lab.results_plugin import triple
from up_lab.store import UPLAB, experiment_config, query

RDF_TYPE = ox.NamedNode("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
EXACT_SHAPLEY = 10  # variants up to which the Shapley value is exact
LAB_MIN_TIME, LAB_MIN_MEMORY_KB = 1.0, 2000.0  # the best-score bounds of Lab

RUNS = """
    SELECT ?request ?variant ?domain ?task ?coverage ?cpu ?cost ?quality ?limit WHERE {
      ?run a uplab:Run ; uplab:coverage ?coverage ; uplab:task ?t ;
           uplab:request/uplab:name ?request ; uplab:variant/uplab:name ?variant .
      ?t uplab:name ?task ; uplab:domain/uplab:name ?domain .
      OPTIONAL { ?run uplab:cpuTime ?cpu }
      OPTIONAL { ?run uplab:bestCost ?cost }
      OPTIONAL { ?run uplab:qualityScore ?quality }
      OPTIONAL { ?run uplab:timeLimit ?limit }
    }"""


class UPLabStatsError(UPLabError):
    """Raised on invalid options or input of the statistics plugin."""


@dataclass
class Tables:
    """Per-task results of one request: rows `(domain, task)`, columns variants."""

    covered: pd.DataFrame
    cpu: pd.DataFrame  # NaN where not covered
    cost: pd.DataFrame  # NaN where not covered or unknown
    quality: pd.DataFrame
    limit: pd.Series  # time limit per task


def tables(store: ox.Store) -> Iterator[Tuple[str, Tables]]:
    """
    Aggregates the runs over seeds as described in the module docstring.

    :param store: The store, with the graph of the `scores` analysis.
    :return: The name and the per-task tables of every request.
    """
    runs = query(store, RUNS)
    values = ["coverage", "cpu", "cost", "quality", "limit"]
    runs[values] = runs[values].apply(pd.to_numeric)
    solved = runs["coverage"] == 1
    runs[["cpu", "cost"]] = runs[["cpu", "cost"]].where(solved, axis=0)
    keys = ["request", "domain", "task", "variant"]
    for request, part in runs.groupby(keys)[values].mean().groupby(level="request"):
        wide = part.droplevel("request").unstack("variant")
        covered = wide["coverage"].fillna(0.0) >= 0.5
        cpu, cost = wide["cpu"].where(covered), wide["cost"].where(covered)
        limit = wide["limit"].mean(axis=1)
        yield str(request), Tables(covered, cpu, cost, wide["quality"], limit)


def node(subject: str, kind: str, **properties: Any) -> Iterator[ox.Triple]:
    """
    Describes a resource of this plugin.

    :param subject: The IRI of the resource.
    :param kind: The local name of its class.
    :param properties: `uplab:` properties; a `NamedNode` value is a link,
        `None` and NaN are skipped.
    :return: The triples.
    """
    yield ox.Triple(ox.NamedNode(subject), RDF_TYPE, ox.NamedNode(UPLAB + kind))
    for name, value in properties.items():
        if isinstance(value, ox.NamedNode):
            yield ox.Triple(ox.NamedNode(subject), ox.NamedNode(UPLAB + name), value)
        elif value is not None and not (isinstance(value, float) and value != value):
            yield triple(subject, name, value)


def _link(kind: str, *parts: str) -> ox.NamedNode:
    return ox.NamedNode(iri(kind, *parts))


def _option(options: Dict[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise UPLabStatsError(f"Option {name} must be a positive number: {value!r}")
    return value


def _pair(kind: str, exp: str, request: str, first: str, second: str) -> Dict[str, Any]:
    """
    The node of a pair of variants. Every analysis has its own: a triple
    stored in two graphs would match twice in the union of the graphs.
    """
    return {
        "subject": iri(kind.lower(), exp, request, first, second),
        "kind": kind,
        "forRequest": _link("request", exp, request),
        "firstVariant": _link("variant", exp, first),
        "secondVariant": _link("variant", exp, second),
    }


def optimality_claims(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Checks the runs that claimed optimality (`uplab:claimedOptimal true`)
    against the effective reference cost of their task.

    :param store: The store, with the graph of `reference-cost`.
    :param exp_dir: The experiment directory.
    :param options: Unused.
    :return: `uplab:wrongOptimalityClaim` for every such run with a valid
        plan: true iff its best valid cost exceeds the reference cost.
    """
    sparql = """
        SELECT ?run ?best ?ref WHERE {
          ?run uplab:claimedOptimal true ; uplab:bestCost ?best ;
               uplab:task/uplab:effectiveReferenceCost ?ref }"""
    for row in query(store, sparql).itertuples():
        wrong = row.best > row.ref and not math.isclose(row.best, row.ref)
        yield triple(row.run, "wrongOptimalityClaim", bool(wrong))


def _wilcoxon(first: pd.Series, second: pd.Series) -> Dict[str, Any]:
    """Wilcoxon signed-rank test on paired values where lower is better."""
    pairs = pd.concat([first, second], axis=1, keys=["a", "b"]).dropna()
    difference = pairs["a"] - pairs["b"]
    result: Dict[str, Any] = {
        "test": "wilcoxon",
        "sampleSize": len(pairs),
        "winsFirst": int((difference < 0).sum()),
        "winsSecond": int((difference > 0).sum()),
    }
    if (difference != 0).any():  # scipy drops the zero differences
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            outcome = stats.wilcoxon(pairs["a"], pairs["b"])
        result.update(statistic=float(outcome.statistic), pValue=float(outcome.pvalue))
    return result


def pairwise_tests(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Two-sided paired tests for every pair of variants of a request, as in the
    analysis of the IPC 2011 (Linares López, Jiménez Celorrio & García Olaya, AIJ 2015,
    following Gerevini et al., AIJ 2009, for the IPC 2004):

    - `coverage`: sign (binomial) test on the tasks exactly one side covers;
    - `cpu-time`: Wilcoxon signed-rank test on the CPU time over the tasks at
      least one side covers, the other side counting as
      `censor_factor` x time limit;
    - `cost`: Wilcoxon signed-rank test on the best cost over the double
      hits (tasks both sides cover).

    A test without any difference gets its sample size but no p-value.

    :param store: The store, with the graph of `scores`.
    :param exp_dir: The experiment directory.
    :param options: `censor_factor` (default 1.0, as for the IPC 2011).
    :return: A `uplab:Comparison` per pair and a `uplab:Test` per measure
        with `sampleSize`, `winsFirst`, `winsSecond`, `statistic`, `pValue`.
    """
    factor = _option(options, "censor_factor", 1.0)
    exp = experiment_config(exp_dir)["name"]
    for request, t in tables(store):
        for a, b in combinations(t.covered.columns, 2):
            comparison = _pair("Comparison", exp, request, a, b)
            yield from node(**comparison)
            only_a = int((t.covered[a] & ~t.covered[b]).sum())
            only_b = int((t.covered[b] & ~t.covered[a]).sum())
            sign: Dict[str, Any] = {"test": "sign", "sampleSize": only_a + only_b}
            sign.update(winsFirst=only_a, winsSecond=only_b)
            if only_a + only_b:
                p_value = stats.binomtest(only_a, only_a + only_b).pvalue
                sign.update(statistic=float(only_a), pValue=float(p_value))
            either, both = t.covered[a] | t.covered[b], t.covered[a] & t.covered[b]
            ceiling = factor * t.limit
            times = [t.cpu[v].where(t.covered[v], ceiling)[either] for v in (a, b)]
            tests = {
                "coverage": sign,
                "cpu-time": _wilcoxon(*times),
                "cost": _wilcoxon(t.cost[a][both], t.cost[b][both]),
            }
            for measure, result in tests.items():
                subject = iri("test", exp, request, a, b, measure)
                link = ox.NamedNode(comparison["subject"])
                yield from node(
                    subject, "Test", comparison=link, measure=measure, **result
                )


def double_hits(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Compares every pair of variants of a request on their double hits, the
    tasks both cover (Linares López, Jiménez Celorrio & García Olaya, AIJ 2015).

    :param store: The store, with the graph of `scores`.
    :param exp_dir: The experiment directory.
    :param options: Unused.
    :return: A `uplab:DoubleHits` per pair with `doubleHits` and, over these
        tasks, `meanCpuTime`, `meanCost` and `meanQualityScore` of each side
        (suffix `First` / `Second`).
    """
    exp = experiment_config(exp_dir)["name"]
    for request, t in tables(store):
        for a, b in combinations(t.covered.columns, 2):
            both = t.covered[a] & t.covered[b]
            means: Dict[str, Any] = {"doubleHits": int(both.sum())}
            for side, variant in (("First", a), ("Second", b)):
                means["meanCpuTime" + side] = float(t.cpu[variant][both].mean())
                means["meanCost" + side] = float(t.cost[variant][both].mean())
                quality = float(t.quality[variant][both].mean())
                means["meanQualityScore" + side] = quality
            yield from node(**_pair("DoubleHits", exp, request, a, b), **means)


def domain_scores(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Coverage aggregated so that every domain weighs the same.

    - `normalisedCoverage`: the fraction of the tasks of the domain that are
      covered; overall, the mean over the domains.
    - `bordaCount` (Vallati, Chrpa & McCluskey, KER 2018): in every domain
      the `n` variants are ranked by coverage, the best gets `n` points and
      the worst 1, tied variants share the mean of their points; overall,
      the sum over the domains.

    :param store: The store, with the graph of `scores`.
    :param exp_dir: The experiment directory.
    :param options: Unused.
    :return: A `uplab:DomainScore` per request, variant and domain, and one
        with the domain name `TOTAL`, with `coveredTasks`, `tasks`,
        `normalisedCoverage` and `bordaCount`.
    """
    exp = experiment_config(exp_dir)["name"]
    for request, t in tables(store):
        by_domain = t.covered.groupby(level="domain")
        counts, sizes = by_domain.sum(), by_domain.size()
        fraction, borda = counts.div(sizes, axis=0), counts.rank(axis=1)
        rows = [
            (d, counts.loc[d], sizes[d], fraction.loc[d], borda.loc[d])
            for d in sizes.index
        ]
        rows.append(("TOTAL", counts.sum(), sizes.sum(), fraction.mean(), borda.sum()))
        for domain, covered, size, normalised, points in rows:
            for variant in t.covered.columns:
                yield from node(
                    iri("domainscore", exp, request, variant, domain),
                    "DomainScore",
                    forRequest=_link("request", exp, request),
                    forVariant=_link("variant", exp, variant),
                    domainName=str(domain),
                    coveredTasks=int(covered[variant]),
                    tasks=int(size),
                    normalisedCoverage=float(normalised[variant]),
                    bordaCount=float(points[variant]),
                )


def shapley(
    players: Sequence[str],
    value: Callable[[FrozenSet[str]], float],
    samples: int = 2000,
    seed: int = 0,
    exact: int = EXACT_SHAPLEY,
) -> Dict[str, float]:
    """
    Shapley value of every player of a coalitional game: its marginal
    contribution `value(S + i) - value(S)` averaged over all orders in which
    the players can join (Fréchette et al., AAAI 2016, for portfolios).

    :param players: The players.
    :param value: The characteristic function.
    :param samples: The number of random orders used above `exact` players.
    :param seed: The seed of the sampling.
    :param exact: Up to this many players all coalitions are enumerated.
    :return: The (estimated) Shapley value of each player.
    """
    n = len(players)
    phi = dict.fromkeys(players, 0.0)
    if n <= exact:
        for player in players:
            others = [other for other in players if other != player]
            for size in range(n):
                weight = math.factorial(size) * math.factorial(n - size - 1)
                for subset in map(frozenset, combinations(others, size)):
                    gain = value(subset | {player}) - value(subset)
                    phi[player] += weight * gain / math.factorial(n)
        return phi
    rng, order = random.Random(seed), list(players)
    for _ in range(samples):
        rng.shuffle(order)
        joined: FrozenSet[str] = frozenset()
        before = value(joined)
        for player in order:
            joined = joined | {player}
            after = value(joined)
            phi[player] += (after - before) / samples
            before = after
    return phi


def portfolio(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    The virtual best solver (VBS) of every request and what each variant
    contributes to its coverage (Fréchette et al., AAAI 2016).

    The VBS covers a task if any variant does, with the lowest CPU time and
    the lowest cost among them. The marginal contribution of a variant is
    the VBS coverage minus the one of the VBS without it; its Shapley value
    is that of the game whose value is the VBS coverage of a set of variants,
    exact up to 10 variants and estimated from `samples` random orders above.

    :param store: The store, with the graph of `scores`.
    :param exp_dir: The experiment directory.
    :param options: `samples` (default 2000) and `seed` (default 0) of the
        sampling.
    :return: A `uplab:VirtualBestSolver` per request (`coveredTasks`,
        `tasks`, `meanCpuTime`), a `uplab:VirtualBestResult` per covered task
        (`vbsCpuTime`, `vbsCost`) and a `uplab:PortfolioContribution` per
        variant (`coveredTasks`, `marginalContribution`, `shapleyValue`,
        `shapleyExact`).
    """
    samples = int(_option(options, "samples", 2000))
    exp = experiment_config(exp_dir)["name"]
    for request, t in tables(store):
        variants: List[str] = list(t.covered.columns)
        matrix = t.covered.to_numpy()
        cache: Dict[FrozenSet[str], float] = {}

        def coverage(subset: FrozenSet[str]) -> float:
            if subset not in cache:
                columns = [variants.index(variant) for variant in subset]
                cache[subset] = float(matrix[:, columns].any(axis=1).sum())
            return cache[subset]

        everyone = frozenset(variants)
        phi = shapley(variants, coverage, samples, int(options.get("seed", 0)))
        link = _link("request", exp, request)
        best_cpu, best_cost = t.cpu.min(axis=1), t.cost.min(axis=1)
        yield from node(
            iri("vbs", exp, request),
            "VirtualBestSolver",
            forRequest=link,
            coveredTasks=int(coverage(everyone)),
            tasks=len(t.covered),
            meanCpuTime=float(best_cpu.mean()),
        )
        for (_, task), solved in t.covered.any(axis=1).items():
            if solved:
                yield from node(
                    iri("vbsresult", exp, request, task),
                    "VirtualBestResult",
                    forRequest=link,
                    forTask=_link("task", exp, task),
                    vbsCpuTime=float(best_cpu[(_, task)]),
                    vbsCost=float(best_cost[(_, task)]),
                )
        for variant in variants:
            marginal = coverage(everyone) - coverage(everyone - {variant})
            yield from node(
                iri("contribution", exp, request, variant),
                "PortfolioContribution",
                forRequest=link,
                forVariant=_link("variant", exp, variant),
                coveredTasks=int(coverage(frozenset({variant}))),
                marginalContribution=int(marginal),
                shapleyValue=float(phi[variant]),
                shapleyExact=len(variants) <= EXACT_SHAPLEY,
            )


def lab_score(
    value: Optional[float], best: float, worst: Optional[float]
) -> Optional[float]:
    """
    The log-scaled score of Downward Lab (`score_total_time`, `score_memory`
    of its Fast Downward parsers) for a covered run:
    `(log v - log worst) / (log best - log worst)` with `v` clamped to
    `[best, worst]`, hence 1 at `best` or below and 0 at `worst` or above.

    :param value: The measurement, or `None` if unknown.
    :param best: The lower bound `L`.
    :param worst: The upper bound `U` (the limit of the run).
    :return: The score, or `None` when it cannot be computed.
    """
    if value is None or worst is None or worst <= best:
        return None
    clamped = min(max(value, best), worst)
    return (math.log(clamped) - math.log(worst)) / (math.log(best) - math.log(worst))


def lab_scores(
    store: ox.Store, exp_dir: Path, options: Dict[str, Any]
) -> Iterator[ox.Triple]:
    """
    Downward Lab's scores of every run (see :func:`lab_score`), 0 if the run
    is not covered:

    - `uplab:labTimeScore`: on the total CPU time, from 1 s to the time limit;
    - `uplab:labMemoryScore`: on the peak memory, from 2000 KB to the memory
      limit (taking 1 MB = 1024 KB, as Lab does).

    :param store: The store, with the graph of `scores`.
    :param exp_dir: The experiment directory.
    :param options: Unused.
    :return: The score triples of every run.
    """
    sparql = """
        SELECT ?run ?coverage ?cpu ?limit ?peak ?memory WHERE {
          ?run a uplab:Run ; uplab:coverage ?coverage .
          OPTIONAL { ?run uplab:cpuTime ?cpu }
          OPTIONAL { ?run uplab:timeLimit ?limit }
          OPTIONAL { ?run uplab:peakMemoryMb ?peak }
          OPTIONAL { ?run uplab:memoryLimitMb ?memory }
        }"""
    frame = query(store, sparql).astype(object)
    for row in frame.where(frame.notna(), None).itertuples():
        peak = None if row.peak is None else row.peak * 1024
        memory = None if row.memory is None else row.memory * 1024
        scores = {
            "labTimeScore": lab_score(row.cpu, LAB_MIN_TIME, row.limit),
            "labMemoryScore": lab_score(peak, LAB_MIN_MEMORY_KB, memory),
        }
        for name, score in scores.items():
            if not row.coverage:
                yield triple(row.run, name, 0.0)
            elif score is not None:
                yield triple(row.run, name, score)
