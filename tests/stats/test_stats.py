import math
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from scipy import stats
from typer.testing import CliRunner

import stats_fixtures as fixtures

from up_lab.analysis import analyze
from up_lab.cli import app
from up_lab.collect import collect
from up_lab.config import UPLabError
from up_lab.plugins import hook
from up_lab.store import load_query, query
from up_lab_stats.analyses import lab_score, shapley
from up_lab_stats.render import anytime_steps

A = "local/route/p1"


@pytest.fixture(scope="session")
def stats_exp(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The experiment of `stats_fixtures`, collected and analyzed once."""
    exp_dir = fixtures.write_experiment(tmp_path_factory.mktemp("stats") / "demo")
    assert collect(exp_dir) == 17
    analyze(exp_dir)
    return exp_dir


def table(exp: Path, name: str, request: str = "sat") -> pd.DataFrame:
    df = query(exp, load_query(f"builtin:{name}"))
    if "request" in df.columns:
        df = df[df["request"] == request].drop(columns="request")
    return df.reset_index(drop=True)


def records(df: pd.DataFrame) -> list:
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].round(9)
    return df.astype(object).where(df.notna(), None).values.tolist()


def test_entry_point_is_discovered():
    code = (
        "import sys; from up_lab.plugins import hook;"
        "assert 'up_lab_stats' not in sys.modules;"
        "assert 'pairwise-tests' in hook('uplab_analyses');"
        "assert 'portfolio' in hook('uplab_queries');"
        "assert 'anytime' in hook('uplab_renderers')"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_plugin_queries_resolve():
    assert "DomainScore" in load_query("builtin:domain-scores")
    assert "DomainScore" in load_query("plugin:domain-scores")
    assert "?solved" in load_query("builtin:coverage")  # up-lab's own
    with pytest.raises(UPLabError):
        load_query("plugin:coverage")  # not a plugin query


def test_wrong_optimality_claims(stats_exp: Path):
    # B claims optimality twice: with cost 5 where A found 2, and rightly on p3.
    sparql = """SELECT ?task ?wrong WHERE { ?r uplab:wrongOptimalityClaim ?wrong ;
      uplab:task/uplab:name ?task } ORDER BY ?task"""
    assert records(query(stats_exp, sparql)) == [
        ["local/route/p1", True],
        ["local/route/p3", False],
    ]
    wrong = table(stats_exp, "wrong-claims").drop(columns="run")
    assert records(wrong) == [["B", "local/route/p1", 5.0, 2.0, "manifest"]]


def test_pairwise_tests(stats_exp: Path):
    df = table(stats_exp, "pairwise-tests").set_index(["first", "second", "measure"])
    counts = df[["test", "n", "wins_first", "wins_second"]].T.to_dict("list")
    assert counts == {
        # only A covers q2, only B covers p3
        ("A", "B", "coverage"): ["sign", 2, 1, 1],
        # A - B: -9, -2, 100 - 20, 0, 8 - 100 (uncovered = the 100 s limit)
        ("A", "B", "cpu-time"): ["wilcoxon", 5, 3, 1],
        # double hits p1, p2, q1: 2 < 5, 4 < 7, 2 = 2
        ("A", "B", "cost"): ["wilcoxon", 3, 2, 0],
        ("A", "C", "coverage"): ["sign", 1, 1, 0],  # p1
        ("A", "C", "cpu-time"): ["wilcoxon", 4, 2, 2],  # p3: covered by neither
        ("A", "C", "cost"): ["wilcoxon", 3, 1, 0],
        ("B", "C", "coverage"): ["sign", 3, 2, 1],  # p1, p3 against q2
        ("B", "C", "cpu-time"): ["wilcoxon", 5, 3, 2],
        ("B", "C", "cost"): ["wilcoxon", 2, 0, 0],  # p2, q1: equal costs
    }
    expected = {
        ("A", "B", "coverage"): stats.binomtest(1, 2).pvalue,
        ("A", "C", "coverage"): stats.binomtest(1, 1).pvalue,
        ("B", "C", "coverage"): stats.binomtest(2, 3).pvalue,
        ("A", "B", "cpu-time"): stats.wilcoxon(
            [1, 2, 100, 5, 8], [10, 4, 20, 5, 100]
        ).pvalue,
        ("A", "C", "cpu-time"): stats.wilcoxon([1, 2, 5, 8], [100, 50, 1, 3]).pvalue,
        ("A", "B", "cost"): stats.wilcoxon([2, 4, 2], [5, 7, 2]).pvalue,
    }
    for key, p_value in expected.items():
        assert df.loc[key, "p_value"] == pytest.approx(p_value), key
    assert df.loc[("B", "C", "coverage"), "statistic"] == 2
    assert df.loc[("A", "B", "cpu-time"), "statistic"] == 3  # rank of the +80
    # no difference at all: the sample size is reported, no p-value
    assert pd.isna(df.loc[("B", "C", "cost"), "p_value"])
    assert pd.isna(df.loc[("B", "C", "cost"), "statistic"])


def test_requests_are_not_mixed(stats_exp: Path):
    df = table(stats_exp, "pairwise-tests", request="set")
    df = df.set_index("measure")[["first", "second", "n", "wins_first", "p_value"]]
    assert records(df.loc[["coverage", "cost"]]) == [
        ["A", "B", 1, 1, 1.0],
        ["A", "B", 0, 0, None],  # no double hit
    ]
    assert len(table(stats_exp, "portfolio", request="set")) == 2  # no C


def test_censor_factor(stats_exp: Path, tmp_path: Path):
    exp = Path(shutil.copytree(stats_exp, tmp_path / "demo"))
    text = (exp / "experiment.yaml").read_text()
    option = "- name: pairwise-tests\n  options: {censor_factor: 0.03}\n"
    (exp / "experiment.yaml").write_text(
        text.replace("- name: pairwise-tests\n", option)
    )
    analyze(exp, names=["pairwise-tests"])
    df = table(exp, "pairwise-tests").set_index(["first", "second", "measure"])
    # Uncovered runs now count as 3 s: A - B = -9, -2, 3 - 20, 0, 8 - 3. The
    # zero is dropped; the ranks of |d| are 2: 1, 5: 2, 9: 3, 17: 4, so the
    # positive differences sum to 2 (with the 100 s limit: 80: 3, hence 3).
    assert df.loc[("A", "B", "cpu-time"), "statistic"] == 2
    expected = stats.wilcoxon([1, 2, 3, 5, 8], [10, 4, 20, 5, 3]).pvalue
    assert df.loc[("A", "B", "cpu-time"), "p_value"] == pytest.approx(expected)
    (exp / "experiment.yaml").write_text(
        text.replace(
            "- name: portfolio\n", "- name: portfolio\n  options: {samples: 0}\n"
        )
    )
    with pytest.raises(UPLabError):
        analyze(exp, names=["portfolio"])


def test_double_hits(stats_exp: Path):
    df = table(stats_exp, "double-hits")
    assert records(df) == [
        # p1, p2, q1: CPU (1+2+5)/3 and (10+4+5)/3, costs (2+4+2)/3 and
        # (5+7+2)/3, quality 1 and (2/5 + 4/7 + 1)/3
        ["A", "B", 3, *(round(x, 9) for x in (8 / 3, 19 / 3, 8 / 3, 14 / 3))]
        + [1.0, round((2 / 5 + 4 / 7 + 1) / 3, 9)],
        # p2, q1, q2: CPU (2+5+8)/3 and (50+1+3)/3, costs (4+2+2)/3, (7+2+2)/3
        ["A", "C", 3, 5.0, 18.0, round(8 / 3, 9), round(11 / 3, 9)]
        + [1.0, round((4 / 7 + 2) / 3, 9)],
        # p2, q1: CPU (4+5)/2 and (50+1)/2, both costs (7+2)/2
        ["B", "C", 2, 4.5, 25.5, 4.5, 4.5] + [round((4 / 7 + 1) / 2, 9)] * 2,
    ]


def test_domain_scores(stats_exp: Path):
    # Coverage    route (3 tasks)  plain (2 tasks)
    #   A         2                2
    #   B         3                1
    #   C         1                2
    # Borda, 3 variants: route B 3, A 2, C 1; plain: A and C tie for the
    # first place and share (3 + 2) / 2 = 2.5 points each, B gets 1.
    assert records(table(stats_exp, "domain-scores")) == [
        ["A", "plain", 2, 2, 1.0, 2.5],
        ["A", "route", 2, 3, round(2 / 3, 9), 2.0],
        ["A", "TOTAL", 4, 5, round((2 / 3 + 1) / 2, 9), 4.5],
        ["B", "plain", 1, 2, 0.5, 1.0],
        ["B", "route", 3, 3, 1.0, 3.0],
        ["B", "TOTAL", 4, 5, 0.75, 4.0],
        ["C", "plain", 2, 2, 1.0, 2.5],
        ["C", "route", 1, 3, round(1 / 3, 9), 1.0],
        ["C", "TOTAL", 3, 5, round((1 / 3 + 1) / 2, 9), 3.5],
    ]


def test_portfolio(stats_exp: Path):
    # Covering variants: p1 {A,B}, p2 {A,B,C}, p3 {B}, q1 {A,B,C}, q2 {A,C}.
    # Only p3 is lost when a variant (B) is removed. For coverage, the Shapley
    # value splits every task evenly among the variants covering it:
    #   A = 1/2 + 1/3 + 1/3 + 1/2 = 5/3
    #   B = 1/2 + 1/3 + 1 + 1/3   = 13/6
    #   C = 1/3 + 1/3 + 1/2       = 7/6          (sum: 5 = VBS coverage)
    assert records(table(stats_exp, "portfolio")) == [
        ["B", 4, 1, round(13 / 6, 9), True, 5, 5],
        ["A", 4, 0, round(5 / 3, 9), True, 5, 5],
        ["C", 3, 0, round(7 / 6, 9), True, 5, 5],
    ]
    sparql = """SELECT ?task ?cpu ?cost WHERE { ?v a uplab:VirtualBestResult ;
      uplab:forRequest/uplab:name "sat" ; uplab:forTask/uplab:name ?task ;
      uplab:vbsCpuTime ?cpu ; uplab:vbsCost ?cost } ORDER BY ?task"""
    assert records(query(stats_exp, sparql)) == [
        ["local/plain/q1", 1.0, 2.0],
        ["local/plain/q2", 3.0, 2.0],
        ["local/route/p1", 1.0, 2.0],
        ["local/route/p2", 2.0, 4.0],
        ["local/route/p3", 20.0, 2.0],
    ]
    sparql = """SELECT ?mean WHERE { ?v a uplab:VirtualBestSolver ;
      uplab:forRequest/uplab:name "sat" ; uplab:meanCpuTime ?mean }"""
    assert records(query(stats_exp, sparql)) == [[5.4]]  # (1 + 3 + 1 + 2 + 20) / 5


def test_shapley():
    # Two players, v({}) = 0, v({x}) = 1, v({y}) = 2, v({x, y}) = 4.
    # Order x, y: x adds 1, y adds 3; order y, x: y adds 2, x adds 2.
    # Hence x = (1 + 2) / 2 = 1.5 and y = (3 + 2) / 2 = 2.5.
    values = {frozenset(): 0, frozenset("x"): 1, frozenset("y"): 2, frozenset("xy"): 4}
    assert shapley(["x", "y"], values.__getitem__) == {"x": 1.5, "y": 2.5}
    # Sampling (forced with exact=0) converges to the same values and
    # always distributes v(all).
    sampled = shapley(["x", "y"], values.__getitem__, samples=4000, seed=1, exact=0)
    assert sampled["x"] == pytest.approx(1.5, abs=0.05)
    assert sum(sampled.values()) == pytest.approx(4)
    assert sampled == shapley(["x", "y"], values.__getitem__, 4000, seed=1, exact=0)


def test_lab_score_formula():
    assert lab_score(0.5, 1, 100) == 1  # at or below the lower bound
    assert lab_score(1, 1, 100) == 1
    assert lab_score(100, 1, 100) == 0  # at or above the upper bound
    assert lab_score(250, 1, 100) == 0
    assert lab_score(10, 1, 100) == pytest.approx(0.5)  # log scale
    assert lab_score(None, 1, 100) is None and lab_score(5, 1, None) is None


def test_lab_scores(stats_exp: Path):
    sparql = """SELECT ?variant ?task ?time ?memory WHERE {
      ?r uplab:labTimeScore ?time ; uplab:labMemoryScore ?memory ;
         uplab:variant/uplab:name ?variant ; uplab:task/uplab:name ?task ;
         uplab:request/uplab:name "sat" }"""
    df = query(stats_exp, sparql).set_index(["variant", "task"])
    assert len(df) == 15
    default = (math.log(91.5 * 1024) - math.log(2048 * 1024)) / (
        math.log(2000) - math.log(2048 * 1024)
    )
    expected = {
        ("A", "local/route/p1"): (1.0, 1.0),  # 1 s; 1 MB is below 2000 KB
        ("B", "local/route/p1"): (0.5, 0.0),  # 10 s of 100 s; memory at the limit
        ("C", "local/route/p1"): (0.0, 0.0),  # not covered
        ("C", "local/route/p2"): (1 - math.log(50) / math.log(100), default),
        ("A", "local/route/p3"): (0.0, 0.0),  # error after 0.25 s
    }
    for key, scores in expected.items():
        assert tuple(df.loc[key]) == pytest.approx(scores), key
    sums = query(stats_exp, load_query("builtin:lab-scores")).set_index(
        ["variant", "domain"]
    )
    # C: 50 s, 1 s and 3 s (sat); the request `set` adds nothing for C
    total = 1 - math.log(50) / math.log(100) + 1 + 1 - math.log(3) / math.log(100)
    assert sums.loc[("C", "TOTAL"), "time_score"] == pytest.approx(total)
    assert sums.loc[("C", "TOTAL"), "memory_score"] == pytest.approx(3 * default)


def test_anytime_steps(stats_exp: Path, tmp_path: Path):
    df = table(stats_exp, "anytime-quality")
    assert len(df) == 12  # 11 covered runs, one of them with two plans
    steps = anytime_steps(df)
    # A: p1 (cost 2 = reference) at 0.5 s; p2 costs 7 at 1.0 s (4/7) and the
    # reference 4 at 1.1 s; q1 at 5 s and q2 at 8 s.
    assert records(steps[steps["variant"] == "A"][["time", "score"]]) == [
        [0.5, 1.0],
        [1.0, round(1 + 4 / 7, 9)],
        [1.1, 2.0],
        [5.0, 3.0],
        [8.0, 4.0],
    ]
    # B: p2 (4/7) at 4 s, q1 (1) at 5 s, p1 (2/5) at 10 s, p3 (1) at 20 s
    assert records(steps[steps["variant"] == "B"][["time", "score"]]) == [
        [4.0, round(4 / 7, 9)],
        [5.0, round(4 / 7 + 1, 9)],
        [10.0, round(4 / 7 + 1 + 2 / 5, 9)],
        [20.0, round(4 / 7 + 2 + 2 / 5, 9)],
    ]
    written = hook("uplab_renderers")["anytime"](df, tmp_path / "any", {})
    assert [path.name for path in written] == ["any.pdf", "any.png"]
    assert all(path.stat().st_size > 0 for path in written)
    assert (
        len(hook("uplab_renderers")["anytime"](df.iloc[:0], tmp_path / "no", {})) == 2
    )
    with pytest.raises(UPLabError):
        anytime_steps(df.drop(columns="reference_cost"))


def test_seeds_are_aggregated_per_task(tmp_path: Path):
    exp = tmp_path / "demo"
    fixtures.write_config(exp, ["A", "B"])
    # A is covered in 2 of 3 runs (CPU 2 s and 4 s), B in 1 of 3.
    long, short = [fixtures.LONG], [fixtures.SHORT]
    fixtures.write_run(exp, A, "A", seed=2, status="TIMEOUT", cpu=100.0)
    fixtures.write_run(exp, A, "A", seed=1, plans=long, cpu=4.0)
    fixtures.write_run(exp, A, "A", seed=0, plans=long, cpu=2.0)
    fixtures.write_run(exp, A, "B", seed=2, status="TIMEOUT", cpu=100.0)
    fixtures.write_run(exp, A, "B", seed=1, status="TIMEOUT", cpu=100.0)
    fixtures.write_run(exp, A, "B", seed=0, plans=short, cpu=1.0)
    assert collect(exp) == 6
    analyze(exp)
    assert records(table(exp, "domain-scores")[["variant", "domain", "coverage"]]) == [
        ["A", "route", 1],
        ["A", "TOTAL", 1],
        ["B", "route", 0],
        ["B", "TOTAL", 0],
    ]
    assert records(table(exp, "double-hits")) == [["A", "B", 0] + [None] * 6]
    tests = table(exp, "pairwise-tests").set_index("measure")
    assert list(tests.loc["coverage", ["n", "wins_first"]]) == [1, 1]
    # mean CPU time of A's covered runs, 3 s, against B's 100 s limit
    expected = stats.wilcoxon([3.0], [100.0])
    assert tests.loc["cpu-time", "statistic"] == expected.statistic
    assert tests.loc["cpu-time", "p_value"] == pytest.approx(expected.pvalue)
    sparql = (
        "SELECT ?cpu WHERE { ?v a uplab:VirtualBestResult ; uplab:vbsCpuTime ?cpu }"
    )
    assert records(query(exp, sparql)) == [[3.0]]


def test_builtin_scores_take_options(stats_exp: Path, tmp_path: Path):
    exp = Path(shutil.copytree(stats_exp, tmp_path / "demo"))
    sparql = """SELECT ?agile WHERE { ?r uplab:agileScore ?agile ;
      uplab:variant/uplab:name "B" ; uplab:task/uplab:name "local/route/p1" ;
      uplab:request/uplab:name "sat" }"""
    assert records(query(exp, sparql)) == [[0.5]]  # 10 s, time limit 100 s
    text = (exp / "experiment.yaml").read_text()
    option = "analyses:\n- name: scores\n  options: {agile_base: 1000}\n"
    (exp / "experiment.yaml").write_text(text.replace("analyses:\n", option))
    counts = analyze(exp)
    # built-ins still run first, whatever the order of `analyses:`
    assert list(counts)[:2] == ["reference-cost", "scores"]
    assert records(query(exp, sparql)) == [[round(2 / 3, 9)]]  # 1 - log 10 / log 1000


def test_cli_analyze_and_report(stats_exp: Path, tmp_path: Path):
    exp = Path(shutil.copytree(stats_exp, tmp_path / "demo"))
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", str(exp)])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["report", str(exp)])
    assert result.exit_code == 0, result.output
    names = {path.name for path in (exp / "reports").iterdir()}
    assert names == {
        "wrong-claims.csv",
        "pairwise-tests.csv",
        "double-hits.csv",
        "domain-scores.csv",
        "portfolio.csv",
        "lab-scores.tex",
        "anytime.pdf",
        "anytime.png",
    }
    csv = pd.read_csv(exp / "reports" / "portfolio.csv")
    assert list(csv[csv["request"] == "sat"]["variant"]) == ["B", "A", "C"]
