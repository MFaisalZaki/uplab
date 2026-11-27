from pathlib import Path

import pytest

from up_lab.analysis import analyze
from up_lab.collect import collect
from up_lab.config import UPLabError
from up_lab.store import load_query, query

SCORES = """SELECT ?variant ?task ?request ?coverage ?quality ?agile ?par10 WHERE {
  ?r uplab:variant/uplab:name ?variant ; uplab:task/uplab:name ?task ;
     uplab:request/uplab:name ?request ; uplab:coverage ?coverage ; uplab:par10 ?par10 .
  OPTIONAL { ?r uplab:qualityScore ?quality }
  OPTIONAL { ?r uplab:agileScore ?agile }
} ORDER BY ?task ?request ?variant"""


def rows(exp: Path, sparql: str) -> list:
    df = query(exp, sparql)
    numeric = df.select_dtypes("number").columns
    df[numeric] = df[numeric].round(9)
    return df.astype(object).where(df.notna(), None).values.tolist()


def test_reference_cost(exp: Path):
    sparql = """SELECT ?task ?cost ?source WHERE {
      GRAPH <urn:uplab:graph:demo:analysis:reference-cost> {
        ?t uplab:effectiveReferenceCost ?cost ; uplab:referenceCostSource ?source }
      ?t uplab:name ?task } ORDER BY ?task"""
    assert rows(exp, sparql) == [
        ["local/plain/q1", 2.0, "best-known"],  # only run: length 2
        ["local/route/p1", 2.0, "manifest"],
        ["local/route/p2", 4.0, "best-known"],  # the invalid plan (cost 2) is ignored
    ]


def test_scores(exp: Path):
    # time limit 100 s: a first plan after 10 s CPU scores 1 - log(10)/log(100) = 0.5
    assert rows(exp, SCORES) == [
        ["bad", "local/plain/q1", "sat", 0, 0.0, 0.0, 1000.0],  # ERROR
        ["good", "local/plain/q1", "sat", 1, 1.0, 0.5, 50.0],
        ["bad", "local/route/p1", "sat", 1, 0.4, 0.5, 12.0],  # cost 5, reference 2
        ["good", "local/route/p1", "sat", 1, 1.0, 1.0, 0.5],
        ["bad", "local/route/p1", "set", 0, 0.0, 0.0, 1000.0],  # TIMEOUT
        ["good", "local/route/p1", "set", 1, 1.0, 1.0, 0.75],
        ["bad", "local/route/p2", "sat", 0, 0.0, 0.0, 1000.0],  # INVALID
        ["good", "local/route/p2", "sat", 1, 1.0, 0.5, 12.0],
    ]
    assert query(exp, "SELECT ?r WHERE { ?r uplab:coverageUnvalidated ?u }").empty


def test_construct_analysis(exp: Path):
    sparql = """SELECT ?variant ?task ?request WHERE {
      GRAPH <urn:uplab:graph:demo:analysis:fast> { ?r uplab:fast true }
      ?r uplab:variant/uplab:name ?variant ; uplab:task/uplab:name ?task ;
         uplab:request/uplab:name ?request } ORDER BY ?variant ?request"""
    assert rows(exp, sparql) == [
        ["bad", "local/plain/q1", "sat"],
        ["good", "local/route/p1", "sat"],
        ["good", "local/route/p1", "set"],
    ]


def test_analyze_replaces_graphs_and_selects(exp_copy: Path):
    before = len(query(exp_copy, "SELECT * WHERE { ?s ?p ?o }"))
    assert analyze(exp_copy) == {"reference-cost": 6, "scores": 32, "fast.rq": 3}
    assert analyze(exp_copy, names=["scores"]) == {"scores": 32}
    assert len(query(exp_copy, "SELECT * WHERE { ?s ?p ?o }")) == before
    with pytest.raises(UPLabError):
        analyze(exp_copy, names=["nope"])


def test_unvalidated_plans_count_as_solved_but_are_flagged(exp_copy: Path, make_run):
    make_run(exp_copy, "local/plain/q1", "other", plans=["fly(a)\n"], cpu=0.5)
    assert collect(exp_copy) == 9
    analyze(exp_copy)
    sparql = """SELECT ?coverage ?flag ?quality WHERE {
      ?r uplab:variant/uplab:name "other" ; uplab:coverage ?coverage ;
         uplab:coverageUnvalidated ?flag . OPTIONAL { ?r uplab:qualityScore ?quality } }"""
    assert rows(exp_copy, sparql) == [[1, True, None]]


def test_builtin_coverage(exp: Path):
    assert rows(exp, load_query("builtin:coverage")) == [
        ["bad", "plain", 0, 1],
        ["bad", "route", 1, 3],
        ["bad", "TOTAL", 1, 4],
        ["good", "plain", 1, 1],
        ["good", "route", 3, 3],
        ["good", "TOTAL", 4, 4],
    ]


def test_builtin_scores(exp: Path):
    df = query(exp, load_query("builtin:scores"))
    assert list(df.columns) == ["variant", "domain", "coverage", "quality", "agile"]
    assert rows(exp, load_query("builtin:scores")) == [
        ["bad", "plain", 0, 0.0, 0.0],
        ["bad", "route", 1, 0.4, 0.5],
        ["bad", "TOTAL", 1, 0.4, 0.5],
        ["good", "plain", 1, 1.0, 0.5],
        ["good", "route", 3, 3.0, 2.5],
        ["good", "TOTAL", 4, 4.0, 3.0],
    ]


def test_builtin_runtimes(exp: Path):
    df = query(exp, load_query("builtin:runtimes"))
    assert list(df.columns) == [
        "run", "variant", "task", "domain", "request", "seed", "outcome",
        "cpu_time", "wall_time", "first_plan_cpu_time",
    ]
    assert len(df) == 8 and df["run"].is_unique
    got = df[["variant", "task", "request", "outcome", "cpu_time", "wall_time"]]
    assert got.values.tolist() == [
        ["bad", "local/plain/q1", "sat", "ERROR", 0.25, 0.5],
        ["bad", "local/route/p1", "sat", "SOLVED", 12.0, 12.25],
        ["bad", "local/route/p1", "set", "TIMEOUT", 100.0, 100.25],
        ["bad", "local/route/p2", "sat", "INVALID", 3.0, 3.25],
        ["good", "local/plain/q1", "sat", "SOLVED", 50.0, 50.25],
        ["good", "local/route/p1", "sat", "SOLVED", 0.5, 0.75],
        ["good", "local/route/p1", "set", "SOLVED", 0.75, 1.0],
        ["good", "local/route/p2", "sat", "SOLVED", 12.0, 12.25],
    ]
    assert df["first_plan_cpu_time"].isna().sum() == 2  # the runs without plans


def test_builtin_errors(exp: Path):
    df = query(exp, load_query("builtin:errors")).drop(columns="run")
    assert df.astype(object).where(df.notna(), None).values.tolist() == [
        ["bad", "local/plain/q1", "sat", 0, "ERROR", 1, "boom"],
        ["bad", "local/route/p2", "sat", 0, "INVALID", 0, None],
    ]


def test_builtin_plansets(exp: Path):
    df = query(exp, load_query("builtin:plansets")).drop(columns="run")
    assert list(df.columns) == [
        "variant", "task", "request", "k", "outcome", "returned", "valid",
        "distinct_valid", "best_cost",
    ]
    assert df.astype(object).where(df.notna(), None).values.tolist() == [
        ["bad", "local/plain/q1", "sat", 1, "ERROR", 0, 0, 0, None],
        ["bad", "local/route/p1", "sat", 1, "SOLVED", 1, 1, 1, 5.0],
        ["bad", "local/route/p1", "set", 3, "TIMEOUT", 0, 0, 0, None],
        ["bad", "local/route/p2", "sat", 1, "INVALID", 1, 0, 0, None],
        ["good", "local/plain/q1", "sat", 1, "SOLVED", 1, 1, 1, 2.0],
        ["good", "local/route/p1", "sat", 1, "SOLVED", 1, 1, 1, 2.0],
        ["good", "local/route/p1", "set", 3, "SOLVED", 3, 3, 2, 2.0],
        ["good", "local/route/p2", "sat", 1, "SOLVED", 1, 1, 1, 4.0],
    ]


def test_builtin_anytime(exp: Path):
    df = query(exp, load_query("builtin:anytime"))
    assert len(df) == 8
    plans = df[(df["variant"] == "good") & (df["request"] == "set")]
    got = plans[["index", "cpu_time", "cost", "valid", "duplicate"]].round(6)
    assert got.values.tolist() == [
        [1, 0.2, 2.0, True, False],
        [2, 0.3, 5.0, True, False],
        [3, 0.4, 2.0, True, True],
    ]
