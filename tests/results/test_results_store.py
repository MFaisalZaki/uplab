import json
from pathlib import Path

import pyoxigraph as ox
import pytest

from up_lab.analysis import analyze
from up_lab.collect import collect
from up_lab.config import UPLabError
from up_lab.store import BUILTIN_QUERIES, load_query, open_store, query


def one(exp: Path, sparql: str):
    df = query(exp, sparql)
    assert df.shape == (1, 1), df
    return df.iloc[0, 0]


def test_collect_writes_records(exp: Path):
    runs = sorted((exp / "runs").iterdir())
    assert len(runs) == 8
    for run_dir in runs:
        record = json.loads((run_dir / "record.jsonld").read_text())
        assert record["@context"]["uplab"] == "urn:uplab:ns#"
        assert record["@graph"][0]["@id"] == f"urn:uplab:run:demo:{run_dir.name}"
        assert (run_dir / "validation.json").exists()


def test_counts(exp: Path):
    count = "SELECT (COUNT(DISTINCT ?x) AS ?n) WHERE { ?x a %s }"
    expected = {
        "uplab:Experiment": 1,
        "uplab:Run": 8,
        "prov:Activity": 8,
        "uplab:Task": 3,
        "uplab:Domain": 2,
        "uplab:BenchmarkSource": 1,
        "uplab:Planner": 1,
        "uplab:Variant": 2,
        "prov:SoftwareAgent": 2,
        "uplab:Request": 2,
        "uplab:Plan": 8,
        "prov:Entity": 8,
        "uplab:Metric": 1,
    }
    assert {c: one(exp, count % c) for c in expected} == expected
    # no duplicated run: every run has exactly one status
    assert one(exp, "SELECT (COUNT(*) AS ?n) WHERE { ?r uplab:status ?s }") == 8


def test_records_are_in_the_experiment_graph(exp: Path):
    df = query(exp, "SELECT DISTINCT ?g WHERE { GRAPH ?g { ?s ?p ?o } } ORDER BY ?g")
    assert list(df["g"]) == [
        "urn:uplab:graph:demo",
        "urn:uplab:graph:demo:analysis:fast",
        "urn:uplab:graph:demo:analysis:reference-cost",
        "urn:uplab:graph:demo:analysis:scores",
    ]
    n = one(
        exp,
        "SELECT (COUNT(*) AS ?n) WHERE { GRAPH <urn:uplab:graph:demo> { ?r a uplab:Run } }",
    )
    assert n == 8


def test_typed_values(exp: Path):
    df = query(
        exp,
        """SELECT ?run ?status ?outcome ?seed ?k ?bound ?cpu ?wall ?mem ?parse ?first ?limit
                  ?valid ?distinct ?best ?ref ?host ?optimal ?started ?task WHERE {
             ?r uplab:name ?run ; uplab:status ?status ; uplab:outcome ?outcome ;
                uplab:seed ?seed ; uplab:k ?k ; uplab:qualityBound ?bound ;
                uplab:cpuTime ?cpu ; uplab:wallTime ?wall ; uplab:peakMemoryMb ?mem ;
                uplab:parseCpuTime ?parse ; uplab:firstPlanCpuTime ?first ;
                uplab:timeLimit ?limit ; uplab:plansValid ?valid ;
                uplab:plansDistinctValid ?distinct ; uplab:bestCost ?best ;
                uplab:referenceCost ?ref ; uplab:hostname ?host ;
                uplab:claimedOptimal ?optimal ; prov:startedAtTime ?started ;
                uplab:task ?task ; uplab:variant/uplab:name "good" ;
                uplab:request/uplab:name "set" .
           }""",
    )
    assert len(df) == 1
    row = df.iloc[0].to_dict()
    run = row.pop("run")
    assert len(run) == 12
    assert row == {
        "status": "SOLVED",
        "outcome": "SOLVED",
        "seed": 0,
        "k": 3,
        "bound": 2.0,
        "cpu": 0.75,
        "wall": 1.0,
        "mem": 91.5,
        "parse": 0.125,
        "first": 0.2,
        "limit": 100.0,
        "valid": 3,
        "distinct": 2,
        "best": 2.0,
        "ref": 2.0,
        "host": "n01",
        "optimal": False,
        "started": "2026-11-27T10:00:00Z",
        "task": "urn:uplab:task:demo:local/route/p1",
    }
    for key, kind in {
        "seed": int,
        "k": int,
        "valid": int,
        "cpu": float,
        "best": float,
        "optimal": bool,
    }.items():
        assert type(df[key].tolist()[0]) is kind, key
    datatype = """SELECT (DATATYPE(?v) AS ?t) WHERE {
        ?r uplab:%s ?v ; uplab:variant/uplab:name "good" ; uplab:request/uplab:name "set" }"""
    xsd = "http://www.w3.org/2001/XMLSchema#"
    assert one(exp, datatype % "cpuTime") == xsd + "double"
    assert one(exp, datatype % "timeLimit") == xsd + "double"
    assert one(exp, datatype % "seed") == xsd + "integer"
    assert one(exp, datatype % "claimedOptimal") == xsd + "boolean"


def test_outcomes(exp: Path):
    df = query(
        exp,
        """SELECT ?variant ?task ?request ?status ?outcome WHERE {
             ?r uplab:status ?status ; uplab:outcome ?outcome ;
                uplab:variant/uplab:name ?variant ; uplab:task/uplab:name ?task ;
                uplab:request/uplab:name ?request .
             FILTER(?status != "SOLVED" || ?outcome != "SOLVED") } ORDER BY ?task ?request""",
    )
    assert df.values.tolist() == [
        ["bad", "local/plain/q1", "sat", "ERROR", "ERROR"],
        ["bad", "local/route/p1", "set", "TIMEOUT", "TIMEOUT"],
        ["bad", "local/route/p2", "sat", "SOLVED", "INVALID"],
    ]


def test_plan_nodes(exp: Path):
    df = query(
        exp,
        """SELECT ?index ?wall ?cpu ?cost ?length ?valid ?duplicate ?of ?within ?file WHERE {
             ?plan a uplab:Plan ; prov:wasGeneratedBy ?r ; uplab:index ?index ;
                   uplab:foundWallTime ?wall ; uplab:foundCpuTime ?cpu ; uplab:cost ?cost ;
                   uplab:length ?length ; uplab:valid ?valid ; uplab:duplicate ?duplicate ;
                   uplab:withinBound ?within ; uplab:file ?file .
             OPTIONAL { ?plan uplab:duplicateOf/uplab:index ?of }
             ?r uplab:variant/uplab:name "good" ; uplab:request/uplab:name "set" .
           } ORDER BY ?index""",
    )
    df["wall"] = df["wall"].round(6)
    df["cpu"] = df["cpu"].round(6)
    df = df.astype(object).where(df.notna(), None)
    assert df.values.tolist() == [
        [1, 0.2, 0.2, 2.0, 2, True, False, None, True, "plans/00001.plan"],
        [2, 0.3, 0.3, 5.0, 1, True, False, None, False, "plans/00002.plan"],
        [3, 0.4, 0.4, 2.0, 2, True, True, 1, True, "plans/00003.plan"],
    ]
    invalid = "SELECT ?v WHERE { ?p uplab:valid ?v ; prov:wasGeneratedBy/uplab:outcome 'INVALID' }"
    assert one(exp, invalid) is False or one(exp, invalid) == False  # noqa: E712


def test_metric_nodes(exp: Path):
    df = query(
        exp,
        "SELECT ?name ?source ?value WHERE { ?r uplab:metric ?m . ?m uplab:name ?name ; uplab:metricSource ?source ; uplab:value ?value }",
    )
    assert df.values.tolist() == [["expanded", "engine", 7]]


def test_prefix_header_and_explicit_prefixes(exp: Path):
    explicit = (
        "PREFIX uplab: <urn:uplab:ns#> SELECT (COUNT(*) AS ?n) WHERE { ?r a uplab:Run }"
    )
    assert one(exp, explicit) == 8
    with pytest.raises(UPLabError):
        query(exp, "SELECT nonsense")


def test_columns_follow_select_order(exp: Path):
    df = query(exp, "SELECT ?z ?a ?m WHERE { BIND(1 AS ?z) BIND('x' AS ?a) }")
    assert list(df.columns) == ["z", "a", "m"]
    assert df.iloc[0, 0] == 1 and df.iloc[0, 1] == "x" and df["m"].isna().all()


def test_collect_is_idempotent_and_rebuilds_the_store(exp_copy: Path):
    import shutil

    shutil.rmtree(exp_copy / "store")
    stamp = {p: p.stat().st_mtime_ns for p in exp_copy.glob("runs/*/validation.json")}
    assert collect(exp_copy, jobs=2) == 8
    assert stamp == {p: p.stat().st_mtime_ns for p in stamp}  # not validated again
    assert collect(exp_copy) == 8  # reloading clears the graph first
    assert one(exp_copy, "SELECT (COUNT(*) AS ?n) WHERE { ?r uplab:status ?s }") == 8
    assert query(exp_copy, "SELECT ?r WHERE { ?r uplab:coverage ?c }").empty


def test_collect_validates_in_parallel(tmp_path: Path):
    from conftest import write_experiment  # noqa: PLC0415

    exp_dir = write_experiment(tmp_path / "demo")
    assert collect(exp_dir, jobs=2) == 8
    assert one(exp_dir, "SELECT (COUNT(*) AS ?n) WHERE { ?p a uplab:Plan }") == 8


def test_read_only_store_requires_collect(tmp_path: Path):
    with pytest.raises(UPLabError):
        open_store(tmp_path, read_only=True)


def test_builtin_queries_exist():
    for name in BUILTIN_QUERIES:
        assert "SELECT" in load_query(f"builtin:{name}")
    with pytest.raises(UPLabError):
        load_query("builtin:nope")
