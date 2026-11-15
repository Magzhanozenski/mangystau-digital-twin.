"""Сеть по открытым источникам (шаг 2.2, пункт 5). Без сети: искусственные примеры и сохранённые файлы."""
from pathlib import Path

import pytest
import yaml

from twin import build_real as br
from twin import load_network, run_power_flow

ROOT = Path(__file__).resolve().parent.parent
REAL = ROOT / "data" / "network_real.yaml"


def _way(wid, nodes, tags=None):
    t = {"power": "line", "voltage": "110000", **(tags or {})}
    return {"type": "way", "id": wid, "nodes": nodes, "tags": t,
            "geometry": [{"lat": 45.0 + 0.01 * n, "lon": 52.0} for n in nodes]}


def _seg(wid, sub1="", sub2=""):
    return {"way": str(wid), "kv": "110.0", "kv_basis": "тег voltage", "km": "1", "flags": "", "sub1": sub1, "sub2": sub2}


def _components(edges):
    par = {}

    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            x = par[x]
        return x
    for e in edges:
        par[f(e["p"])] = f(e["q"])
    return {f(e["p"]) for e in edges}


def test_circuits_of():
    assert br.circuits_of({"circuits": "2"})[:2] == (2, "источник")
    assert br.circuits_of({"cables": "6"})[:2] == (2, "допущение")
    assert br.circuits_of({})[:2] == (1, "допущение")


def test_tap_on_inner_node_connects():
    """Ответвление к внутреннему узлу другой линии — одна компонента, узел ответвления — терминал."""
    snap = {"elements": [_way(1, [1, 2, 3]), _way(2, [2, 9])]}
    edges, _ = br.graph_110(snap, [_seg(1, "way/A", "way/B"), _seg(2, "", "way/C")])
    assert len(_components(edges)) == 1
    assert len(edges) == 3 and all(("n", 2) in (e["p"], e["q"]) for e in edges)


def test_chain_merge_keeps_connectivity():
    """Две параллельные ветви между узлами 5–6 и линии от обоих узлов — связь не теряется при склейке."""
    snap = {"elements": [_way(10, [5, 7, 6]), _way(11, [5, 8, 6]), _way(12, [5, 4]), _way(13, [6, 3])]}
    segs = [_seg(10), _seg(11), _seg(12, "", "way/A"), _seg(13, "", "way/B")]
    edges, _ = br.graph_110(snap, segs)
    assert len(_components(edges)) == 1
    assert sum(1 for e in edges if {e["p"], e["q"]} == {("n", 5), ("n", 6)}) == 2


def test_continuation_merged_into_one_line():
    snap = {"elements": [_way(1, [1, 2]), _way(2, [2, 3], {"circuits": "1"})]}
    edges, _ = br.graph_110(snap, [_seg(1, "way/A", ""), _seg(2, "", "way/B")])
    assert len(edges) == 1 and edges[0]["ways"] == [1, 2]


@pytest.fixture(scope="module")
def real():
    if not REAL.exists():
        pytest.skip("нет data/network_real.yaml")
    return yaml.safe_load(REAL.read_text(encoding="utf-8")), REAL.read_text(encoding="utf-8")


def test_header_odbl(real):
    _, text = real
    head = text.split("\nmeta:")[0]
    assert "ODbL" in head and "openstreetmap.org/copyright" in head and "© участники OpenStreetMap" in head


def test_marks_and_sources(real):
    d, _ = real
    load_network(REAL)  # пометки status/source у элементов
    for sec in ("buses", "lines", "generators"):
        for item in d[sec]:
            assert item["marks"], (sec, item["id"])
            for field, m in item["marks"].items():
                assert m["status"] in ("источник", "допущение"), (sec, item["id"], field)
                if m["status"] == "источник":
                    assert str(m.get("source", "")).startswith("http"), (sec, item["id"], field)
                else:
                    assert m.get("basis"), (sec, item["id"], field)
            if item["status"] == "источник":
                assert all(m["status"] == "источник" for m in item["marks"].values())


def test_only_tagged_lines_and_voltage_levels(real):
    """Линии — только с тегом напряжения (110 или 220 кВ), и класс линии совпадает с классом её шин (шаг 2.4)."""
    d, _ = real
    seg = {s["way"]: s for s in br._csv("osm_segments.csv")}
    kv = {b["id"]: b["vn_kv"] for b in d["buses"]}
    for l in d["lines"]:
        for w in l["osm_ways"].split(";"):
            assert seg[w]["kv_basis"] == "тег voltage", (l["id"], w)
            assert float(seg[w]["kv"]) == kv[l["from"]] == kv[l["to"]], (l["id"], w)
    assert d["meta"]["base_kv"] == 110
    assert {220, 110, 10} <= set(kv.values())


def test_transformers_marked(real):
    d, _ = real
    kv = {b["id"]: b["vn_kv"] for b in d["buses"]}
    assert d["transformers"]
    for t in d["transformers"]:
        assert kv[t["from"]] > kv[t["to"]]
        for field in ("sn_mva", "params", "ratio"):
            m = t["marks"][field]
            assert m["status"] in ("источник", "допущение") and (m.get("source") or m.get("basis"))
    sns = {t["id"]: t["sn_mva"] for t in d["transformers"]}
    assert sns["AT-Каражанбас-1"] == 150 and sns["AT-Бейнеу-1"] == 63  # мощности из источников


def test_circuits_as_parallel_lines(real):
    d, _ = real
    by_route = {}
    for l in d["lines"]:
        by_route.setdefault(l["id"].rsplit("-", 1)[0], []).append(l)
    for lst in by_route.values():
        assert len(lst) == lst[0]["marks"]["circuits"]["value"]


def test_shares_and_islands(real):
    d, _ = real
    assert sum(b["share"] for b in d["buses"]) == pytest.approx(1.0, abs=1e-5)
    assert sum(1 for g in d["generators"] if g["slack"]) == 1
    assert all(isl["generators"] for isl in d["islands"])  # у каждого острова есть источник
    for k in ("lengths_km", "conductors_by_km", "loads_by_bus", "generation_by_mw"):
        assert k in d["summary"]


def test_reproducible(real):
    d, _ = real
    try:
        built, _ = br.build()
    except FileNotFoundError:
        pytest.skip("нет снимка OSM")
    built["meta"]["built"] = d["meta"]["built"]
    assert built == d


def test_core_runs_on_real(real):
    r = run_power_flow({"T": 25, "load": 60}, load_network(REAL))
    assert len(r["live"]) == len(r["V"]) and r["unserved"] == 0


def test_bus_loads_from_estimates():
    est = {"hours": 8760, "consumers": [
        {"id": "a", "name": "A", "island": "X", "energy_mln_kwh": 87.6, "status": "источник", "source": "http://s",
         "location": "стр. 1", "allocation": [{"osm": "way/1", "frac": 0.5, "basis": "b"}, {"osm": "way/2", "frac": 0.5, "basis": "b"}]},
        {"id": "b", "name": "B", "island": "X", "avg_mw": 4, "status": "допущение", "basis": "нет данных",
         "allocation": [{"osm": "way/2", "frac": 1.0, "basis": "b"}]}]}
    out = br.bus_loads(est)
    assert out[("way/1", 110)]["mw"] == pytest.approx(5) and out[("way/1", 110)]["sourced"]
    assert out[("way/2", 110)]["mw"] == pytest.approx(9) and not out[("way/2", 110)]["sourced"]
    est["consumers"][1]["allocation"][0]["frac"] = 0.7
    with pytest.raises(ValueError):
        br.bus_loads(est)


def test_load_scenarios_and_marks(real):
    d, _ = real
    ls = d["meta"]["load_scenarios"]
    assert ls["status"] == "допущение" and ls["min"] < ls["base"] < ls["peak"]
    assert ls["base"] == pytest.approx(sum(b.get("load_mw_avg", 0) for b in d["buses"]), abs=1e-2)
    for b in d["buses"]:
        if b["share"] > 0:
            m = b["marks"]["load_energy"]
            assert m["status"] in ("источник", "допущение")
            if m["status"] == "источник":
                assert m["source"].startswith("http")
