"""Обработка снимка OpenStreetMap (шаг 2.2). Без сети: только сохранённый снимок и искусственные примеры."""
import pandas as pd
import pytest

from twin import osm


@pytest.fixture(scope="module")
def snap():
    try:
        return osm.load_snapshot()
    except FileNotFoundError:
        pytest.skip("нет снимка data/osm/power_*.json")


def _way(wid, nodes, coords, voltage=None):
    tags = {"power": "line"}
    if voltage:
        tags["voltage"] = voltage
    return {"type": "way", "id": wid, "nodes": nodes, "tags": tags,
            "geometry": [{"lat": a, "lon": b} for a, b in coords]}


def _sub(wid, lat, lon, d=0.002):
    poly = [(lat - d, lon - d), (lat - d, lon + d), (lat + d, lon + d), (lat + d, lon - d), (lat - d, lon - d)]
    return {"type": "way", "id": wid, "tags": {"power": "substation"}, "geometry": [{"lat": a, "lon": b} for a, b in poly]}


def test_haversine():
    assert osm.haversine_km(43.0, 51.0, 44.0, 51.0) == pytest.approx(111.2, abs=0.2)


def test_distance_to_polygon():
    sq = [(43.0, 51.0), (43.0, 51.01), (43.01, 51.01), (43.01, 51.0), (43.0, 51.0)]
    assert osm._dist_to_polygon_km(43.005, 51.005, sq) == 0.0
    assert osm._dist_to_polygon_km(43.005, 51.02, sq) == pytest.approx(0.81, abs=0.02)  # ~0,01° долготы


def test_voltage_inferred_only_for_plain_continuation():
    # A(110) — B(без тега) — C(без тега): цепочка продолжений → 110 выводится для обоих
    data = {"elements": [
        _way(1, [1, 2], [(43.0, 51.0), (43.0, 51.1)], "110000"),
        _way(2, [2, 3], [(43.0, 51.1), (43.0, 51.2)]),
        _way(3, [3, 4], [(43.0, 51.2), (43.0, 51.3)]),
    ]}
    s = {x["way"]: x for x in osm.segments(data)}
    assert s[2]["kv"] == 110 and s[3]["kv"] == 110
    assert s[2]["kv_basis"].startswith("выведено")


def test_voltage_not_inferred_at_branch_or_substation():
    # в узле 2 сходятся три участка — ответвление, вывод запрещён
    branch = {"elements": [
        _way(1, [1, 2], [(43.0, 51.0), (43.0, 51.1)], "110000"),
        _way(2, [2, 3], [(43.0, 51.1), (43.0, 51.2)]),
        _way(3, [2, 4], [(43.0, 51.1), (43.1, 51.1)], "35000"),
    ]}
    assert {x["way"]: x for x in osm.segments(branch)}[2]["kv"] is None
    # стык на ПС — вывод запрещён (на ПС меняется класс напряжения)
    on_sub = {"elements": [
        _way(1, [1, 2], [(43.0, 51.0), (43.0, 51.1)], "220000"),
        _way(2, [2, 3], [(43.0, 51.1), (43.0, 51.2)]),
        _sub(9, 43.0, 51.1),
    ]}
    seg = {x["way"]: x for x in osm.segments(on_sub)}
    assert seg[2]["kv"] is None and seg[2]["sub1"] == "way/9"
    # противоречивые соседи — вывод запрещён
    conflict = {"elements": [
        _way(1, [1, 2], [(43.0, 51.0), (43.0, 51.1)], "110000"),
        _way(2, [2, 3], [(43.0, 51.1), (43.0, 51.2)]),
        _way(3, [3, 4], [(43.0, 51.2), (43.0, 51.3)], "35000"),
    ]}
    assert {x["way"]: x for x in osm.segments(conflict)}[2]["kv"] is None


def test_routes_merge_through_continuations_not_substations():
    data = {"elements": [
        _way(1, [1, 2], [(43.0, 51.0), (43.0, 51.1)], "110000"),
        _way(2, [2, 3], [(43.0, 51.1), (43.0, 51.2)], "110000"),
        _way(3, [3, 4], [(43.0, 51.2), (43.0, 51.3)], "110000"),
        _sub(9, 43.0, 51.2),  # узел 3 — на ПС: линии 1+2 и 3 — разные
    ]}
    rts = osm.routes(osm.segments(data))
    assert sorted(r["n_ways"] for r in rts) == [1, 2]
    two = next(r for r in rts if r["n_ways"] == 2)
    assert sorted(two["ends"].split(";")) == ["way/9", "обрыв"]


def test_snapshot_metadata(snap):
    assert snap["osm3s"]["timestamp_osm_base"]
    assert "area(3600215686)" in snap["query"]  # граница Мангистауской области


def test_snapshot_processing(snap):
    subs = osm.substations(snap)
    segs = osm.segments(snap, subs)
    rts = osm.routes(segs)
    assert len(segs) == sum(1 for e in snap["elements"] if e["tags"].get("power") in ("line", "cable"))
    assert all(s["km"] > 0 for s in segs)
    # класс напряжения — только из тега или выведен по правилу; иначе «неизвестно»
    for s in segs:
        assert (s["kv"] is None) == (s["kv_basis"] in ("неизвестно",) or "неоднозначен" in s["kv_basis"])
    # каждый участок — ровно в одной линии; длины сохраняются
    ways = [w for r in rts for w in r["ways"].split(";")]
    assert sorted(ways) == sorted(str(s["way"]) for s in segs)
    for kv in {s["kv"] for s in segs}:
        assert sum(r["km"] for r in rts if r["kv"] == kv) == pytest.approx(sum(s["km"] for s in segs if s["kv"] == kv), abs=0.1)


def test_saved_tables_reproduce(snap, tmp_path):
    """CSV в data/network_sources получаются из снимка заново — без расхождений."""
    if not (osm.SOURCES_DIR / "osm_lines.csv").exists():
        pytest.skip("нет data/network_sources/osm_lines.csv")
    osm.export_tables(out_dir=tmp_path)
    for name in ("osm_segments", "osm_lines", "osm_substations"):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / f"{name}.csv"), pd.read_csv(osm.SOURCES_DIR / f"{name}.csv"))
