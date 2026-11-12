"""Сравнение условной и реальной сети (шаг 2.2, пункт 7). Без сети."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import compare_network as cn  # noqa: E402


def _e(p, q, km, kv="110.0"):
    return {"p": p, "q": q, "km": km, "kv_km": {kv: km}}


def test_shortest_picks_min_and_counts_unknown():
    A, B, X = ("s", "A"), ("s", "B"), ("n", 1)
    edges = [_e(A, B, 50), _e(A, X, 10, ""), _e(X, B, 15)]
    km, kv, n = cn.shortest(edges, {"A"}, {"B"})
    assert km == pytest.approx(25) and kv == {"": 10, "110.0": 15} and n == 2


def test_shortest_none_when_disconnected():
    assert cn.shortest([_e(("s", "A"), ("s", "C"), 5)], {"A"}, {"B"}) is None


def test_candidates_by_place_and_radius():
    subs = [{"osm": "a", "nearest_place": "Бейнеу", "nearest_place_km": "3"},
            {"osm": "b", "nearest_place": "Бейнеу", "nearest_place_km": "9"},
            {"osm": "c", "nearest_place": "Шетпе", "nearest_place_km": "1"}]
    assert cn.candidates(8, subs) == ["a"]  # Бейнеу: радиус 5 км
    assert cn.candidates(3, subs) == []     # Промзона: место не определено


def test_conditional_network_has_no_sources():
    cond = yaml.safe_load((ROOT / "data" / "network.yaml").read_text(encoding="utf-8"))
    sh = cn.field_shares(cond)
    assert sh["цепи (по км)"] is None
    assert all(v == 0 for k, v in sh.items() if v is not None)


def test_every_conditional_bus_has_place_rule():
    cond = yaml.safe_load((ROOT / "data" / "network.yaml").read_text(encoding="utf-8"))
    assert {b["id"] for b in cond["buses"]} == set(cn.PLACES)
    assert all(rule[3] for rule in cn.PLACES.values())  # основание записано
