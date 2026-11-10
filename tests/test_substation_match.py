"""Опознание ПС OSM (шаг 2.2, пункт 4). Без сети: искусственные примеры и сохранённые таблицы."""
import pytest

from twin import substation_match as sm

SRC = {k: f"https://example.org/{k}" for k in
       ("lada_kegoc_2026", "mrek_tru_2021", "kus_2019", "mrek_2021", "eco_gpes_uzen")}


def _s(osm, voltage="", place="", km="", name="", power="substation"):
    return {"osm": osm, "power": power, "name": name, "voltage_tag": voltage, "lat": "0", "lon": "0",
            "nearest_place": place, "nearest_place_km": km}


def _by(rows):
    return {r["osm"]: r for r in rows}


def test_voltage_classes():
    assert sm.voltage_classes("220000;110000") == {220, 110}
    assert sm.voltage_classes("110000;fixme") == {110}
    assert sm.voltage_classes("") == set()


def test_levels():
    rows = _by(sm.identify([
        _s("a", name="ТЭЦ-3", power="plant", place="Умирзак", km="3.8"),
        _s("b", "220000;110000", "Бейнеу", "3.47"),
        _s("c", "", "Актау", "1.2"),
        _s("d", "", "Толеп", "40.0"),
    ], SRC))
    assert (rows["a"]["confidence"], rows["a"]["status"]) == ("высокая", "источник")
    assert rows["b"]["confidence"] == "средняя" and rows["b"]["match_name"] == "ПС 220 кВ «Бейнеу»"
    assert "example.org/lada_kegoc_2026" in rows["b"]["source"]
    assert (rows["c"]["confidence"], rows["c"]["status"]) == ("низкая", "допущение")
    assert "1.2 км" in rows["c"]["basis"] and "не установлено" in rows["c"]["match_name"]
    assert rows["d"]["confidence"] == rows["d"]["status"] == "не опознана"
    assert rows["d"]["match_name"] == "" and rows["d"]["basis"]  # основание отказа записано


def test_ambiguous_is_not_identified_by_document():
    """Два кандидата 220 кВ у одного посёлка — документ не привязывается ни к одному."""
    rows = _by(sm.identify([_s("x", "220000", "Бейнеу", "2.0"), _s("y", "220000", "Бейнеу", "4.0")], SRC))
    for r in rows.values():
        assert r["confidence"] == "низкая" and "выбрать нельзя" in r["basis"]


def test_110_not_confused_with_220_110():
    rows = _by(sm.identify([_s("p", "220000;110000;35000", "Бейнеу", "3.0"),
                            _s("q", "110000;35000", "Бейнеу", "1.8")], SRC))
    assert rows["p"]["match_name"] == "ПС 220 кВ «Бейнеу»"
    assert rows["q"]["match_name"] == "ПС 110/35/6 кВ «Бейнеу»"


def test_doc_too_far_from_place():
    rows = _by(sm.identify([_s("z", "220000", "Каражанбас", "16.0")], SRC))
    assert rows["z"]["confidence"] == "не опознана"


def test_real_tables():
    try:
        subs = sm.read_substations()
    except FileNotFoundError:
        pytest.skip("нет data/network_sources/osm_substations.csv")
    rows = sm.identify(subs, sm.load_doc_sources())
    assert len(rows) == len(subs)
    for r in rows:
        assert r["confidence"] in ("высокая", "средняя", "низкая", "не опознана")
        assert r["basis"], r["osm"]
        if r["status"] == "источник":
            assert r["source"].startswith("http")
    names = {r["match_name"] for r in rows if r["confidence"] == "средняя"}
    assert "УРПС 220/110/35/10 кВ «Каражанбас»" in names
