"""Погода Open-Meteo (шаг 2.1). Все тесты работают без интернета: сеть в них отключена."""
import copy
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import yaml

from twin import weather as w

SAMPLE = Path(__file__).parent / "data" / "openmeteo_archive_aktau_2024-07-15.json"
POINTS = ("aktau", "zhanaozen", "beyneu")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*a, **k):
        raise w.WeatherUnavailable("сеть в тестах отключена")
    monkeypatch.setattr(w, "_request", fail)


@pytest.fixture
def sample():
    return json.loads(SAMPLE.read_text(encoding="utf-8"))


# ---------- точки ----------

def test_points_have_status(net):
    pts = w.load_points()
    assert {p["id"] for p in pts["points"]} == set(POINTS)
    for p in pts["points"]:
        assert p["status"] == "источник" and p["source"].startswith("https://www.geonames.org/")
    # каждый узел сети привязан ровно к одной точке
    assert sorted(bp["bus"] for bp in pts["bus_point"]) == sorted(b["id"] for b in net.buses)


def test_points_loader_rejects_missing_status(tmp_path):
    raw = yaml.safe_load(Path(w.POINTS_PATH).read_text(encoding="utf-8"))
    f = tmp_path / "p.yaml"
    del raw["points"][0]["status"]
    f.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="aktau"):
        w.load_points(f)
    raw["points"][0]["status"] = "источник"
    del raw["points"][0]["source"]
    f.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ValueError, match="нет поля source"):
        w.load_points(f)


# ---------- разбор ответа ----------

def test_parse_sample(sample):
    df = w.parse_response(sample, model="era5")
    assert len(df) == 24
    assert list(df.time.dt.hour) == list(range(24))
    assert df.time.iloc[0] == pd.Timestamp("2024-07-15 00:00")
    assert w.gaps(df) == {"nan": {c: 0 for c in w.COLUMNS}, "missing_hours": 0, "duplicate_hours": 0}
    assert all(v == 0 for v in w.out_of_range(df).values())
    # местное время: ночью солнца нет, днём есть, жарче всего после полудня.
    # Радиация — среднее за предыдущий час, поэтому в 22:00 (сумерки 21–22 ч) бывает ~1 Вт/м².
    assert (df[df.time.dt.hour.isin([0, 1, 2, 3, 22, 23])].solar_wm2 < 5).all()
    assert df.solar_wm2.max() > 500
    assert 11 <= df.loc[df.T_c.idxmax(), "time"].hour <= 18
    assert (df.grid_lat.iloc[0], df.grid_lon.iloc[0]) == (43.75, 51.25)


def test_parse_rejects_wrong_units(sample):
    bad = copy.deepcopy(sample)
    bad["hourly_units"]["wind_speed_10m"] = "km/h"
    with pytest.raises(ValueError, match="единицы"):
        w.parse_response(bad)


def test_parse_rejects_wrong_timezone(sample):
    bad = copy.deepcopy(sample)
    bad["utc_offset_seconds"] = 0
    with pytest.raises(ValueError, match="часовой пояс"):
        w.parse_response(bad)


def test_missing_values_stay_nan(sample):
    bad = copy.deepcopy(sample)
    bad["hourly"]["temperature_2m"][5] = None
    df = w.parse_response(bad)
    assert w.gaps(df)["nan"]["T_c"] == 1  # пропуск не заполнен


def test_gaps_finds_missing_and_duplicate_hours(sample):
    df = w.parse_response(sample)
    df = pd.concat([df.drop(index=7), df.iloc[[3]]]).sort_values("time")
    g = w.gaps(df)
    assert g["missing_hours"] == 1 and g["duplicate_hours"] == 1


# ---------- кэш и работа без сети ----------

def test_forecast_without_network_and_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "CACHE_DIR", tmp_path)
    with pytest.raises(w.WeatherUnavailable):
        w.forecast("aktau")


def test_history_without_network_and_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "CACHE_DIR", tmp_path)
    with pytest.raises(w.WeatherUnavailable):
        w.history("archive", "aktau", date(2024, 1, 1), date(2024, 1, 31))


@pytest.mark.parametrize("pid", POINTS)
def test_saved_archive_is_complete(pid):
    """Сохранённый архив ERA5 2023–2025: каждый час на месте, значения физичны (читается из CSV, без сети)."""
    if not (w.CACHE_DIR / "archive" / f"{pid}_2025.csv").exists():
        pytest.skip("нет сохранённого архива data/weather/archive")
    df = w.history("archive", pid, date(2023, 1, 1), date(2025, 12, 31))
    assert len(df) == (365 + 366 + 365) * 24
    assert w.gaps(df) == {"nan": {c: 0 for c in w.COLUMNS}, "missing_hours": 0, "duplicate_hours": 0}
    assert all(v == 0 for v in w.out_of_range(df).values())
    # средний суточный ход: максимум температуры днём, ночью радиации нет
    by_hour = df.groupby(df.time.dt.hour)
    assert 11 <= by_hour.T_c.mean().idxmax() <= 18
    assert by_hour.solar_wm2.mean()[[0, 1, 2, 3, 23]].max() < 1
    # лето теплее зимы
    by_month = df.groupby(df.time.dt.month).T_c.mean()
    assert by_month[7] - by_month[1] > 20
