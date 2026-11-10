"""Почасовая погода Open-Meteo для точек из data/weather_points.yaml (шаг 2.1).

Три набора данных:
  archive   — реанализ ERA5 (archive-api.open-meteo.com), модель задана явно: models=era5;
  day_ahead — прогноз, выпущенный за сутки до часа (previous-runs-api, переменные *_previous_day1),
              нужен, чтобы оценивать ошибку прогноза на сутки вперёд;
  forecast  — текущий прогноз (api.open-meteo.com) на ближайшие дни.

Время — местное (Asia/Aqtau, UTC+5). Температура, ветер, влажность, облачность — мгновенные значения
на начало часа; радиация (solar_wm2) — среднее за ПРЕДЫДУЩИЙ час (соглашение Open-Meteo), т. е. строка
13:00 содержит среднее за 12:00–13:00. Это важно при сопоставлении с нагрузкой и в DLR.
Пропуски не заполняются: остаются NaN и считаются функцией gaps().
Если сети нет, archive/day_ahead читаются из CSV в data/weather/, а прогноз — из последнего сохранённого
файла (с указанием, когда он получен). Подмены синтетической погодой нет: без данных — ошибка.

Данные Open-Meteo — лицензия CC BY 4.0, https://open-meteo.com/en/licence
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
POINTS_PATH = ROOT / "data" / "weather_points.yaml"
CACHE_DIR = ROOT / "data" / "weather"

URLS = {
    "archive": "https://archive-api.open-meteo.com/v1/archive",
    "day_ahead": "https://previous-runs-api.open-meteo.com/v1/forecast",
    "forecast": "https://api.open-meteo.com/v1/forecast",
}

# Переменная Open-Meteo → колонка в CSV
VARIABLES = {
    "temperature_2m": "T_c",
    "relative_humidity_2m": "rh_pct",
    "wind_speed_10m": "wind_ms",
    "wind_direction_10m": "wind_dir_deg",
    "shortwave_radiation": "solar_wm2",
    "cloud_cover": "cloud_pct",
}
COLUMNS = list(VARIABLES.values())


class WeatherUnavailable(RuntimeError):
    """Нет ни ответа Open-Meteo, ни сохранённых данных."""


# ---------- точки ----------

def load_points(path: str | Path = POINTS_PATH) -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for section in ("points", "bus_point"):
        for item in raw[section]:
            key = item.get("id", item.get("bus"))
            if item.get("status") not in ("источник", "допущение"):
                raise ValueError(f"{section}: у элемента {key} нет пометки status (источник/допущение)")
            if item["status"] == "источник" and not item.get("source"):
                raise ValueError(f"{section}: у элемента {key} status=источник, но нет поля source")
            if item["status"] == "допущение" and not item.get("basis"):
                raise ValueError(f"{section}: у элемента {key} status=допущение, но нет поля basis")
    ids = {p["id"] for p in raw["points"]}
    for bp in raw["bus_point"]:
        if bp["point"] not in ids:
            raise ValueError(f"bus_point: узел {bp['bus']} ссылается на неизвестную точку {bp['point']}")
    return raw


def point(point_id: str, points: dict | None = None) -> dict:
    points = points or load_points()
    for p in points["points"]:
        if p["id"] == point_id:
            return p
    raise KeyError(point_id)


# ---------- запрос и разбор ----------

def _request(url: str, params: dict, retries: int = 3, timeout: float = 60) -> dict:
    full = url + "?" + urllib.parse.urlencode(params)
    last = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=timeout) as r:
                data = json.load(r)
            if data.get("error"):
                raise WeatherUnavailable(f"Open-Meteo: {data.get('reason')}")
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(2 * (k + 1))
    raise WeatherUnavailable(f"Open-Meteo недоступен ({url}): {last!r}")


def parse_response(data: dict, suffix: str = "", model: str = "") -> pd.DataFrame:
    """JSON Open-Meteo → таблица с колонками time, T_c, ..., grid_lat, grid_lon, grid_elevation, model."""
    h = data["hourly"]
    units = data.get("hourly_units", {})
    expect = {"temperature_2m": "°C", "relative_humidity_2m": "%", "wind_speed_10m": "m/s",
              "wind_direction_10m": "°", "shortwave_radiation": "W/m²", "cloud_cover": "%"}
    df = pd.DataFrame({"time": pd.to_datetime(h["time"])})
    for var, col in VARIABLES.items():
        key = var + suffix
        if key not in h:
            raise ValueError(f"в ответе нет переменной {key}")
        if units.get(key) != expect[var]:
            raise ValueError(f"{key}: единицы {units.get(key)!r}, ожидались {expect[var]!r}")
        df[col] = pd.to_numeric(pd.Series(h[key], dtype="object"), errors="coerce").astype(float)
    if data.get("utc_offset_seconds") != 5 * 3600:
        raise ValueError(f"часовой пояс ответа UTC{data.get('utc_offset_seconds', 0) / 3600:+g}, ожидался UTC+5")
    df["grid_lat"] = data["latitude"]
    df["grid_lon"] = data["longitude"]
    df["grid_elevation"] = data.get("elevation")
    df["model"] = model
    return df


def _params(p: dict, tz: str, hourly: list[str], **extra) -> dict:
    return {"latitude": p["lat"], "longitude": p["lon"], "hourly": ",".join(hourly),
            "wind_speed_unit": "ms", "timezone": tz, **extra}


def fetch(dataset: str, point_id: str, start: date | None = None, end: date | None = None,
          forecast_days: int = 3) -> pd.DataFrame:
    """Запрос к Open-Meteo без кэша."""
    pts = load_points()
    p, tz = point(point_id, pts), pts["timezone"]
    if dataset == "archive":
        prm = _params(p, tz, list(VARIABLES), start_date=start, end_date=end, models="era5")
        return parse_response(_request(URLS[dataset], prm), model="era5")
    if dataset == "day_ahead":
        prm = _params(p, tz, [v + "_previous_day1" for v in VARIABLES], start_date=start, end_date=end)
        return parse_response(_request(URLS[dataset], prm), suffix="_previous_day1", model="best_match")
    if dataset == "forecast":
        prm = _params(p, tz, list(VARIABLES), forecast_days=forecast_days)
        df = parse_response(_request(URLS[dataset], prm), model="best_match")
        df["issued"] = datetime.now().strftime("%Y-%m-%dT%H:%M")
        return df
    raise ValueError(dataset)


# ---------- кэш ----------

def _year_path(dataset: str, point_id: str, year: int) -> Path:
    return CACHE_DIR / dataset / f"{point_id}_{year}.csv"


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["time"])


def history(dataset: str, point_id: str, start: date, end: date, refresh: bool = False) -> pd.DataFrame:
    """Архив или прогнозы на сутки за период; хранится в data/weather/<dataset>/<точка>_<год>.csv.

    Недостающие годы загружаются из Open-Meteo. Текущий год не кэшируется (он ещё не закончился).
    """
    if dataset not in ("archive", "day_ahead"):
        raise ValueError(dataset)
    parts = []
    for y in range(start.year, end.year + 1):
        path = _year_path(dataset, point_id, y)
        if path.exists() and not refresh:
            parts.append(_read_csv(path))
            continue
        ys, ye = date(y, 1, 1), date(y, 12, 31)
        complete = ye < date.today()
        df = fetch(dataset, point_id, ys, ye if complete else date.today())
        if complete:
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False, float_format="%.6g")
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df[(df.time >= pd.Timestamp(start)) & (df.time < pd.Timestamp(end) + pd.Timedelta(days=1))].reset_index(drop=True)


def forecast(point_id: str, days: int = 3) -> pd.DataFrame:
    """Текущий прогноз. Сохраняется в data/weather/forecast/ (не в git).

    Без сети возвращается последний сохранённый прогноз; когда он получен — в колонке issued.
    """
    folder = CACHE_DIR / "forecast"
    try:
        df = fetch("forecast", point_id, forecast_days=days)
    except WeatherUnavailable:
        saved = sorted(folder.glob(f"{point_id}_*.csv"))
        if not saved:
            raise
        return _read_csv(saved[-1])
    folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(folder / f"{point_id}_{df['issued'].iloc[0].replace(':', '')}.csv", index=False, float_format="%.6g")
    return df


def tomorrow(point_id: str) -> pd.DataFrame:
    """24 часа завтрашнего дня из текущего прогноза."""
    df = forecast(point_id)
    d = pd.Timestamp(date.today()) + pd.Timedelta(days=1)
    return df[(df.time >= d) & (df.time < d + pd.Timedelta(days=1))].reset_index(drop=True)


# ---------- проверки ----------

LIMITS = {  # физически допустимые диапазоны для региона — грубый фильтр ошибок, а не климат
    "T_c": (-45, 55), "rh_pct": (0, 100), "wind_ms": (0, 45),
    "wind_dir_deg": (0, 360), "solar_wm2": (0, 1250), "cloud_pct": (0, 100),
}


def gaps(df: pd.DataFrame) -> dict:
    """Пропуски: число NaN по колонкам и пропущенные часы в ряду времени."""
    t = df["time"]
    full = pd.date_range(t.min(), t.max(), freq="h")
    return {"nan": {c: int(df[c].isna().sum()) for c in COLUMNS},
            "missing_hours": int(len(full.difference(t))),
            "duplicate_hours": int(t.duplicated().sum())}


def out_of_range(df: pd.DataFrame) -> dict:
    return {c: int(((df[c] < lo) | (df[c] > hi)).sum()) for c, (lo, hi) in LIMITS.items()}
