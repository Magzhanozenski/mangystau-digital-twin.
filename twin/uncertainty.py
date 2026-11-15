"""Калиброванные интервалы P10–P50–P90 для прогноза погоды на сутки вперёд (шаг 3.2).

Прогноз — Open-Meteo Previous Runs (выпуск накануне, data/weather/day_ahead), «истина» — реанализ ERA5
(data/weather/archive), точки — Актау, Жанаозен, Бейнеу. Почасовой истории нагрузки нет, поэтому калибруется
только погода: интервалы станут входом для DLR (3.3) и модели нагрузки.

Метод: ошибка e = факт − прогноз; для каждой группы (сезон × час суток) берутся эмпирические квантили e на периоде
калибровки (по умолчанию 2023–2024); интервал = прогноз + квантиль (без предположения о нормальности).
Группа меньше MIN_GROUP часов заменяется группой «тот же час, все сезоны». Ветер и радиация не опускаются ниже 0.
Проверка покрытия — на отложенном 2025 г. (validation/check_uncertainty.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
WEATHER = ROOT / "data" / "weather"
OUT = WEATHER / "uncertainty_quantiles.csv"

POINTS = ("aktau", "zhanaozen", "beyneu")
VARS = {"T_c": None, "wind_ms": 0.0, "solar_wm2": 0.0}   # переменная → нижняя граница (None — нет)
QUANTILES = (0.1, 0.5, 0.9)
MIN_GROUP = 60          # ДОПУЩЕНИЕ: минимум часов в группе (сезон × час) для своих квантилей
CALIB_YEARS = (2023, 2024)
TEST_YEARS = (2025,)
SEASON = {12: "зима", 1: "зима", 2: "зима", 3: "весна", 4: "весна", 5: "весна",
          6: "лето", 7: "лето", 8: "лето", 9: "осень", 10: "осень", 11: "осень"}


def paired(point: str, years: tuple = CALIB_YEARS + TEST_YEARS) -> pd.DataFrame:
    """Почасовые пары прогноз/факт: time, var, fc, obs, err, season, hour, year (только полные пары)."""
    rows = []
    for y in years:
        fp, ap = WEATHER / "day_ahead" / f"{point}_{y}.csv", WEATHER / "archive" / f"{point}_{y}.csv"
        if not (fp.exists() and ap.exists()):
            continue
        fc = pd.read_csv(fp, parse_dates=["time"])
        ob = pd.read_csv(ap, parse_dates=["time"])
        j = fc.merge(ob, on="time", suffixes=("_fc", "_obs"))
        for v in VARS:
            d = pd.DataFrame({"time": j["time"], "var": v, "fc": j[f"{v}_fc"], "obs": j[f"{v}_obs"]}).dropna()
            rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df["err"] = df["obs"] - df["fc"]
    df["season"] = df["time"].dt.month.map(SEASON)
    df["hour"] = df["time"].dt.hour
    df["year"] = df["time"].dt.year
    return df


def fit(point: str, years: tuple = CALIB_YEARS, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Таблица квантилей ошибки: point, var, season, hour, n, source, q10, q50, q90."""
    d = paired(point, years) if df is None else df[df["year"].isin(years)]
    out = []
    for v in VARS:
        dv = d[d["var"] == v]
        by_hour = dv.groupby("hour")["err"]
        for season in ("зима", "весна", "лето", "осень"):
            for h in range(24):
                g = dv[(dv["season"] == season) & (dv["hour"] == h)]["err"]
                src = "сезон×час"
                if len(g) < MIN_GROUP:
                    g, src = (by_hour.get_group(h) if h in by_hour.groups else pd.Series(dtype=float)), "час"
                if len(g) == 0:
                    continue
                qs = np.quantile(g, QUANTILES)
                out.append({"point": point, "var": v, "season": season, "hour": h, "n": len(g), "source": src,
                            **{f"q{int(q * 100)}": float(x) for q, x in zip(QUANTILES, qs)}})
    return pd.DataFrame(out)


def naive(point: str, years: tuple = CALIB_YEARS, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Простой вариант для сравнения: прогноз ± 1,2816·σ ошибки за весь период, без учёта сезона, часа и смещения."""
    d = paired(point, years) if df is None else df[df["year"].isin(years)]
    return pd.DataFrame([{"point": point, "var": v, "sigma": float(d[d["var"] == v]["err"].std())} for v in VARS])


def apply(fc: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Интервалы для прогноза: fc — time, var, fc (одна точка); table — квантили этой точки.
    Возвращает fc с колонками p10, p50, p90 (ветер и радиация — не ниже 0; p10 ≤ p50 ≤ p90)."""
    x = fc.copy()
    x["season"] = x["time"].dt.month.map(SEASON)
    x["hour"] = x["time"].dt.hour
    x = x.merge(table[["var", "season", "hour", "q10", "q50", "q90"]], on=["var", "season", "hour"], how="left")
    for q in ("10", "50", "90"):
        x[f"p{q}"] = x["fc"] + x[f"q{q}"]
    for v, lo in VARS.items():
        if lo is not None:
            m = x["var"] == v
            for q in ("p10", "p50", "p90"):
                x.loc[m, q] = x.loc[m, q].clip(lower=lo)
    x[["p10", "p50", "p90"]] = np.sort(x[["p10", "p50", "p90"]].to_numpy(), axis=1)
    return x.drop(columns=["q10", "q50", "q90"])


def build(path: Path = OUT) -> pd.DataFrame:
    """Квантили по всем точкам на периоде калибровки → data/weather/uncertainty_quantiles.csv."""
    t = pd.concat([fit(p) for p in POINTS], ignore_index=True)
    t.to_csv(path, index=False, float_format="%.4f")
    return t


if __name__ == "__main__":
    t = build()
    print(f"{OUT.name}: строк {len(t)}; групп с заменой на «час»: {(t.source == 'час').sum()}")
    print(t.groupby(["point", "var"])[["q10", "q50", "q90"]].mean().round(2))
