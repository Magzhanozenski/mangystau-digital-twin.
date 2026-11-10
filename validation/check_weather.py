"""Проверка погоды Open-Meteo (шаг 2.1).

Запуск из корня репозитория:  python validation/check_weather.py
Работает с сохранёнными CSV в data/weather/; наблюдения станций NOAA ISD скачиваются
в data/weather/stations/ (не в git), если их там нет.

Что проверяется:
  1. Целостность архива ERA5 2023–2025: все часы на месте, значения в физических пределах.  → ошибка, если нет
  2. ERA5 против наблюдений станций (Актау — WMO 38111 и аэропорт UATE, Бейнеу — WMO 35927):
     смещение, MAE, корреляция температуры и ветра; сдвиг по времени с наибольшей корреляцией.
     Ошибкой считается только признак сбоя обработки: корреляция T < 0.9 или лучший сдвиг ≠ 0 ч
     (это пороги-допущения для поиска ошибок часового пояса и единиц, а не оценка качества данных).
  3. Ошибка прогноза на сутки вперёд (previous-runs, *_previous_day1) против ERA5 и против станций.
  4. MONTH_TYPICAL из core_js/day.js (прежние допущения) против средних ERA5 по Актау.
Пункты 2–4 только печатают цифры: какими бы они ни были, они идут в docs/decisions.md.

Станции: NOAA NCEI Integrated Surface Database (ISD, global-hourly), первичные сводки SYNOP (FM-12)
и METAR (FM-15); https://www.ncei.noaa.gov/products/land-based-station/integrated-surface-database ,
файлы — https://noaa-global-hourly-pds.s3.amazonaws.com/<год>/<станция>.csv . Время — UTC, ветер — м/с ×10.
Берутся только значения с кодами качества ISD 0, 1, 4, 5, 9 (прошли контроль или не проверялись).

Почему не Meteostat: его почасовые ряды заполняют пропуски модельными данными (у Бейнеу — 24 значения
в сутки при 8 сводках SYNOP), а флаги источника в выгрузке обрываются на 2022 г. — отделить наблюдения нельзя.

Важно: ERA5 усваивает наблюдения станций (в том числе эти сводки), поэтому сверка не полностью независима.
"""
import gzip
import re
import sys
import urllib.request
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import weather as w  # noqa: E402

START, END = date(2023, 1, 1), date(2025, 12, 31)
STATIONS = {  # точка → [(станция ISD, описание)]; расстояния — от координат точки (гаверсинус)
    "aktau": [("38111099999", "Актау-город WMO 38111 (SEVCENKO), SYNOP, 11 км"),
              ("38001099999", "аэропорт Актау UATE (WMO 38001), METAR, 22 км")],
    "beyneu": [("35927099999", "Бейнеу WMO 35927, SYNOP, 12 км")],
}
NO_STATION = {"zhanaozen": "в ISD станция UZEN (Жанаозен) есть только за 1981–1986; ближайшая действующая — "
                           "Аккудук WMO 38232 в 110 км, примерно на 125 м ниже (78 м по ISD), ряд до 09.2024: не сопоставима"}
ISD_OK = set("01459")
ISD_URL = "https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{sid}.csv"


def _isd_value(field: pd.Series, pos: int, qpos: int, scale: float) -> pd.Series:
    parts = field.astype(str).str.split(",", expand=True)
    v = pd.to_numeric(parts[pos], errors="coerce")
    v = v.where(~parts[pos].str.lstrip("+-").str.fullmatch(r"9+"))  # 9999 / +9999 — нет значения
    return (v / scale).where(parts[qpos].isin(ISD_OK))


def station(sid: str) -> pd.DataFrame:
    """Наблюдения ISD за START…END, местное время, только записи на начало часа."""
    parts = []
    for y in range(START.year, END.year + 1):
        path = w.CACHE_DIR / "stations" / f"{sid}_{y}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(ISD_URL.format(year=y, sid=sid), path)
        df = pd.read_csv(path, usecols=["DATE", "REPORT_TYPE", "TMP", "WND"], dtype=str)
        t = pd.to_datetime(df["DATE"])
        keep = (t.dt.minute == 0) & df["REPORT_TYPE"].str.strip().isin(["FM-12", "FM-15"])
        out = pd.DataFrame({"time": t + pd.Timedelta(hours=5),  # UTC → UTC+5
                            "T_c": _isd_value(df["TMP"], 0, 1, 10),
                            "wind_ms": _isd_value(df["WND"], 3, 4, 10),
                            "type": df["REPORT_TYPE"].str.strip()})[keep]
        parts.append(out)
    df = pd.concat(parts)
    # если в один час есть и SYNOP, и METAR — берём SYNOP
    df = df.sort_values(["time", "type"]).drop_duplicates("time", keep="first")
    return df.drop(columns="type").reset_index(drop=True)


def stats(model: pd.Series, obs: pd.Series) -> dict:
    ok = model.notna() & obs.notna()
    m, o = model[ok], obs[ok]
    return {"n": int(ok.sum()), "bias": float((m - o).mean()), "mae": float((m - o).abs().mean()),
            "corr": float(np.corrcoef(m, o)[0, 1]) if ok.sum() > 2 else float("nan")}


def best_lag(model: pd.DataFrame, obs: pd.DataFrame, col: str = "T_c") -> int:
    s = model.set_index("time")[col]
    best = max(range(-3, 4), key=lambda k: obs.set_index("time")[col].corr(s.shift(k, freq="h")))
    return best


def fmt(s: dict, unit: str) -> str:
    return f"n={s['n']:6d}  смещение {s['bias']:+.2f} {unit}  MAE {s['mae']:.2f} {unit}  r={s['corr']:.3f}"


def month_typical() -> list:
    js = (ROOT / "core_js" / "day.js").read_text(encoding="utf-8")
    body = re.search(r"const MONTH_TYPICAL = \[(.*?)\];", js).group(1)
    return [list(map(float, x.split(","))) for x in re.findall(r"\[([^\]]+)\]", body)]


def main() -> int:
    errors = []

    print("== 1. Архив ERA5 2023–2025: целостность")
    arch = {}
    for p in ("aktau", "zhanaozen", "beyneu"):
        df = w.history("archive", p, START, END)
        g, r = w.gaps(df), w.out_of_range(df)
        bad = sum(g["nan"].values()) + g["missing_hours"] + g["duplicate_hours"] + sum(r.values())
        print(f"  {p:10s} часов {len(df)}  ячейка {df.grid_lat[0]}, {df.grid_lon[0]} ({df.grid_elevation[0]:.0f} м)  "
              f"пропусков {sum(g['nan'].values()) + g['missing_hours']}, дублей {g['duplicate_hours']}, вне пределов {sum(r.values())}")
        if bad or len(df) != 1096 * 24:
            errors.append(f"архив {p}: нарушена целостность")
        arch[p] = df

    print("\n== 2. ERA5 против наблюдений станций")
    obs_all = {}
    for p, lst in STATIONS.items():
        for k_st, (sid, desc) in enumerate(lst):
            obs = station(sid)
            obs = obs[(obs.time >= pd.Timestamp(START)) & (obs.time < pd.Timestamp(END) + pd.Timedelta(days=1))]
            if k_st == 0:
                obs_all[p] = obs  # для оценки прогноза — основная станция точки
            j = obs.merge(arch[p], on="time", suffixes=("_obs", ""))
            st, sw = stats(j["T_c"], j["T_c_obs"]), stats(j["wind_ms"], j["wind_ms_obs"])
            lag = best_lag(arch[p], obs)
            print(f"  {p} — {desc}; наблюдения {obs.time.min():%Y-%m-%d} … {obs.time.max():%Y-%m-%d}")
            print(f"     T:     {fmt(st, '°C')}")
            print(f"     ветер: {fmt(sw, 'м/с')}  (станция — 10 м на площадке, ERA5 — среднее по ячейке)")
            print(f"     лучший сдвиг ERA5 относительно станции по T: {lag:+d} ч")
            hot = j[j.T_c_obs >= 35]
            if len(hot):
                print(f"     T при ≥ 35 °C на станции: {fmt(stats(hot['T_c'], hot['T_c_obs']), '°C')}")
            if st["corr"] < 0.9 or lag != 0:
                errors.append(f"{p}: признак сбоя обработки (r={st['corr']:.3f}, сдвиг {lag:+d} ч)")
    for p, why in NO_STATION.items():
        print(f"  {p} — не проверен: {why}")

    print("\n== 3. Прогноз на сутки вперёд (previous_day1)")
    for p in ("aktau", "zhanaozen", "beyneu"):
        da = w.history("day_ahead", p, START, END)
        full = da.dropna(subset=w.COLUMNS)
        print(f"  {p}: часов со всеми величинами {len(full)} из {len(da)}, с {full.time.min():%Y-%m-%d %H:%M}")
        j = da.merge(arch[p], on="time", suffixes=("_fc", ""))
        print(f"     против ERA5   T:     {fmt(stats(j['T_c_fc'], j['T_c']), '°C')}")
        print(f"     против ERA5   ветер: {fmt(stats(j['wind_ms_fc'], j['wind_ms']), 'м/с')}")
        day = j[j.solar_wm2 > 50]
        print(f"     против ERA5   радиация (днём): {fmt(stats(day['solar_wm2_fc'], day['solar_wm2']), 'Вт/м²')}")
        if p in obs_all:
            k = da.merge(obs_all[p], on="time", suffixes=("_fc", "_obs"))
            print(f"     против станции T:     {fmt(stats(k['T_c_fc'], k['T_c_obs']), '°C')}")
            print(f"     против станции ветер: {fmt(stats(k['wind_ms_fc'], k['wind_ms_obs']), 'м/с')}")

    print("\n== 4. MONTH_TYPICAL (core_js/day.js, допущение) против ERA5 Актау 2023–2025")
    a = arch["aktau"].copy()
    a["d"] = a.time.dt.date
    daily = a.groupby("d").agg(Tmax=("T_c", "max"), Tmin=("T_c", "min"), wind=("wind_ms", "mean"))
    daily["m"] = pd.to_datetime(daily.index).month
    mon = daily.groupby("m").mean()
    print("  мес  Tmax доп/ERA5      Tmin доп/ERA5      ветер доп/ERA5")
    for i, (tx, tn, wd) in enumerate(month_typical(), start=1):
        r = mon.loc[i]
        print(f"  {i:3d}  {tx:5.0f} / {r.Tmax:5.1f} ({tx - r.Tmax:+5.1f})  {tn:5.0f} / {r.Tmin:5.1f} ({tn - r.Tmin:+5.1f})"
              f"  {wd:4.1f} / {r.wind:4.1f} ({wd - r.wind:+4.1f})")

    print()
    if errors:
        print("ОШИБКИ:")
        for e in errors:
            print("  ", e)
        return 1
    print("Целостность в порядке, признаков сбоя обработки нет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
