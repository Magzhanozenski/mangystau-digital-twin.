"""Данные эксперимента B: генератор нагрузки прототипа на реальной погоде 2023–2025 (решение пользователя).

«Правда» L = true_load(h, T, Tlag, выходной) + шум AR(1) (ε = 0,85·ε + N(0; 1,6), как в makeHistory прототипа).
T — среднее ERA5 по трём точкам (Актау, Жанаозен, Бейнеу), Tlag — среднее за 3 часа, Tmax — максимум суток,
выходной — суббота/воскресенье по календарю. Для проверки на 2025 г. те же признаки считаются и по прогнозу на сутки
(Open-Meteo, выпуск накануне): модель получает прогноз, «правда» остаётся по факту.
Это СИНТЕТИКА на реальной погоде: выводы касаются методов, а не нагрузки Мангистау.
Сохраняет experiments/exp_b_data.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import exp_b_models as B  # noqa: E402

OUT = Path(__file__).resolve().parent / "exp_b_data.csv"
POINTS = ("aktau", "zhanaozen", "beyneu")
NOISE_SEED = 21


def region_T(source: str, years=(2023, 2024, 2025)) -> pd.Series:
    cols = []
    for p in POINTS:
        d = pd.concat([pd.read_csv(ROOT / "data" / "weather" / source / f"{p}_{y}.csv", parse_dates=["time"])
                       for y in years]).set_index("time")["T_c"].rename(p)
        cols.append(d)
    return pd.concat(cols, axis=1).mean(axis=1, skipna=False)


def derived(T: pd.Series) -> pd.DataFrame:
    Tlag = (T + T.shift(1) + T.shift(2)) / 3
    Tmax = T.groupby(T.index.normalize()).transform("max")
    return pd.DataFrame({"T": T, "Tlag": Tlag, "Tmax": Tmax})


def build() -> pd.DataFrame:
    act = derived(region_T("archive"))
    fc = derived(region_T("day_ahead")).add_suffix("_fc")
    d = act.join(fc, how="left")
    d["h"] = d.index.hour
    d["weekend"] = d.index.dayofweek >= 5
    d["year"] = d.index.year
    rng = np.random.default_rng(NOISE_SEED)
    eps, e = np.zeros(len(d)), 0.0
    for i in range(len(d)):
        e = 0.85 * e + rng.normal(0, 1.6)
        eps[i] = e
    d["load"] = B.true_load(d["h"].to_numpy(), d["T"].to_numpy(), d["Tlag"].bfill().to_numpy(), d["weekend"].to_numpy()) + eps
    d = d.iloc[2:]  # первые 2 часа — нет Tlag
    d.index.name = "time"
    return d


if __name__ == "__main__":
    d = build()
    d.to_csv(OUT, float_format="%.4f")
    for y in (2023, 2024, 2025):
        x = d[d.year == y]
        print(f"{y}: часов {len(x)}, T {x['T'].min():.1f}…{x['T'].max():.1f} °C, часов T ≥ 35: {(x['T'] >= 35).sum()}, "
              f"нагрузка {x['load'].min():.0f}…{x['load'].max():.0f} МВт; прогноз T есть: {x['T_fc'].notna().sum()}")
    tr = d[d.year <= 2024]
    print("T ≥ 35 в 2025 выше максимума обучения:", int((d[d.year == 2025]['T'] > tr['T'].max()).sum()), "ч; макс. обучения",
          round(tr['T'].max(), 1))
