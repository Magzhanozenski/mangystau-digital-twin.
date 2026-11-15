"""Почасовая нагрузка сети МРЭК на 2025 г. для эксперимента A — ДОПУЩЕНИЕ (почасовых данных нет).

Модель (решение пользователя: «с зависимостью от температуры»):
  L(t) = M(d) · S(h),   S — суточный профиль шага 3.1 (load_estimates.yaml, среднее 1);
  M(d) = a + b·max(0, Tmax_d − 30) + 0,42·b·max(0, 10 − Tmean_d),
  T — среднее ERA5 по трём точкам (Актау, Жанаозен, Бейнеу), °C. Пороги 30 / 10 °C и отношение 0,42 (холод к жаре,
  как 1,3/3,1 в trueLoad прототипа) — допущения.
  a и b подбираются так, чтобы средняя нагрузка года = 392,1 МВт и максимум часа = 546 МВт (МРЭК 2023, источник;
  применение к 2025 г. — допущение).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
MEAN_MW, PEAK_MW = 392.1, 546.0
T_HOT, T_COLD, COLD_RATIO = 30.0, 10.0, 0.42


def region_temperature(year: int = 2025) -> pd.DataFrame:
    parts = [pd.read_csv(ROOT / "data" / "weather" / "archive" / f"{p}_{year}.csv", parse_dates=["time"])[["time", "T_c"]]
             .set_index("time")["T_c"].rename(p) for p in ("aktau", "zhanaozen", "beyneu")]
    t = pd.concat(parts, axis=1)
    return pd.DataFrame({"time": t.index, "t_c": t.mean(axis=1).to_numpy()})


def hourly_load(year: int = 2025) -> pd.DataFrame:
    w = region_temperature(year)
    shape = np.array(yaml.safe_load((ROOT / "data" / "network_sources" / "load_estimates.yaml")
                                    .read_text(encoding="utf-8"))["daily_profile"]["shape"])
    day = w["time"].dt.normalize()
    tmax = w.groupby(day)["t_c"].transform("max").to_numpy()
    tmean = w.groupby(day)["t_c"].transform("mean").to_numpy()
    x = np.maximum(0, tmax - T_HOT) + COLD_RATIO * np.maximum(0, T_COLD - tmean)
    s = shape[w["time"].dt.hour.to_numpy()]

    def load(b):
        a = (MEAN_MW - b * np.mean(x * s)) / np.mean(s)
        return (a + b * x) * s, a

    b = brentq(lambda b: load(b)[0].max() - PEAK_MW, 0.0, 200.0)
    L, a = load(b)
    out = w.assign(load_mw=L)
    out.attrs.update({"a": a, "b": b})
    return out


if __name__ == "__main__":
    d = hourly_load()
    print(f"a = {d.attrs['a']:.1f} МВт, b = {d.attrs['b']:.2f} МВт/°C; среднее {d.load_mw.mean():.1f}, "
          f"мин {d.load_mw.min():.1f}, макс {d.load_mw.max():.1f} МВт; час максимума {d.time[d.load_mw.idxmax()]}, "
          f"T {d.t_c[d.load_mw.idxmax()]:.1f} °C")
    m = d.groupby(d.time.dt.month).load_mw.mean().round(0)
    print("по месяцам:", dict(m))
