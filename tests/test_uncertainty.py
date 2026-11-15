"""Интервалы P10–P90 прогноза погоды (шаг 3.2). Без сети: сохранённые CSV и искусственные примеры."""
import numpy as np
import pandas as pd
import pytest

from twin import uncertainty as U


def _fake(n_days=400, bias=1.0, sd=2.0, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2023-01-01", periods=n_days * 24, freq="h")
    fc = 10 + 10 * np.sin(np.arange(len(t)) / 24 / 365 * 2 * np.pi)
    obs = fc + bias + rng.normal(0, sd, len(t))
    d = pd.DataFrame({"time": t, "var": "T_c", "fc": fc, "obs": obs})
    d["err"] = d["obs"] - d["fc"]
    d["season"] = d["time"].dt.month.map(U.SEASON)
    d["hour"] = d["time"].dt.hour
    d["year"] = d["time"].dt.year
    return d


def test_quantiles_recover_known_error_distribution():
    d = _fake()
    t = U.fit("x", years=(2023, 2024), df=d)
    t = t[t["var"] == "T_c"]
    # ошибка ~ N(1, 2): q10 ≈ 1 − 2,56, q90 ≈ 1 + 2,56
    assert t["q10"].mean() == pytest.approx(1 - 1.2816 * 2, abs=0.25)
    assert t["q90"].mean() == pytest.approx(1 + 1.2816 * 2, abs=0.25)
    assert t["q50"].mean() == pytest.approx(1, abs=0.15)


def test_apply_orders_and_clips():
    fc = pd.DataFrame({"time": pd.to_datetime(["2025-07-01 03:00", "2025-07-01 13:00"]), "var": "wind_ms", "fc": [0.5, 5.0]})
    tab = pd.DataFrame({"var": "wind_ms", "season": "лето", "hour": [3, 13], "q10": [-2.0, -1.0], "q50": [0.0, 0.5],
                        "q90": [1.0, 2.0]})
    a = U.apply(fc, tab)
    assert (a["p10"] >= 0).all()                       # ветер не ниже 0
    assert (a["p10"] <= a["p50"]).all() and (a["p50"] <= a["p90"]).all()
    assert a.loc[1, "p90"] == pytest.approx(7.0)


def test_coverage_on_synthetic_holdout():
    """Независимый год с той же ошибкой — покрытие ≈ 80 %."""
    d = _fake(n_days=1096)
    t = U.fit("x", years=(2023, 2024), df=d)
    test = d[d["year"] == 2025]
    a = U.apply(test[["time", "var", "fc"]], t)
    cov = ((test["obs"].to_numpy() >= a["p10"].to_numpy()) & (test["obs"].to_numpy() <= a["p90"].to_numpy())).mean()
    assert cov == pytest.approx(0.8, abs=0.03)


def test_saved_table_complete():
    if not U.OUT.exists():
        pytest.skip("нет data/weather/uncertainty_quantiles.csv")
    t = pd.read_csv(U.OUT)
    assert len(t) == len(U.POINTS) * len(U.VARS) * 4 * 24
    assert (t["q10"] <= t["q50"]).all() and (t["q50"] <= t["q90"]).all()
    assert (t["n"] >= 1).all()


def test_real_coverage_close_to_target():
    """Покрытие на отложенном 2025 г. по ERA5 — в пределах 70–90 % для всех точек и величин (факт шага 3.2)."""
    if not U.OUT.exists():
        pytest.skip("нет таблицы квантилей")
    t = pd.read_csv(U.OUT)
    for p in U.POINTS:
        d = U.paired(p)
        test = d[d["year"] == 2025]
        if test.empty:
            pytest.skip("нет данных 2025 г.")
        for v in U.VARS:
            x = test[test["var"] == v]
            if v == "solar_wm2":
                x = x[(x["fc"] > 0) | (x["obs"] > 0)]
            a = U.apply(x[["time", "var", "fc"]], t[t["point"] == p])
            cov = ((x["obs"].to_numpy() >= a["p10"].to_numpy()) & (x["obs"].to_numpy() <= a["p90"].to_numpy())).mean()
            assert 0.70 <= cov <= 0.90, (p, v, cov)
