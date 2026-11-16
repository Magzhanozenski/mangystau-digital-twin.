"""Эксперимент B: нейросеть против линейной модели на независимом периоде (пункт 3).

Обучение — 2023–2024 (признаки по факту ERA5), проверка — 2025: (а) признаки по факту — качество модели;
(б) признаки по прогнозу на сутки — как в эксплуатации. «Правда» — генератор прототипа (exp_b_data.py).
Модели: L0 (прототип), L1 (честная линейная), MLP (прототип; 10 инициализаций — разброс от случайности).
Метрики MAE, MAPE, RMSE — все часы и жара (факт T ≥ 35 °C). Значимость разницы MAE (MLP − L1 и MLP − L0):
95 % ДИ бутстрепом по суткам и тест Диболда — Мариано (потеря — модуль ошибки, дисперсия Ньюи — Уэста, лаг 24 ч).
Результаты — experiments/exp_b_results.json.

Запуск из корня:  .venv\\Scripts\\python experiments\\exp_b_run.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import exp_b_data as BD  # noqa: E402
import exp_b_models as B  # noqa: E402

OUT = Path(__file__).resolve().parent / "exp_b_results.json"
SEEDS = [(5, 9)] + [(s, s + 100) for s in range(11, 20)]   # (инициализация, перемешивание); первая — как в прототипе
BOOT = 2000


def X(d: pd.DataFrame, fc: bool = False) -> dict:
    s = "_fc" if fc else ""
    return {"h": d["h"].to_numpy(), "T": d[f"T{s}"].to_numpy(), "Tlag": d[f"Tlag{s}"].to_numpy(),
            "Tmax": d[f"Tmax{s}"].to_numpy(), "weekend": d["weekend"].to_numpy().astype(bool)}


def metrics(y, p) -> dict:
    e = y - p
    return {"MAE": float(np.mean(np.abs(e))), "MAPE": float(100 * np.mean(np.abs(e) / y)),
            "RMSE": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "n": int(len(y))}


def dm_test(e1, e2, lag: int = 24) -> tuple[float, float]:
    """Диболд — Мариано: d = |e1| − |e2|; HAC-дисперсия (Ньюи — Уэст). Возвращает (статистика, p, двусторонний)."""
    d = np.abs(e1) - np.abs(e2)
    n = len(d)
    dc = d - d.mean()
    g0 = np.dot(dc, dc) / n
    s = g0 + 2 * sum((1 - k / (lag + 1)) * np.dot(dc[k:], dc[:-k]) / n for k in range(1, lag + 1))
    stat = d.mean() / math.sqrt(s / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return float(stat), float(p)


def boot_diff(e1, e2, days, rng) -> tuple[float, float]:
    """95 % ДИ разницы MAE (модель 1 − модель 2) бутстрепом по суткам."""
    d = pd.DataFrame({"a": np.abs(e1), "b": np.abs(e2), "day": days})
    g = d.groupby("day")[["a", "b"]].agg(["sum", "count"])
    sa, sb, c = g[("a", "sum")].to_numpy(), g[("b", "sum")].to_numpy(), g[("a", "count")].to_numpy()
    idx = rng.integers(0, len(g), size=(BOOT, len(g)))
    vals = (sa[idx].sum(1) - sb[idx].sum(1)) / c[idx].sum(1)
    return tuple(float(x) for x in np.quantile(vals, [0.025, 0.975]))


def main() -> int:
    d = BD.build()
    tr = d[(d.year <= 2024)].dropna(subset=["T", "Tlag", "Tmax"])
    te = d[d.year == 2025]
    te_fc = te.dropna(subset=["T_fc", "Tlag_fc", "Tmax_fc"])
    y_tr, y_te, y_fc = tr["load"].to_numpy(), te["load"].to_numpy(), te_fc["load"].to_numpy()
    hot, hot_fc = te["T"].to_numpy() >= 35, te_fc["T"].to_numpy() >= 35
    preds = {"act": {}, "fc": {}}
    l0 = B.Linear(B.l0_features).fit(X(tr), y_tr)
    l1 = B.Linear(B.l1_features).fit(X(tr), y_tr)
    for name, m in (("L0", l0), ("L1", l1)):
        preds["act"][name] = m.predict(X(te))
        preds["fc"][name] = m.predict(X(te_fc, fc=True))
    mlp_runs = {"act": [], "fc": []}
    for s_init, s_shuf in SEEDS:
        m = B.MLP(seed=s_init, shuffle_seed=s_shuf).fit(X(tr), y_tr)
        mlp_runs["act"].append(m.predict(X(te)))
        mlp_runs["fc"].append(m.predict(X(te_fc, fc=True)))
    preds["act"]["MLP"], preds["fc"]["MLP"] = mlp_runs["act"][0], mlp_runs["fc"][0]   # инициализация прототипа
    rng = np.random.default_rng(3)
    res = {"train": [str(tr.index.min()), str(tr.index.max()), len(tr)], "test": [str(te.index.min()), str(te.index.max())],
           "hot_hours": {"act": int(hot.sum()), "fc": int(hot_fc.sum())}, "seeds": SEEDS, "metrics": {}, "mlp_spread": {},
           "compare": {}}
    for inp, (yy, hh, frame) in {"act": (y_te, hot, te), "fc": (y_fc, hot_fc, te_fc)}.items():
        res["metrics"][inp] = {}
        for name, p in preds[inp].items():
            res["metrics"][inp][name] = {"все": metrics(yy, p), "жара": metrics(yy[hh], p[hh])}
        spread = [metrics(yy, p)["MAE"] for p in mlp_runs[inp]]
        spread_hot = [metrics(yy[hh], p[hh])["MAE"] for p in mlp_runs[inp]]
        res["mlp_spread"][inp] = {"MAE_min": min(spread), "MAE_max": max(spread), "hot_MAE_min": min(spread_hot),
                                  "hot_MAE_max": max(spread_hot)}
        days = frame.index.normalize().to_numpy()
        res["compare"][inp] = {}
        for other in ("L1", "L0"):
            e_n, e_o = yy - preds[inp]["MLP"], yy - preds[inp][other]
            stat, p = dm_test(e_n, e_o)
            ci = boot_diff(e_n, e_o, days, rng)
            stat_h, p_h = dm_test(e_n[hh], e_o[hh], lag=min(24, int(hh.sum()) // 4))
            ci_h = boot_diff(e_n[hh], e_o[hh], days[hh], rng)
            res["compare"][inp][f"MLP−{other}"] = {
                "dMAE": float(np.mean(np.abs(e_n)) - np.mean(np.abs(e_o))), "ci": ci, "dm": stat, "p": p,
                "dMAE_hot": float(np.mean(np.abs(e_n[hh])) - np.mean(np.abs(e_o[hh]))), "ci_hot": ci_h,
                "dm_hot": stat_h, "p_hot": p_h}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    for inp in ("act", "fc"):
        print(f"\n== 2025, признаки {'по факту' if inp == 'act' else 'по прогнозу'} (жарких часов {res['hot_hours'][inp]})")
        for name, m in res["metrics"][inp].items():
            a, h = m["все"], m["жара"]
            print(f"  {name}: MAE {a['MAE']:.2f} МВт, MAPE {a['MAPE']:.2f} %, RMSE {a['RMSE']:.2f} | "
                  f"жара: MAE {h['MAE']:.2f}, MAPE {h['MAPE']:.2f} %, смещение {h['bias']:+.2f}")
        sp = res["mlp_spread"][inp]
        print(f"  MLP, 10 инициализаций: MAE {sp['MAE_min']:.2f}…{sp['MAE_max']:.2f}; жара {sp['hot_MAE_min']:.2f}…{sp['hot_MAE_max']:.2f}")
        for k, c in res["compare"][inp].items():
            print(f"  {k}: ΔMAE {c['dMAE']:+.3f} [{c['ci'][0]:+.3f}; {c['ci'][1]:+.3f}], DM p = {c['p']:.2g} | "
                  f"жара ΔMAE {c['dMAE_hot']:+.3f} [{c['ci_hot'][0]:+.3f}; {c['ci_hot'][1]:+.3f}], p = {c['p_hot']:.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
