"""Проверка интервалов P10–P90 прогноза погоды (шаг 3.2, пункт 2) на отложенном 2025 г.

Для каждой точки и величины: доля фактов ERA5 внутри [P10, P90] (цель 80 %), 95 % доверительный интервал —
бутстреп по суткам (часы внутри суток зависимы), средняя ширина, средний пинбол-лосс (q10, q90);
то же для простого варианта «прогноз ± 1,2816·σ». Разрезы: сезоны, часть суток, жаркие часы (прогноз T ≥ 35 °C),
слабый ветер (прогноз < 2 м/с — опасный для DLR). Радиация — только светлое время (прогноз > 0 или факт > 0).
Сверка со станциями NOAA ISD (Актау WMO 38111, Бейнеу WMO 35927; ряды до 08.2025) — интервалы, откалиброванные
по ERA5, против наблюдений.
Покрытие не подгоняется: код выхода 1 только при сбое данных. Отчёт — validation/check_uncertainty_report.md.

Запуск из корня:  .venv\\Scripts\\python validation\\check_uncertainty.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

from twin import uncertainty as U  # noqa: E402

REPORT = Path(__file__).resolve().parent / "check_uncertainty_report.md"
Z = 1.2816
UNITS = {"T_c": "°C", "wind_ms": "м/с", "solar_wm2": "Вт/м²"}
NAMES = {"T_c": "температура", "wind_ms": "ветер", "solar_wm2": "радиация"}
B = 1000
rng = np.random.default_rng(11)


def pinball(y, q, tau):
    d = y - q
    return np.mean(np.maximum(tau * d, (tau - 1) * d))


def boot_ci(df, col="inside"):
    """95 % ДИ доли по бутстрепу суток."""
    days = df["time"].dt.normalize()
    g = df.groupby(days)[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    if len(g) < 5:
        return (np.nan, np.nan)
    idx = rng.integers(0, len(g), size=(B, len(g)))
    vals = s[idx].sum(1) / c[idx].sum(1)
    return tuple(np.quantile(vals, [0.025, 0.975]))


def evaluate(x, lo, hi, p10, p90):
    """x — DataFrame с obs; lo/hi — границы интервала."""
    x = x.assign(inside=((x["obs"] >= lo) & (x["obs"] <= hi)).astype(float))
    ci = boot_ci(x)
    return {"n": len(x), "cov": x["inside"].mean(), "ci": ci, "width": float(np.mean(hi - lo)),
            "pin": (pinball(x["obs"].to_numpy(), p10, 0.1) + pinball(x["obs"].to_numpy(), p90, 0.9)) / 2}


def fmt(r):
    if not r["n"]:
        return "—"
    return f"{100 * r['cov']:.1f} % [{100 * r['ci'][0]:.1f}; {100 * r['ci'][1]:.1f}]"


def main() -> int:
    out = []

    def say(s=""):
        print(s)
        out.append(s)
    table = pd.read_csv(U.OUT)
    say("# Покрытие интервалов P10–P90 прогноза погоды на сутки (отложенный 2025 г.)")
    say("")
    say("Цель — 80 %. В скобках — 95 % ДИ (бутстреп по суткам). Калибровка — 2023–2024, «простой» — прогноз ± 1,28σ.")
    say("Пинбол — средний по квантилям 10 % и 90 % (меньше — лучше).")
    errors = 0
    summary = []
    for p in U.POINTS:
        d = U.paired(p)
        test = d[d["year"].isin(U.TEST_YEARS)]
        if test.empty:
            say(f"- {p}: нет данных 2025 г.")
            errors += 1
            continue
        nv = U.naive(p, df=d).set_index("var")["sigma"]
        say("")
        say(f"## {p}")
        say("")
        say("| величина | разрез | часов | покрытие (калибр.) | ширина | пинбол | покрытие (простой) | ширина | пинбол |")
        say("|---|---|---|---|---|---|---|---|---|")
        for v in U.VARS:
            x = test[test["var"] == v].copy()
            if v == "solar_wm2":
                x = x[(x["fc"] > 0) | (x["obs"] > 0)]
            a = U.apply(x[["time", "var", "fc"]], table[table["point"] == p]).set_index(x.index)
            lo_n = x["fc"] - Z * nv[v]
            hi_n = x["fc"] + Z * nv[v]
            if U.VARS[v] is not None:
                lo_n, hi_n = lo_n.clip(lower=0), hi_n.clip(lower=0)
            cuts = {"весь год": x.index}
            for s in ("зима", "весна", "лето", "осень"):
                cuts[s] = x.index[x["season"] == s]
            cuts["ночь 0–5 ч"] = x.index[x["hour"] < 6]
            cuts["день 12–17 ч"] = x.index[(x["hour"] >= 12) & (x["hour"] < 18)]
            tfc = test[test["var"] == "T_c"].set_index("time")["fc"]
            hot = x.index[x["time"].map(tfc).fillna(-99) >= 35]
            cuts["жара (прогноз T ≥ 35 °C)"] = hot
            if v == "wind_ms":
                cuts["слабый ветер (прогноз < 2 м/с)"] = x.index[x["fc"] < 2]
            for name, ix in cuts.items():
                if not len(ix):
                    continue
                xc = x.loc[ix]
                rc = evaluate(xc, a.loc[ix, "p10"].to_numpy(), a.loc[ix, "p90"].to_numpy(),
                              a.loc[ix, "p10"].to_numpy(), a.loc[ix, "p90"].to_numpy())
                rn = evaluate(xc, lo_n.loc[ix].to_numpy(), hi_n.loc[ix].to_numpy(),
                              lo_n.loc[ix].to_numpy(), hi_n.loc[ix].to_numpy())
                say(f"| {NAMES[v]} | {name} | {rc['n']} | {fmt(rc)} | {rc['width']:.2f} | {rc['pin']:.3f} | "
                    f"{fmt(rn)} | {rn['width']:.2f} | {rn['pin']:.3f} |")
                if name == "весь год":
                    summary.append((p, v, rc, rn))
    # станции
    say("")
    say("## Сверка со станциями NOAA ISD (интервалы по ERA5 против наблюдений)")
    say("")
    try:
        import check_weather as cw
        sts = {"aktau": ("38111099999", "Актау WMO 38111"), "beyneu": ("35927099999", "Бейнеу WMO 35927")}
        say("| точка | станция | величина | часов | покрытие | средняя ошибка (станция − P50) |")
        say("|---|---|---|---|---|---|")
        for p, (sid, desc) in sts.items():
            obs = cw.station(sid)
            obs = obs[obs["time"].dt.year.isin(U.TEST_YEARS)]
            d = U.paired(p, U.TEST_YEARS)
            for v in ("T_c", "wind_ms"):
                x = d[d["var"] == v][["time", "var", "fc"]]
                a = U.apply(x, table[table["point"] == p]).merge(obs[["time", v]].rename(columns={v: "st"}), on="time").dropna()
                a = a.assign(obs=a["st"])
                r = evaluate(a, a["p10"].to_numpy(), a["p90"].to_numpy(), a["p10"].to_numpy(), a["p90"].to_numpy())
                say(f"| {p} | {desc} | {NAMES[v]} | {r['n']} | {fmt(r)} | {np.mean(a['st'] - a['p50']):+.2f} {UNITS[v]} |")
    except Exception as e:  # станции — дополнительная сверка; без кэша/интернета не ломаем проверку
        say(f"- станции не проверены: {e!r}")
    say("")
    say("## Итог (весь 2025 г.)")
    say("")
    for p, v, rc, rn in summary:
        verdict = "в пределах ДИ" if rc["ci"][0] <= 0.8 <= rc["ci"][1] else ("занижено" if rc["cov"] < 0.8 else "завышено")
        better = "лучше" if rc["pin"] < rn["pin"] else "НЕ лучше"
        say(f"- {p}, {NAMES[v]}: покрытие {fmt(rc)} — {verdict}; пинбол калибровки {better} простого "
            f"({rc['pin']:.3f} против {rn['pin']:.3f}), ширина {rc['width']:.2f} против {rn['width']:.2f} {UNITS[v]}")
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
