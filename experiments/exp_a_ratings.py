"""Эксперимент A: справочный допустимый ток против DLR при одинаковой нагрузке (этап 4).

«Правда» — DLR по фактической погоде (ERA5, ветер приведён к станциям; twin/dlr.py). Методы (все — по прогнозу на
сутки, как в эксплуатации):
  M1 — ПУЭ табл. 1.3.29 без поправки на температуру (510 / 710 А);
  M2 — ПУЭ × поправка табл. 1.3.3 по прогнозной температуре воздуха;
  M3 — DLR по прогнозу (P50: прогноз + медиана ошибки шага 3.2; ветер приведён к станциям);
  M4 — осторожный DLR по прогнозу: T и солнце — P90, ветер — P10 (интервалы шага 3.2), ветер приведён к станциям.
Сеть — network_real (состав генерации 2023 г.), нормальная схема и N-1 по каждой цепи. Токи — на сетке уровней
нагрузки с линейной интерполяцией по часам (токи от метода оценки допустимого тока не зависят).
События на «линия × схема × час»: перегрузка = ток > допустимого. Ложная тревога — метод видит перегрузку, «правды»
нет; скрытый риск — «правда» есть, метод не видит. Уровень «час»: был ли в этот час хоть один случай (любая цепь,
любая схема N-1).
Результаты — experiments/exp_a_results.json.

Запуск из корня:  .venv\\Scripts\\python experiments\\exp_a_ratings.py [--stress 1.0 1.2 1.4]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from twin import dlr as L  # noqa: E402
from twin import load_network, run_power_flow  # noqa: E402
from twin import uncertainty as U  # noqa: E402
import exp_a_load  # noqa: E402

OUT = Path(__file__).resolve().parent / "exp_a_results.json"
YEAR = 2025
METHODS = {"M1": "ПУЭ без поправки", "M2": "ПУЭ с поправкой на T прогноза", "M3": "DLR по прогнозу (P50)",
           "M4": "осторожный DLR по прогнозу (ветер P10, T и солнце P90)"}


# ---------------- токи по сетке нагрузки ----------------
def current_grid(net, levels: np.ndarray, contingencies: list) -> tuple[np.ndarray, list, list]:
    """I[уровень, схема, цепь], А (max по концам); ok[уровень, схема] — режим сошёлся."""
    lines = [l["id"] for l in net.lines]
    base = net.raw["meta"]["load_scenarios"]["base"]
    scen_out = [g["id"] for g in net.gens if g.get("scenario")]
    I = np.zeros((len(levels), len(contingencies), len(lines)), dtype=np.float32)
    ok = np.zeros((len(levels), len(contingencies)), dtype=bool)
    for a, lev in enumerate(levels):
        ctrl = net.default_ctrl()
        ctrl["gP"] = {g: p * lev / base for g, p in ctrl["gP"].items()}
        for c, out in enumerate(contingencies):
            r = run_power_flow({"T": 25, "load": float(lev), "ctrl": ctrl, "gensOut": scen_out, "linesOut": out}, net)
            ok[a, c] = r["status"] == "ok"
            if ok[a, c]:
                I[a, c] = [r["flows"][lid]["I"] for lid in lines]
    return I, ok, lines


def interp_hours(I_grid: np.ndarray, levels: np.ndarray, load: np.ndarray) -> np.ndarray:
    """Линейная интерполяция по нагрузке: I_grid[уровень, цепь] → I[час, цепь]."""
    k = np.clip(np.searchsorted(levels, load) - 1, 0, len(levels) - 2)
    w = ((load - levels[k]) / (levels[k + 1] - levels[k]))[:, None]
    return (1 - w) * I_grid[k] + w * I_grid[k + 1]


# ---------------- допустимые токи по методам ----------------
def weather(point: str, kind: str, table: pd.DataFrame, wmap: dict) -> dict:
    """Ряды T, ветер (приведённый к станции), направление, солнце для точки.
    kind: truth (ERA5), p50 (прогноз + медиана ошибки), cons (T, солнце — P90; ветер — P10)."""
    src = "archive" if kind == "truth" else "day_ahead"
    w = pd.read_csv(ROOT / "data" / "weather" / src / f"{point}_{YEAR}.csv", parse_dates=["time"])
    out = {"time": w["time"], "dir": w["wind_dir_deg"].to_numpy(), "elev": float(w["grid_elevation"].iloc[0])}
    if kind == "truth":
        T, V, S = w["T_c"].to_numpy(), w["wind_ms"].to_numpy(), w["solar_wm2"].to_numpy()
    else:
        tab = table[table["point"] == point]
        q = {v: U.apply(pd.DataFrame({"time": w["time"], "var": v, "fc": w[v]}), tab) for v in U.VARS}
        if kind == "p50":
            T, V, S = (q[v]["p50"].to_numpy() for v in ("T_c", "wind_ms", "solar_wm2"))
        else:
            T, V, S = q["T_c"]["p90"].to_numpy(), q["wind_ms"]["p10"].to_numpy(), q["solar_wm2"]["p90"].to_numpy()
        out["T_fc_raw"] = w["T_c"].to_numpy()
    out.update(T=T, V=L.map_wind(V, *wmap[point]), S=S)
    return out


def ratings(net_raw: dict) -> tuple[dict, dict, dict]:
    """Допустимые токи по маршрутам: {метод: {маршрут: ряд}}, сведения о маршрутах, условия «правды»."""
    geo = L.line_geometries(net_raw)
    table = pd.read_csv(U.OUT)
    wmap = L.load_wind_mapping()
    wx = {}
    res = {m: {} for m in ("truth", "M1", "M2", "M3", "M4")}
    info, cond = {}, {}
    for key, g in geo.items():
        pt, dist = L.nearest_point(*g["mid"])
        for kind in ("truth", "p50", "cons"):
            if (pt, kind) not in wx:
                wx[(pt, kind)] = weather(pt, kind, table, wmap)
        tr, p5, cs = wx[(pt, "truth")], wx[(pt, "p50")], wx[(pt, "cons")]
        c = g["conductor"]
        res["truth"][key] = L.line_rating(c, g["az"], tr["T"], tr["V"], tr["dir"], tr["S"], tr["elev"])
        res["M1"][key] = np.full(len(tr["T"]), L.CONDUCTORS[c]["I_pue"])
        res["M2"][key] = L.static_pue(c, p5["T_fc_raw"])
        res["M3"][key] = L.line_rating(c, g["az"], p5["T"], p5["V"], p5["dir"], p5["S"], p5["elev"])
        res["M4"][key] = L.line_rating(c, g["az"], cs["T"], cs["V"], cs["dir"], cs["S"], cs["elev"])
        info[key] = {"point": pt, "dist_km": round(dist, 1), "conductor": c, "name": g["name"]}
        cond[key] = {"hot": tr["T"] >= 35, "calm": tr["V"] < 1.0}
    return res, info, cond


# ---------------- подсчёт событий ----------------
def count(net, I_grid, ok, levels, conts, load, rat, cond, stress):
    lines = [l["id"] for l in net.lines]
    route = [lid.rsplit("-", 1)[0] for lid in lines]
    H = len(load)
    R = {m: np.stack([rat[m][r] for r in route], axis=1) for m in rat}   # [час, цепь]
    hot = np.stack([cond[r]["hot"] for r in route], axis=1)
    calm = np.stack([cond[r]["calm"] for r in route], axis=1)
    keys = ("TP", "FP", "FN", "TN")
    el = {m: {k: 0 for k in keys} for m in METHODS}
    el_hot = {m: {k: 0 for k in keys} for m in METHODS}
    el_calm = {m: {k: 0 for k in keys} for m in METHODS}
    hour_true = np.zeros(H, bool)
    hour_alarm = {m: np.zeros(H, bool) for m in METHODS}
    lstat = {m: {"FP": np.zeros(len(lines), int), "FN": np.zeros(len(lines), int)} for m in METHODS}
    skipped = 0
    maxload = {m: 0.0 for m in ("truth",) + tuple(METHODS)}
    L_h = load * stress
    for c, out in enumerate(conts):
        good = ok[:, c]
        if not good.all():
            skipped += int((~good).sum())
        if good.sum() < 2:
            continue
        lv, Ig = levels[good], I_grid[good, c]
        I = interp_hours(Ig, lv, L_h)
        inserv = np.array([lid not in out for lid in lines])[None, :]
        valid = (L_h >= lv[0]) & (L_h <= lv[-1])
        mask = inserv & valid[:, None]
        for m in maxload:  # наибольшая загрузка относительно допустимого тока метода
            maxload[m] = max(maxload[m], float(np.max(np.where(mask, I / R[m], 0.0))))
        tv = (I > R["truth"]) & mask
        hour_true |= tv.any(1)
        for m in METHODS:
            al = (I > R[m]) & mask
            hour_alarm[m] |= al.any(1)
            cm = {"TP": tv & al, "FP": al & ~tv, "FN": tv & ~al, "TN": mask & ~tv & ~al}
            for k, v in cm.items():
                el[m][k] += int(v.sum())
                el_hot[m][k] += int((v & hot).sum())
                el_calm[m][k] += int((v & calm).sum())
            lstat[m]["FP"] += cm["FP"].sum(0)
            lstat[m]["FN"] += cm["FN"].sum(0)
    hour = {m: {"TP": int((hour_true & hour_alarm[m]).sum()), "FP": int((~hour_true & hour_alarm[m]).sum()),
                "FN": int((hour_true & ~hour_alarm[m]).sum()), "TN": int((~hour_true & ~hour_alarm[m]).sum())}
            for m in METHODS}
    worst = {m: {k: sorted(((int(v), lines[i]) for i, v in enumerate(lstat[m][k]) if v), reverse=True)[:5]
                 for k in ("FP", "FN")} for m in METHODS}
    return {"stress": stress, "element": el, "element_hot": el_hot, "element_calm": el_calm, "hour": hour,
            "hours_true": int(hour_true.sum()), "worst": worst, "max_loading": maxload,
            "skipped_level_contingency": skipped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=float, nargs="+", default=[1.0])
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")
    t0 = time.time()
    net = load_network(ROOT / "data" / "network_real.yaml")
    load = exp_a_load.hourly_load(YEAR)["load_mw"].to_numpy()
    lo, hi = load.min() * min(args.stress) * 0.98, load.max() * max(args.stress) * 1.02
    levels = np.linspace(lo, hi, int(np.ceil((hi - lo) / 10)) + 1)
    conts = [[]] + [[l["id"]] for l in net.lines]
    I_grid, ok, _ = current_grid(net, levels, conts)
    rat, info, cond = ratings(net.raw)
    results = {"year": YEAR, "levels_mw": [float(levels[0]), float(levels[-1]), len(levels)],
               "contingencies": len(conts), "lines": len(net.lines), "methods": METHODS,
               "ratio_truth_to_pue_median": float(np.median(np.concatenate(
                   [rat["truth"][k] / rat["M1"][k] for k in rat["truth"]]))),
               "runs": [count(net, I_grid, ok, levels, conts, load, rat, cond, s) for s in args.stress],
               "seconds": round(time.time() - t0, 1)}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    keep = [r for r in prev.get("runs", []) if r["stress"] not in args.stress]
    results["runs"] = sorted(keep + results["runs"], key=lambda r: r["stress"])
    OUT.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in results["runs"]:
        print(f"\n× {r['stress']}: часов с перегрузкой по «правде» (N-1) — {r['hours_true']}; макс. загрузка: "
              + ", ".join(f"{k} {v:.2f}" for k, v in r["max_loading"].items()))
        for m in METHODS:
            e, h = r["element"][m], r["hour"][m]
            print(f"  {m}: линия·схема·час TP {e['TP']}, ложных {e['FP']}, скрытых {e['FN']} | "
                  f"часы: TP {h['TP']}, ложных {h['FP']}, скрытых {h['FN']}")
    print(f"\nвремя {results['seconds']} с")
    return 0


if __name__ == "__main__":
    sys.exit(main())
