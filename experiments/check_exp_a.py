"""Проверка эксперимента A без интерполяции (пункт 4).

Для выборки часов (все часы с событиями при ×1,6 + случайные) режим считается напрямую при точной нагрузке часа
для нормальной схемы и всех N-1; события пересчитываются простым кодом и сравниваются с событиями из интерполяции
(exp_a_ratings). Отчёт — experiments/check_exp_a_report.md; код выхода 1 — если решения о перегрузке расходятся
больше чем в 0,1 % проверенных «линия·схема·час».

Запуск из корня:  .venv\\Scripts\\python experiments\\check_exp_a.py
"""
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import exp_a_load  # noqa: E402
import exp_a_ratings as A  # noqa: E402
from twin import load_network, run_power_flow  # noqa: E402

REPORT = Path(__file__).resolve().parent / "check_exp_a_report.md"
STRESS = 1.6
N_RANDOM = 150


def main() -> int:
    warnings.filterwarnings("ignore")
    net = load_network(ROOT / "data" / "network_real.yaml")
    load = exp_a_load.hourly_load(A.YEAR)["load_mw"].to_numpy() * STRESS
    lines = [l["id"] for l in net.lines]
    route = [lid.rsplit("-", 1)[0] for lid in lines]
    conts = [[]] + [[lid] for lid in lines]
    rat, _, _ = A.ratings(net.raw)
    R = {m: np.stack([rat[m][r] for r in route], axis=1) for m in rat}
    lo, hi = load.min() * 0.98, load.max() * 1.02
    levels = np.linspace(lo, hi, int(np.ceil((hi - lo) / 10)) + 1)
    I_grid, ok, _ = A.current_grid(net, levels, conts)
    # выборка: часы, где по интерполяции есть хоть одна перегрузка по любому методу, + случайные
    ev = set()
    for c in range(len(conts)):
        I = A.interp_hours(I_grid[:, c], levels, load)
        for m in R:
            ev |= set(np.where((I > R[m]).any(1))[0].tolist())
    rng = np.random.default_rng(5)
    hours = sorted(ev | set(rng.choice(len(load), N_RANDOM, replace=False).tolist()))
    base = net.raw["meta"]["load_scenarios"]["base"]
    scen_out = [g["id"] for g in net.gens if g.get("scenario")]
    checked = mism = 0
    max_rel = 0.0
    by_m = {m: 0 for m in R}
    for h in hours:
        ctrl = net.default_ctrl()
        ctrl["gP"] = {g: p * load[h] / base for g, p in ctrl["gP"].items()}
        for c, out in enumerate(conts):
            r = run_power_flow({"T": 25, "load": float(load[h]), "ctrl": ctrl, "gensOut": scen_out, "linesOut": out}, net)
            if r["status"] != "ok":
                continue
            Id = np.array([r["flows"][lid]["I"] for lid in lines])
            Ii = A.interp_hours(I_grid[:, c], levels, np.array([load[h]]))[0]
            inserv = np.array([lid not in out for lid in lines])
            big = inserv & (Id > 1)
            if big.any():
                max_rel = max(max_rel, float(np.max(np.abs(Ii[big] - Id[big]) / Id[big])))
            for m in R:
                d = (Id > R[m][h]) & inserv
                i = (Ii > R[m][h]) & inserv
                diff = int((d != i).sum())
                mism += diff
                by_m[m] += diff
                checked += int(inserv.sum())
    share = mism / max(checked, 1)
    out = ["# Проверка эксперимента A без интерполяции", "",
           f"- нагрузка ×{STRESS}; часов проверено {len(hours)} (с событиями — {len(ev)}, случайных — {N_RANDOM}); "
           f"«линия·схема·час·метод» — {checked}",
           f"- макс. относительная ошибка тока интерполяции: {100 * max_rel:.3f} %",
           f"- расхождений в решении «перегрузка да/нет»: {mism} ({100 * share:.4f} %) — по методам: "
           + ", ".join(f"{m} {v}" for m, v in by_m.items()),
           f"- итог: {'интерполяция не меняет результат' if share < 1e-3 else 'РАСХОЖДЕНИЕ больше 0,1 %'}"]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0 if share < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
