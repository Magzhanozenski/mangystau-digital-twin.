"""Независимая проверка суточного графика (шаг 3.1, пункт 4).

1) Каждое ограничение проверяется на готовых графиках отдельным кодом (циклы по часам, без матриц MILP).
2) Малые задачи (3 агрегата × 5 ч, без цикла, с начальным состоянием) решаются полным перебором состава:
   для каждой допустимой матрицы включений — линейная задача загрузки (scipy linprog); минимум сравнивается с MILP.
3) Каждый час графиков network_real проверяется расчётом режима (twin.powerflow).
Отчёт — validation/check_dispatch_report.md. Код выхода 1 — если нарушено ограничение, перебор не совпал
или час не сошёлся. Перенапряжения в часах НЕ считаются ошибкой графика (известная проблема сети, шаг 2.4) —
они только записываются.

Запуск из корня:  .venv\\Scripts\\python validation\\check_dispatch.py
"""
import itertools
import random
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import dispatch as D  # noqa: E402
from twin import load_network  # noqa: E402

REPORT = Path(__file__).resolve().parent / "check_dispatch_report.md"
TOL = 1e-5


def check_constraints(units, sch, cyclic=True):
    """Максимальное нарушение по каждому виду ограничений (0 — всё выполнено)."""
    T = sch.hours
    viol = {k: 0.0 for k in ("мощность", "логика", "набор", "сброс", "мин. работа", "мин. простой",
                             "обязательная работа", "баланс", "резерв")}
    for un in units:
        u, p, v, w = sch.u[un.id], sch.p[un.id], sch.v[un.id], sch.w[un.id]
        su = max(un.ramp, un.p_min)
        for t in range(T):
            tp = (t - 1) % T
            u_prev = u[tp] if (t or cyclic) else int(un.on0)
            p_prev = p[tp] if (t or cyclic) else un.p0
            if u[t]:
                viol["мощность"] = max(viol["мощность"], un.p_min - p[t], p[t] - un.p_max)
            else:
                viol["мощность"] = max(viol["мощность"], abs(p[t]))
            viol["логика"] = max(viol["логика"], abs((u[t] - u_prev) - (v[t] - w[t])), v[t] + w[t] - 1)
            viol["набор"] = max(viol["набор"], p[t] - p_prev - (un.ramp * u_prev + su * v[t]))
            viol["сброс"] = max(viol["сброс"], p_prev - p[t] - (un.ramp * u[t] + su * w[t]))
            if un.must_run:
                viol["обязательная работа"] = max(viol["обязательная работа"], 1 - u[t])
            # минимальное время работы: после пуска в час t агрегат включён ещё min_up часов
            if v[t]:
                for k in range(un.min_up):
                    if not cyclic and t + k >= T:
                        break
                    viol["мин. работа"] = max(viol["мин. работа"], 1 - u[(t + k) % T])
            if w[t]:
                for k in range(un.min_down):
                    if not cyclic and t + k >= T:
                        break
                    viol["мин. простой"] = max(viol["мин. простой"], u[(t + k) % T])
    for t in range(T):
        tot = sum(sch.p[un.id][t] for un in units)
        viol["баланс"] = max(viol["баланс"], abs(tot - sch.demand[t]))
        spare = sum(un.p_max * sch.u[un.id][t] for un in units) - tot
        viol["резерв"] = max(viol["резерв"], sch.reserve[t] - spare)
    return viol


def brute_force(units, load, reserve, loss):
    """Полный перебор матриц включений (без цикла) + LP загрузки. Возвращает минимум топлива или None."""
    n, T = len(units), len(load)
    demand = [l * (1 + loss) for l in load]
    best = None
    for bits in itertools.product((0, 1), repeat=n * T):
        U = np.array(bits).reshape(n, T)
        V = np.zeros_like(U)
        W = np.zeros_like(U)
        ok = True
        for i, un in enumerate(units):
            prev = int(un.on0)
            run = un.hours0
            for t in range(T):
                if un.must_run and not U[i, t]:
                    ok = False
                if U[i, t] and not prev:
                    V[i, t] = 1
                    if not un.on0 and t < un.start_h:
                        ok = False
                    if run < un.min_down:  # не достоял
                        ok = False
                    run = 1
                elif not U[i, t] and prev:
                    W[i, t] = 1
                    if run < un.min_up:  # не доработал
                        ok = False
                    run = 1
                else:
                    run += 1
                prev = U[i, t]
            if not ok:
                break
        if not ok:
            continue
        # LP по p (n·T): границы, набор/сброс, баланс, резерв
        nv = n * T
        idx = lambda i, t: i * T + t  # noqa: E731
        bounds = [(un.p_min * U[i, t], un.p_max * U[i, t]) for i, un in enumerate(units) for t in range(T)]
        A, b, Ae, be = [], [], [], []
        for i, un in enumerate(units):
            su = max(un.ramp, un.p_min)
            for t in range(T):
                if t:
                    r = np.zeros(nv); r[idx(i, t)] = 1; r[idx(i, t - 1)] = -1
                    A.append(r); b.append(un.ramp * U[i, t - 1] + su * V[i, t])
                    r = np.zeros(nv); r[idx(i, t - 1)] = 1; r[idx(i, t)] = -1
                    A.append(r); b.append(un.ramp * U[i, t] + su * W[i, t])
                else:
                    r = np.zeros(nv); r[idx(i, 0)] = 1
                    A.append(r); b.append(un.p0 + (un.ramp if un.on0 else 0) + su * V[i, 0])
                    r = np.zeros(nv); r[idx(i, 0)] = -1
                    A.append(r); b.append(-un.p0 + un.ramp * U[i, 0] + su * W[i, 0])
        for t in range(T):
            r = np.zeros(nv)
            for i in range(n):
                r[idx(i, t)] = 1
            Ae.append(r); be.append(demand[t])
            A.append(r.copy()); b.append(sum(units[i].p_max * U[i, t] for i in range(n)) - reserve)
        c = np.array([un.heat_rate + D.TIE_BREAK * i for i, un in enumerate(units) for t in range(T)])
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), A_eq=np.array(Ae), b_eq=np.array(be), bounds=bounds,
                      method="highs")
        if res.status != 0:
            continue
        val = res.fun + sum(units[i].startup_gj * V[i].sum() for i in range(n))
        best = val if best is None else min(best, val)
    return best


def random_case(rng):
    units = []
    for k in range(3):
        pmax = rng.choice([60, 80, 100])
        units.append(D.Unit(f"U{k}", p_min=round(pmax * rng.uniform(0.2, 0.5), 1), p_max=pmax,
                            ramp=round(pmax * rng.uniform(0.2, 1.0), 1), start_h=rng.choice([0, 1, 2]),
                            min_up=rng.choice([1, 2, 3]), min_down=rng.choice([1, 2]),
                            heat_rate=rng.uniform(8, 12), startup_gj=rng.uniform(0, 300),
                            on0=(k == 0), p0=pmax * 0.6 if k == 0 else 0.0, hours0=rng.choice([1, 5])))
    load = [rng.uniform(40, 120) for _ in range(5)]
    return units, load


def main() -> int:
    out, bad = [], 0

    def say(s=""):
        print(s)
        out.append(s)
    say("# Проверка суточного графика (twin/dispatch.py)")
    # 2) перебор
    say("")
    say("## Полный перебор на малых задачах (3 агрегата × 5 ч, начальное состояние, без цикла)")
    rng = random.Random(7)
    feasible, k = 0, 0
    while feasible < 6 and k < 25:  # набираем не меньше 6 задач, где допустимый график есть
        k += 1
        units, load = random_case(rng)
        reserve, loss = 10.0, 0.0
        sch = D.solve(units, load, reserve_mw=reserve, loss=loss, cyclic=False)
        bf = brute_force(units, load, reserve, loss)
        if not sch.ok and bf is None:
            say(f"- задача {k}: обе постановки — нет допустимого графика (совпадает)")
            continue
        if not sch.ok or bf is None:
            say(f"- задача {k}: РАСХОЖДЕНИЕ — MILP {sch.status}, перебор {bf}")
            bad += 1
            continue
        viol = check_constraints(units, sch, cyclic=False)
        d = abs(sch.fuel_gj - bf) / max(1.0, abs(bf))
        ok = d < 1e-6 and max(viol.values()) < TOL
        bad += 0 if ok else 1
        feasible += 1
        starts = sum(sum(v) for v in sch.v.values())
        say(f"- задача {k} (пусков {starts}): MILP {sch.fuel_gj:.4f} ГДж, перебор {bf:.4f} ГДж, отн. разница {d:.1e}; "
            f"макс. нарушение ограничений {max(viol.values()):.1e} — {'совпадает' if ok else 'РАСХОЖДЕНИЕ'}")
    if feasible < 6:
        say(f"- ВНИМАНИЕ: допустимых задач только {feasible}")
        bad += 1
    # 1), 3) графики сети
    net = load_network(ROOT / "data" / "network_real.yaml")
    for scen in (None, "2026"):
        for day in ("average", "peak"):
            units = D.units_from_network(net, scen)
            load = D.daily_load(day)
            sch = D.solve(units, load)
            say("")
            say(f"## Сутки «{day}», состав {'2026 (с ГПЭС Жанаозен)' if scen else '2023'}")
            if not sch.ok:
                say(f"- MILP: {sch.status}")
                bad += 1
                continue
            viol = check_constraints(units, sch)
            worst = max(viol.values())
            bad += 0 if worst < TOL else 1
            starts = sum(sum(v) for v in sch.v.values())
            say(f"- топливо {sch.fuel_gj:,.0f} ГДж; нагрузка {min(load):.0f}–{max(load):.0f} МВт; пусков {starts}")
            say(f"- ограничения: макс. нарушение {worst:.1e} — {'выполнены' if worst < TOL else 'НАРУШЕНЫ'} "
                f"({', '.join(k for k, v in viol.items() if v >= TOL) or 'все'})")
            on = {uid: sum(sch.u[uid]) for uid in sch.units}
            say("- часы работы: " + ", ".join(f"{k} {v}" for k, v in on.items()))
            pf = D.hourly_power_flow(net, sch, load)
            nok = sum(h["status"] != "ok" for h in pf)
            bad += 1 if nok else 0
            dsl = max(abs(h["slack_pf"] - h["slack_plan"]) for h in pf if h["status"] == "ok")
            hv = sum(1 for h in pf if h["status"] == "ok" and h["maxV"] > 1.10)
            say(f"- расчёт режима по часам: сошлось {24 - nok}/24; |P балансирующего − план| ≤ {dsl:.1f} МВт "
                f"(разница потерь с оценкой 1,4 %); часов с U > 1,10 о.е.: {hv}; "
                f"U {min(h['minV'] for h in pf if h['minV']):.3f}…{max(h['maxV'] for h in pf if h['maxV']):.3f} о.е.")
            say("- график, МВт (· — выключен):")
            say("")
            say("| агрегат | " + " | ".join(str(t) for t in range(24)) + " |")
            say("|---|" + "---|" * 24)
            for uid in sch.units:
                say(f"| {uid} | " + " | ".join(f"{p:.0f}" if sch.u[uid][t] else "·" for t, p in enumerate(sch.p[uid])) + " |")
    say("")
    say("Итог: " + ("все проверки пройдены" if not bad else f"{bad} проверок не пройдено"))
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
