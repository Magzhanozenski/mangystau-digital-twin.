"""Суточный график включения и загрузки агрегатов (шаг 3.1): задача MILP, решатель HiGHS (scipy.optimize.milp).

Ограничения агрегата i в час t (u — включён, v — пуск, w — останов, p — мощность, МВт):
  p_min·u ≤ p ≤ p_max·u;  u_t − u_{t−1} = v_t − w_t;  v_t + w_t ≤ 1
  набор:  p_t − p_{t−1} ≤ R·u_{t−1} + max(R, p_min)·v_t      (в час пуска — не больше max(R, p_min))
  сброс:  p_{t−1} − p_t ≤ R·u_t + max(R, p_min)·w_t          (в час останова — с уровня не выше max(R, p_min))
  мин. время работы:  Σ v_τ (τ = t−UT+1 … t) ≤ u_t;   мин. время простоя:  Σ w_τ (τ = t−DT+1 … t) ≤ 1 − u_t
  пуск: агрегат, выключенный в начале суток, не может выдать мощность раньше часа start_h;  обязательная работа: u = 1.
Система:  Σ p = нагрузка·(1 + потери);   Σ (p_max·u − p) ≥ резерв.
Цель: минимум топлива  Σ (удельный расход·p + топливо на пуск·v),  ГДж.

Ограничения записаны и проверяются отдельно (validation/check_dispatch.py), без матриц MILP.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .network import Network, load_network

ROOT = Path(__file__).resolve().parent.parent
LOSS_FACTOR = 0.014  # ДОПУЩЕНИЕ: потери в сети ≈ 1,4 % нагрузки (check_network_real: 4,7–7,7 МВт при 314–546 МВт)
TIE_BREAK = 1e-4     # ГДж/МВт·ч на номер агрегата: делает решение однозначным при равных удельных расходах (не физика)


@dataclass
class Unit:
    id: str
    p_min: float
    p_max: float
    ramp: float
    start_h: int
    min_up: int
    min_down: int
    heat_rate: float
    startup_gj: float
    must_run: bool = False
    on0: bool = False          # состояние перед первым часом
    p0: float = 0.0            # мощность перед первым часом
    hours0: int = 99           # сколько часов агрегат уже в этом состоянии


@dataclass
class Schedule:
    ok: bool
    status: str
    hours: int
    units: list = field(default_factory=list)
    u: dict = field(default_factory=dict)
    p: dict = field(default_factory=dict)
    v: dict = field(default_factory=dict)
    w: dict = field(default_factory=dict)
    fuel_gj: float = float("nan")
    demand: list = field(default_factory=list)
    reserve: list = field(default_factory=list)


def units_from_network(net: Network, scenario: str | None = None, initial: dict | None = None) -> list[Unit]:
    """Агрегаты из network_real: генераторы с параметрами графика; эквиваленты с fixed_p (Атырау) не входят.
    scenario=None — состав 2023 г. (без агрегатов сценариев); "2026" — с ГПЭС Жанаозен.
    initial: {id: {"on": bool, "p": МВт, "hours": ч}}; по умолчанию включены только обязательные (на минимуме)."""
    out = []
    for g in net.gens:
        if g.get("fixed_p") or "p_min_mw" not in g:
            continue
        if g.get("scenario") and g["scenario"] != scenario:
            continue
        ini = (initial or {}).get(g["id"], {})
        on0 = ini.get("on", bool(g.get("must_run")))
        out.append(Unit(g["id"], g["p_min_mw"], g["p_max_mw"], g["ramp_mw_h"], int(g["start_h"]), int(g["min_up_h"]),
                        int(g["min_down_h"]), g["heat_rate_gj_mwh"], g["startup_gj"], bool(g.get("must_run")),
                        on0, ini.get("p", g["p_min_mw"] if on0 else 0.0), ini.get("hours", 99)))
    return out


def daily_load(day: str = "average", path: Path = ROOT / "data" / "network_sources" / "load_estimates.yaml") -> list[float]:
    d = yaml.safe_load(path.read_text(encoding="utf-8"))["daily_profile"]
    return [d["days"][day]["mean_mw"] * s for s in d["shape"]]


def solve(units: list[Unit], load: list[float], reserve_mw: float | None = None, loss: float = LOSS_FACTOR,
          cyclic: bool = True, time_limit: float = 60.0) -> Schedule:
    """reserve_mw=None — резерв = мощность крупнейшего агрегата (критерий N-1 по генерации, допущение).
    cyclic=True — типовые сутки: состояние перед первым часом = состояние последнего часа (окна минимального
    времени работы/простоя замыкаются по кругу); cyclic=False — начальное состояние из Unit.on0/p0/hours0."""
    T, n = len(load), len(units)
    res = max(u.p_max for u in units) if reserve_mw is None else reserve_mw
    demand = [l * (1 + loss) for l in load]
    # индексы переменных: [u, v, w, p] × (агрегат, час)
    def iu(i, t): return (0 * n + i) * T + t
    def iv(i, t): return (1 * n + i) * T + t
    def iw(i, t): return (2 * n + i) * T + t
    def ip(i, t): return (3 * n + i) * T + t
    N = 4 * n * T
    c = np.zeros(N)
    lb, ub = np.zeros(N), np.ones(N)
    integ = np.zeros(N)
    for i, un in enumerate(units):
        for t in range(T):
            c[ip(i, t)] = un.heat_rate + TIE_BREAK * i  # разрыв равенства у одинаковых агрегатов
            c[iv(i, t)] = un.startup_gj
            integ[iu(i, t)] = integ[iv(i, t)] = integ[iw(i, t)] = 1
            ub[ip(i, t)] = un.p_max
            if un.must_run:
                lb[iu(i, t)] = 1
            if not cyclic and not un.on0 and t < un.start_h:   # пуск: раньше start_h часов мощность не выдать
                ub[iu(i, t)] = ub[iv(i, t)] = 0
    rows, lo, hi = [], [], []

    def add(coefs, l, h):
        rows.append(coefs)
        lo.append(l)
        hi.append(h)
    inf = np.inf
    for i, un in enumerate(units):
        su = max(un.ramp, un.p_min)
        for t in range(T):
            add({ip(i, t): 1, iu(i, t): -un.p_min}, 0, inf)
            add({ip(i, t): 1, iu(i, t): -un.p_max}, -inf, 0)
            tp = (t - 1) % T
            if t or cyclic:
                add({iu(i, t): 1, iu(i, tp): -1, iv(i, t): -1, iw(i, t): 1}, 0, 0)
            else:
                c0 = 1 if un.on0 else 0
                add({iu(i, t): 1, iv(i, t): -1, iw(i, t): 1}, c0, c0)
            add({iv(i, t): 1, iw(i, t): 1}, -inf, 1)
            # набор и сброс
            if t or cyclic:
                add({ip(i, t): 1, ip(i, tp): -1, iu(i, tp): -un.ramp, iv(i, t): -su}, -inf, 0)
                add({ip(i, tp): 1, ip(i, t): -1, iu(i, t): -un.ramp, iw(i, t): -su}, -inf, 0)
            else:
                r0 = un.ramp if un.on0 else 0.0
                add({ip(i, 0): 1, iv(i, 0): -su}, -inf, un.p0 + r0)
                add({ip(i, 0): -1, iu(i, 0): -un.ramp, iw(i, 0): -su}, -inf, -un.p0)
            # минимальное время работы и простоя (с учётом состояния до начала суток)
            win_up = [(t - k) % T for k in range(min(un.min_up, T))] if cyclic else range(max(0, t - un.min_up + 1), t + 1)
            win_dn = [(t - k) % T for k in range(min(un.min_down, T))] if cyclic else range(max(0, t - un.min_down + 1), t + 1)
            add({**{iv(i, k): 1 for k in win_up}, iu(i, t): -1}, -inf, 0)
            add({**{iw(i, k): 1 for k in win_dn}, iu(i, t): 1}, -inf, 1)
        if not cyclic and un.on0 and un.hours0 < un.min_up:      # должен доработать
            for t in range(min(T, un.min_up - un.hours0)):
                lb[iu(i, t)] = 1
        if not cyclic and not un.on0 and un.hours0 < un.min_down:  # должен достоять
            for t in range(min(T, un.min_down - un.hours0)):
                ub[iu(i, t)] = 0
    for t in range(T):
        add({ip(i, t): 1 for i in range(n)}, demand[t], demand[t])
        add({**{iu(i, t): units[i].p_max for i in range(n)}, **{ip(i, t): -1 for i in range(n)}}, res, inf)
    A = lil_matrix((len(rows), N))
    for r, coefs in enumerate(rows):
        for j, a in coefs.items():
            A[r, j] += a
    sol = milp(c, constraints=LinearConstraint(A.tocsr(), lo, hi), integrality=integ, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-6})
    sch = Schedule(ok=sol.success, status=sol.message, hours=T, units=[u.id for u in units], demand=demand,
                   reserve=[res] * T)
    if not sol.success:
        return sch
    x = sol.x
    for i, un in enumerate(units):
        sch.u[un.id] = [int(round(x[iu(i, t)])) for t in range(T)]
        sch.v[un.id] = [int(round(x[iv(i, t)])) for t in range(T)]
        sch.w[un.id] = [int(round(x[iw(i, t)])) for t in range(T)]
        sch.p[un.id] = [float(x[ip(i, t)]) * sch.u[un.id][t] for t in range(T)]
    sch.fuel_gj = float(sol.fun)
    return sch


def hourly_power_flow(net: Network, sch: Schedule, load: list[float], temps: list[float] | None = None) -> list[dict]:
    """Проверка каждого часа графика расчётом режима: выключенные агрегаты — gensOut, остальные — P из графика;
    балансирующий агрегат (slack) добирает фактические потери."""
    from .powerflow import run_power_flow
    slack = next(g["id"] for g in net.gens if g["slack"])
    out = []
    for t, L in enumerate(load):
        ctrl = net.default_ctrl()
        off = [g["id"] for g in net.gens if g["id"] in sch.u and not sch.u[g["id"]][t]]
        off += [g["id"] for g in net.gens if g.get("scenario") and g["id"] not in sch.u]
        ctrl["gP"] = {gid: sch.p[gid][t] for gid in sch.u if gid != slack}
        r = run_power_flow({"T": (temps or [25] * len(load))[t], "load": L, "ctrl": ctrl, "gensOut": off}, net)
        out.append({"hour": t, "status": r["status"], "slack_plan": sch.p.get(slack, [0] * len(load))[t],
                    "slack_pf": r["gen"][slack]["P"], "losses": r.get("losses"), "minV": r.get("minV"),
                    "maxV": max((r["V"][b] for b in r["live"]), default=None), "violations": r["violations"]})
    return out


if __name__ == "__main__":
    import sys
    net = load_network(ROOT / "data" / "network_real.yaml")
    for scen in (None, "2026"):
        for day in ("average", "peak"):
            units = units_from_network(net, scen)
            load = daily_load(day)
            s = solve(units, load)
            print(f"\n== сутки «{day}», состав {'2026 (с ГПЭС)' if scen else '2023'}: {s.status}; топливо {s.fuel_gj:,.0f} ГДж")
            if s.ok:
                for uid in s.units:
                    print(f"  {uid:12} " + " ".join(f"{p:5.0f}" if s.u[uid][t] else "    ." for t, p in enumerate(s.p[uid])))
    sys.exit(0)
