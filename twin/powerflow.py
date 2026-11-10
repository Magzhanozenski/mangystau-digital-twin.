"""Расчёт установившегося режима сети (Python-версия ядра core_js/engine2.js).

Ступени решения, как в JS-ядре:
  1) метод Ньютона–Рафсона с плоского старта;
  2) демпфированный Ньютон (ограничение шага);
  3) метод продолжения по нагрузке — отличает численную несходимость от физического
     предела передаваемой мощности (нос P–V кривой, проверка вырождения якобиана).
Пределы генераторов по Q (перевод PV → PQ), острова с местной генерацией, АЧР при дефиците.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import Network, load_network


@dataclass
class Gen:
    id: str
    bus: int
    name: str
    p_set: float
    v_set: float
    p_max: float
    q_min: float
    q_max: float
    slack: bool = False
    fixed_q: float | None = None


class Island:
    """Остров сети: узлы, матрица проводимостей, параметры линий."""

    def __init__(self, net: Network, ids: list[int], lines: list[dict], T: float):
        self.ids = ids
        self.idx = {b: i for i, b in enumerate(ids)}
        n = len(ids)
        self.Y = np.zeros((n, n), dtype=complex)
        self.lp = {}
        for l in lines:
            if l["from"] not in self.idx:
                continue
            R, X, B = net.line_pu(l, T)
            y = 1 / complex(R, X)
            i, j = self.idx[l["from"]], self.idx[l["to"]]
            self.Y[i, i] += y + 1j * B / 2
            self.Y[j, j] += y + 1j * B / 2
            self.Y[i, j] -= y
            self.Y[j, i] -= y
            self.lp[l["id"]] = (i, j, y, B)
        self.n = n


def _setup(isl: Island, gens: list[Gen], Pd, Qd, lam, base):
    n = isl.n
    typ = ["PQ"] * n
    Vs = np.ones(n)
    Pg = np.zeros(n)
    Qf = np.zeros(n)
    for g in gens:
        k = isl.idx[g.bus]
        if g.slack:
            typ[k] = "SL"
            Vs[k] = g.v_set
            continue
        Pg[k] += g.p_set / base
        if g.fixed_q is None:
            if typ[k] != "SL":
                typ[k] = "PV"
                Vs[k] = g.v_set
        else:
            Qf[k] += g.fixed_q / base
    pvq = [i for i in range(n) if typ[i] != "SL"]
    pq = [i for i in range(n) if typ[i] == "PQ"]
    return typ, Vs, pvq, pq, Pg - lam * Pd, Qf - lam * Qd


def _jacobian(Y, V, pvq, pq):
    """Якобиан в полярных координатах (формулы dS/dθ, dS/d|V|)."""
    Vn = V / np.abs(V)
    Ib = Y @ V
    dS_dVm = np.diag(V) @ np.conj(Y @ np.diag(Vn)) + np.diag(np.conj(Ib)) @ np.diag(Vn)
    dS_dVa = 1j * np.diag(V) @ np.conj(np.diag(Ib) - Y @ np.diag(V))
    J11 = dS_dVa[np.ix_(pvq, pvq)].real
    J12 = dS_dVm[np.ix_(pvq, pq)].real
    J21 = dS_dVa[np.ix_(pq, pvq)].imag
    J22 = dS_dVm[np.ix_(pq, pq)].imag
    return np.block([[J11, J12], [J21, J22]])


def _nr(isl, setup, V0, th0, damped, max_it):
    typ, Vs, pvq, pq, Psp, Qsp = setup
    Vm = np.array([V0[i] if typ[i] == "PQ" else Vs[i] for i in range(isl.n)], dtype=float)
    th = np.array(th0, dtype=float)
    npv = len(pvq)
    for it in range(max_it):
        V = Vm * np.exp(1j * th)
        S = V * np.conj(isl.Y @ V)
        mis = np.concatenate([Psp[pvq] - S.real[pvq], Qsp[pq] - S.imag[pq]])
        mx = np.max(np.abs(mis)) if mis.size else 0.0
        if mx < 1e-8:
            return {"ok": True, "V": Vm, "th": th, "P": S.real, "Q": S.imag, "it": it, "setup": setup}
        if not np.isfinite(mx) or mx > 1e3:
            return {"ok": False}
        J = _jacobian(isl.Y, V, pvq, pq)
        try:
            dx = np.linalg.solve(J, mis)
        except np.linalg.LinAlgError:
            return {"ok": False}
        a = 1.0
        if damped:
            mth = np.max(np.abs(dx[:npv])) if npv else 0.0
            mv = np.max(np.abs(dx[npv:])) if len(pq) else 0.0
            a = min(1.0, 0.3 / max(mth, 1e-9), 0.08 / max(mv, 1e-9))
        th[pvq] += a * dx[:npv]
        Vm[pq] += a * dx[npv:]
        if np.any(Vm < 0.2) or np.any(Vm > 2):
            return {"ok": False}
    return {"ok": False}


def _solve_q(isl, gens, Pd, Qd, lam, V0, th0, damped, base):
    """Решение с ограничениями генераторов по Q: генератор на пределе становится PQ."""
    for _ in range(5):
        setup = _setup(isl, gens, Pd, Qd, lam, base)
        sol = _nr(isl, setup, V0, th0, damped, 80 if damped else 30)
        if not sol["ok"]:
            return {"ok": False}
        changed = False
        for g in gens:
            if g.slack or g.fixed_q is not None:
                continue
            k = isl.idx[g.bus]
            qg = (sol["Q"][k] + lam * Qd[k]) * base
            if qg > g.q_max + 1e-6:
                g.fixed_q, changed = g.q_max, True
            elif qg < g.q_min - 1e-6:
                g.fixed_q, changed = g.q_min, True
        if not changed:
            return sol
        V0, th0 = sol["V"], sol["th"]
    return {"ok": False}


def _sigma_min(isl, sol):
    _, _, pvq, pq, _, _ = sol["setup"]
    V = sol["V"] * np.exp(1j * sol["th"])
    J = _jacobian(isl.Y, V, pvq, pq)
    return float(np.linalg.svd(J, compute_uv=False)[-1]) if J.size else 1.0


def _continuation(isl, gens, Pd, Qd, base):
    n = isl.n
    for g in gens:
        g.fixed_q = None
    lam = 0.25
    sol = _solve_q(isl, gens, Pd, Qd, lam, np.ones(n), np.zeros(n), True, base)
    if not sol["ok"]:
        return {"status": "numerical"}
    sig0 = _sigma_min(isl, sol)
    curve = [(lam, float(sol["V"].min()))]
    step = 0.2
    while lam < 1 - 1e-9:
        try_lam = min(1.0, lam + step)
        saved = [g.fixed_q for g in gens]
        s2 = _solve_q(isl, gens, Pd, Qd, try_lam, sol["V"], sol["th"], True, base)
        if s2["ok"]:
            lam, sol = try_lam, s2
            curve.append((lam, float(sol["V"].min())))
        else:
            for g, q in zip(gens, saved):
                g.fixed_q = q
            step /= 2
            if step < 0.003:
                break
    if lam >= 1 - 1e-9:
        return {"status": "ok", "sol": sol, "curve": curve}
    ratio = _sigma_min(isl, sol) / sig0
    return {"status": "pv_limit", "sol": sol, "curve": curve, "lambda_max": lam,
            "sigma_ratio": ratio, "confirmed": ratio < 0.25}


def run_power_flow(scn: dict, net: Network | None = None) -> dict:
    """Расчёт режима. scn: T, load, linesOut, gensOut, ctrl (как в JS), ratingA (необязательно)."""
    net = net or load_network()
    base, kv = net.base_mva, net.base_kv
    T = scn.get("T", 25)
    ctrl = scn.get("ctrl") or net.default_ctrl()
    lines_out = set(scn.get("linesOut", []))
    gens_out = set(scn.get("gensOut", []))
    rating = scn.get("ratingA") or {}
    lim = net.model["voltage_limits_pu"]
    achr_res = net.model["achr_reserve"]["value"]

    line_in = [l for l in net.lines if l["id"] not in lines_out]
    adj = {b["id"]: [] for b in net.buses}
    for l in line_in:
        adj[l["from"]].append(l["to"])
        adj[l["to"]].append(l["from"])
    seen, comps = set(), []
    for b in net.buses:
        if b["id"] in seen:
            continue
        comp, q = [], [b["id"]]
        seen.add(b["id"])
        while q:
            u = q.pop(0)
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(sorted(comp))

    tan = net.tan_phi(T)
    load_bus, shed_mw = {}, 0.0
    for b in net.buses:
        p0 = scn["load"] * b["share"]
        sh = float(ctrl.get("shed", {}).get(str(b["id"]), ctrl.get("shed", {}).get(b["id"], 0)) or 0)
        shed_mw += p0 * sh
        load_bus[b["id"]] = {"P": p0 * (1 - sh), "P0": p0}

    def bsk(bid):
        d = ctrl.get("bsk", {})
        return float(d.get(str(bid), d.get(bid, 0)) or 0)

    res = {"V": {}, "ang": {}, "gen": {}, "flows": {}, "islands": [], "live": [], "unserved": 0.0,
           "achr": 0.0, "shedMW": shed_mw, "loadBus": load_bus, "violations": []}
    worst, lam_max, pv = "ok", None, None

    for ids in comps:
        gens = [Gen(g["id"], g["bus"], g["name"],
                    0.0 if g["slack"] else min(g["p_max_mw"], ctrl["gP"].get(g["id"], g["p_mw"])),
                    ctrl["vset"].get(g["id"], g["v_pu"]), g["p_max_mw"], g["q_min_mvar"], g["q_max_mvar"])
                for g in net.gens if g["id"] not in gens_out and g["bus"] in ids]
        isl_load = sum(load_bus[i]["P"] for i in ids)
        if not gens:
            if isl_load > 0.01:
                res["unserved"] += isl_load
                res["violations"].append(("unserved", ids, isl_load))
            for i in ids:
                res["V"][i] = 0.0
                res["ang"][i] = 0.0
            continue
        slack = next((g for g in gens if g.id == "G1"), None) or max(gens, key=lambda g: g.p_max)
        slack.slack = True
        main = slack.id == "G1"
        cap = sum(g.p_max for g in gens)
        frac = 0.0
        if not main and isl_load * 1.03 > cap:
            frac = min(1.0, 1 - cap * achr_res / isl_load)
            res["achr"] += isl_load * frac
            res["violations"].append(("achr", ids, isl_load * frac))
        isl = Island(net, ids, line_in, T)
        Pd = np.zeros(isl.n)
        Qd = np.zeros(isl.n)
        for i in ids:
            k = isl.idx[i]
            p = load_bus[i]["P"] * (1 - frac)
            Pd[k], Qd[k] = p / base, p * tan / base
            isl.Y[k, k] += 1j * bsk(i) / base
        info = {"ids": ids, "slack": slack.id, "main": main, "achrMW": isl_load * frac}
        res["islands"].append(info)

        flatV, flatT = np.ones(isl.n), np.zeros(isl.n)
        sol, how = _solve_q(isl, gens, Pd, Qd, 1.0, flatV, flatT, False, base), "nr"
        if not sol["ok"]:
            for g in gens:
                g.fixed_q = None
            sol, how = _solve_q(isl, gens, Pd, Qd, 1.0, flatV, flatT, True, base), "damped"
        lam_used = 1.0
        if not sol["ok"]:
            cr = _continuation(isl, gens, Pd, Qd, base)
            if cr["status"] == "ok":
                sol, how = cr["sol"], "continuation"
            elif cr["status"] == "pv_limit":
                worst = "pv_limit"
                lam_max = cr["lambda_max"] if lam_max is None else min(lam_max, cr["lambda_max"])
                pv = cr
                res["violations"].append(("pv_limit", ids, cr["lambda_max"]))
                sol, how, lam_used = cr["sol"], "pv", cr["lambda_max"]
            else:
                if worst != "pv_limit":
                    worst = "numerical"
                info["method"] = "numerical"
                for i in ids:
                    res["V"][i] = None
                continue
        info["method"] = how
        for i in ids:
            k = isl.idx[i]
            res["V"][i] = float(sol["V"][k])
            res["ang"][i] = float(np.degrees(sol["th"][k]))
            res["live"].append(i)
        for g in gens:
            k = isl.idx[g.bus]
            others = sum(h.p_set / base for h in gens if h.bus == g.bus and h is not g)
            P = (sol["P"][k] + lam_used * Pd[k] - others) * base if g.slack else g.p_set
            Q = g.fixed_q if g.fixed_q is not None else (sol["Q"][k] + lam_used * Qd[k]) * base
            res["gen"][g.id] = {"P": float(P), "Q": float(Q), "Pmax": g.p_max, "atQlim": g.fixed_q is not None, "slack": g.slack}
        Vc = sol["V"] * np.exp(1j * sol["th"])
        for lid, (i, j, y, B) in isl.lp.items():
            line = net.line(lid)
            imax = rating.get(lid) or net.static_rating_a(line, T)
            Iij = (Vc[i] - Vc[j]) * y + Vc[i] * 1j * B / 2
            Iji = (Vc[j] - Vc[i]) * y + Vc[j] * 1j * B / 2
            Sij = Vc[i] * np.conj(Iij) * base
            Sji = Vc[j] * np.conj(Iji) * base
            Ia = abs(Sij) / (np.sqrt(3) * kv * abs(Vc[i])) * 1000
            Ib = abs(Sji) / (np.sqrt(3) * kv * abs(Vc[j])) * 1000
            I = max(Ia, Ib)
            res["flows"][lid] = {"P": float(Sij.real), "Q": float(Sij.imag), "I": float(I), "Imax": imax,
                                 "loading": float(I / imax), "loss": float((Sij + Sji).real)}
    for l in net.lines:
        if l["id"] not in res["flows"]:
            res["flows"][l["id"]] = {"out": l["id"] in lines_out, "dead": l["id"] not in lines_out,
                                     "P": 0.0, "Q": 0.0, "I": 0.0, "loading": 0.0, "loss": 0.0,
                                     "Imax": rating.get(l["id"]) or net.static_rating_a(l, T)}
    for g in net.gens:
        res["gen"].setdefault(g["id"], {"P": 0.0, "Q": 0.0, "Pmax": g["p_max_mw"], "out": True})

    res.update(status=worst, lambdaMax=lam_max, pv=pv, converged=worst == "ok",
               losses=sum(f.get("loss", 0) for f in res["flows"].values()))
    if worst == "ok":
        for i in res["live"]:
            v = res["V"][i]
            if v < lim["crit_low"] or v > lim["crit_high"]:
                res["violations"].append(("voltage", i, v))
        for lid, f in res["flows"].items():
            if not f.get("out") and not f.get("dead") and f["loading"] > 1:
                res["violations"].append(("overload", lid, f["loading"]))
        for gid, g in res["gen"].items():
            if not g.get("out") and g.get("slack") and g["P"] > g["Pmax"] + 0.5:
                res["violations"].append(("pmax", gid, g["P"]))
        res["minV"] = min(res["V"][i] for i in res["live"])
        res["maxLoading"] = max([0.0] + [f.get("loading", 0) for f in res["flows"].values()])
    res["critical"] = len(res["violations"]) > 0
    return res
