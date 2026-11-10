"""Построение той же модели сети в pandapower — для сверки с собственным ядром.

pandapower — открытая библиотека расчёта энергосетей (https://www.pandapower.org).
Если она не установлена, модуль импортируется, но функции бросают ImportError с подсказкой.
"""
from __future__ import annotations

import math

from .network import Network, load_network

try:
    import pandapower as pp
except ImportError:  # pragma: no cover
    pp = None


def _require():
    if pp is None:
        raise ImportError("pandapower не установлен: pip install pandapower")


def build_pp(scn: dict, net: Network | None = None, islands: list | None = None):
    """Модель pandapower для сценария. islands — список островов из run_power_flow
    (какой генератор балансирующий и доля АЧР), чтобы сравнивать одинаковые постановки."""
    _require()
    net = net or load_network()
    T = scn.get("T", 25)
    ctrl = scn.get("ctrl") or net.default_ctrl()
    lines_out = set(scn.get("linesOut", []))
    gens_out = set(scn.get("gensOut", []))
    tan = net.tan_phi(T)
    rf = net.resistance_factor(T)

    m = pp.create_empty_network(sn_mva=net.base_mva, f_hz=net.freq)
    bus = {b["id"]: pp.create_bus(m, vn_kv=net.base_kv, name=b["name"]) for b in net.buses}

    for l in net.lines:
        c = net.conductors[l["conductor"]]
        pp.create_line_from_parameters(
            m, bus[l["from"]], bus[l["to"]], length_km=l["km"],
            r_ohm_per_km=c["r_ohm_km"] * rf, x_ohm_per_km=c["x_ohm_km"],
            c_nf_per_km=c["b_s_km"] / (2 * math.pi * net.freq) * 1e9,
            max_i_ka=net.static_rating_a(l, T) / 1000, name=l["id"],
            in_service=l["id"] not in lines_out,
        )

    achr = {}
    slack_of = {}
    for isl in islands or []:
        slack_of[isl["slack"]] = True
        load_sum = sum(scn["load"] * net.bus(i)["share"] for i in isl["ids"])
        for i in isl["ids"]:
            achr[i] = isl["achrMW"] / load_sum if load_sum > 0 else 0.0

    shed = ctrl.get("shed", {})
    bsk = ctrl.get("bsk", {})
    for b in net.buses:
        sh = float(shed.get(str(b["id"]), shed.get(b["id"], 0)) or 0)
        p = scn["load"] * b["share"] * (1 - sh) * (1 - achr.get(b["id"], 0.0))
        if p > 0:
            pp.create_load(m, bus[b["id"]], p_mw=p, q_mvar=p * tan)
        q = float(bsk.get(str(b["id"]), bsk.get(b["id"], 0)) or 0)
        if q:
            # в pandapower q_mvar > 0 — потребление, поэтому батарея конденсаторов задаётся отрицательной
            pp.create_shunt(m, bus[b["id"]], q_mvar=-q, p_mw=0.0)

    for g in net.gens:
        if g["id"] in gens_out:
            continue
        vs = ctrl["vset"].get(g["id"], g["v_pu"])
        if g["slack"]:
            pp.create_ext_grid(m, bus[g["bus"]], vm_pu=vs, name=g["id"],
                               min_q_mvar=g["q_min_mvar"], max_q_mvar=g["q_max_mvar"])
        else:
            pset = min(g["p_max_mw"], ctrl["gP"].get(g["id"], g["p_mw"]))
            pp.create_gen(m, bus[g["bus"]], p_mw=pset, vm_pu=vs, name=g["id"],
                          min_q_mvar=g["q_min_mvar"], max_q_mvar=g["q_max_mvar"],
                          max_p_mw=g["p_max_mw"], slack=g["id"] in slack_of)
    return m


def run_pp(scn: dict, net: Network | None = None, islands: list | None = None) -> dict:
    """Расчёт режима в pandapower. Возвращает словарь в формате run_power_flow (V, ang, flows, gen)."""
    _require()
    net = net or load_network()
    m = build_pp(scn, net, islands)
    pp.runpp(m, algorithm="nr", init="flat", enforce_q_lims=True, tolerance_mva=1e-9, max_iteration=50)
    out = {"V": {}, "ang": {}, "flows": {}, "gen": {}}
    for b, (vm, va) in zip(net.buses, m.res_bus[["vm_pu", "va_degree"]].itertuples(index=False)):
        out["V"][b["id"]] = float(vm) if vm == vm else 0.0
        out["ang"][b["id"]] = float(va) if va == va else 0.0
    for idx, l in enumerate(net.lines):
        r = m.res_line.iloc[idx]
        out["flows"][l["id"]] = {"P": float(r.p_from_mw), "Q": float(r.q_from_mvar),
                                 "I": float(max(r.i_from_ka, r.i_to_ka) * 1000), "loss": float(r.pl_mw)}
    for idx, row in m.gen.iterrows():
        r = m.res_gen.loc[idx]
        out["gen"][row["name"]] = {"P": float(r.p_mw), "Q": float(r.q_mvar)}
    for idx, row in m.ext_grid.iterrows():
        r = m.res_ext_grid.loc[idx]
        out["gen"][row["name"]] = {"P": float(r.p_mw), "Q": float(r.q_mvar)}
    return out
