"""Проверки Python-ядра расчёта режима."""
import numpy as np
import pytest

from twin import run_power_flow
from twin.powerflow import Island, _jacobian


def test_jacobian_matches_finite_differences(net):
    isl = Island(net, [b["id"] for b in net.buses], net.lines, 35)
    n = isl.n
    Vm = 1 - 0.02 * np.arange(n)
    th = -0.03 * np.arange(n)
    pvq, pq = list(range(1, n)), [1, 2, 3, 6, 7]

    def f(Vm, th):
        V = Vm * np.exp(1j * th)
        S = V * np.conj(isl.Y @ V)
        return np.concatenate([S.real[pvq], S.imag[pq]])

    J = _jacobian(isl.Y, Vm * np.exp(1j * th), pvq, pq)
    h = 1e-7
    f0 = f(Vm, th)
    Jn = np.zeros_like(J)
    for c, i in enumerate(pvq):
        t2 = th.copy(); t2[i] += h
        Jn[:, c] = (f(Vm, t2) - f0) / h
    for c, i in enumerate(pq):
        v2 = Vm.copy(); v2[i] += h
        Jn[:, len(pvq) + c] = (f(v2, th) - f0) / h
    assert np.max(np.abs(J - Jn)) < 1e-4 * np.max(np.abs(J))


def test_power_balance(net):
    r = run_power_flow({"T": 35, "load": 180}, net)
    assert r["status"] == "ok"
    gen = sum(g["P"] for g in r["gen"].values() if not g.get("out"))
    load = sum(r["loadBus"][i]["P"] for i in r["live"])
    assert gen - load - r["losses"] == pytest.approx(0, abs=1e-6)


def test_q_limits_respected(net):
    r = run_power_flow({"T": 45, "load": 230}, net)
    for gid, g in r["gen"].items():
        lim = net.gen(gid)
        if g.get("out") or g.get("slack"):
            continue
        assert lim["q_min_mvar"] - 1e-6 <= g["Q"] <= lim["q_max_mvar"] + 1e-6


def test_island_with_local_generation(net):
    # без L7 и L9 Жанаозен отделяется от МАЭК и питается от ГТЭС Жанаозен
    r = run_power_flow({"T": 30, "load": 170, "linesOut": ["L7", "L9"]}, net)
    isl = [i for i in r["islands"] if not i["main"]]
    assert len(isl) == 1 and isl[0]["ids"] == [6] and isl[0]["slack"] == "G2"
    assert r["unserved"] == 0 and r["status"] == "ok"


def test_unserved_without_generation(net):
    r = run_power_flow({"T": 30, "load": 170, "linesOut": ["L9", "L10"]}, net)
    assert r["unserved"] == pytest.approx(170 * net.bus(8)["share"])
    assert r["critical"]


def test_pv_limit_detected(net):
    r = run_power_flow({"T": 40, "load": 200, "linesOut": ["L4", "L11"]}, net)
    assert r["status"] == "pv_limit"
    assert 0.5 < r["lambdaMax"] < 1
    assert r["pv"]["confirmed"]


def test_matches_js_core(net, js_data):
    """Python-ядро совпадает с JS-ядром на всех сценариях (статус, λmax, напряжения, потоки)."""
    worst_v = worst_p = 0.0
    for c in js_data["cases"]:
        r = run_power_flow(c, net)
        assert r["status"] == c["status"]
        if c["status"] == "pv_limit":
            assert r["lambdaMax"] == pytest.approx(c["lambdaMax"], abs=1e-9)
            continue
        worst_v = max(worst_v, max(abs(v - c["V"][str(b)]) for b, v in r["V"].items()))
        worst_p = max(worst_p, max(abs(f["P"] - c["flows"][l]["P"]) for l, f in r["flows"].items()))
    assert worst_v < 1e-9 and worst_p < 1e-7
