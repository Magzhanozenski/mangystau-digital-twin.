"""Трансформаторы и разные классы напряжения в ядре (шаг 2.4, пункт 2). Сверка с pandapower на малых схемах."""
import copy

import pytest
import yaml

from twin import load_network, run_power_flow
from twin.network import DEFAULT_PATH

pp_model = pytest.importorskip("twin.pp_model")
pytest.importorskip("pandapower")

BASE = yaml.safe_load(DEFAULT_PATH.read_text(encoding="utf-8"))
M = "допущение"


def _net(tmp_path, trafos, buses, lines, gens):
    raw = {"meta": BASE["meta"], "conductors": copy.deepcopy(BASE["conductors"]), "model": BASE["model"],
           "buses": buses, "lines": lines, "generators": gens, "transformers": trafos}
    raw["conductors"]["АС-300"] = {"r_ohm_km": 0.098, "x_ohm_km": 0.429, "b_s_km": 2.64e-6, "i_rated_a": 690,
                                   "diameter_m": 0.024, "status": M, "basis": "тест"}
    p = tmp_path / "net.yaml"
    p.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_network(p)


def _b(i, kv, share):
    return {"id": i, "name": f"B{i}", "short": f"B{i}", "vn_kv": kv, "share": share, "status": M, "basis": "тест"}


def _t(tid, a, b, sn, uk, pk, p0, i0):
    return {"id": tid, "from": a, "to": b, "sn_mva": sn, "uk_pct": uk, "pk_kw": pk, "p0_kw": p0, "i0_pct": i0,
            "status": M, "basis": "тест"}


def _g(gid, bus, slack, p=0.0, pmax=500):
    return {"id": gid, "bus": bus, "name": gid, "slack": slack, "p_mw": None if slack else p, "p_max_mw": pmax,
            "q_min_mvar": -300, "q_max_mvar": 300, "v_pu": 1.02, "ramp_mw_h": 100, "status": M, "basis": "тест"}


@pytest.fixture
def net(tmp_path):
    buses = [_b(1, 220, 0.0), _b(2, 220, 0.0), _b(3, 110, 0.3), _b(4, 110, 0.3), _b(5, 10, 0.4)]
    lines = [{"id": "L220", "from": 1, "to": 2, "km": 150, "conductor": "АС-300", "name": "220", "status": M, "basis": "т"},
             {"id": "L110", "from": 3, "to": 4, "km": 40, "conductor": "АС-185", "name": "110", "status": M, "basis": "т"}]
    trafos = [_t("AT1", 2, 3, 125, 11.0, 315, 65, 0.40), _t("AT2", 2, 3, 63, 11.0, 200, 37, 0.45),
              _t("T1", 4, 5, 25, 10.5, 120, 25, 0.45)]
    return _net(tmp_path, trafos, buses, lines, [_g("G1", 1, True)])


def _cmp(r, p):
    dv = max(abs(p["V"][b] - r["V"][b]) for b in r["live"])
    dp = max(abs(p["flows"][k]["P"] - f["P"]) for k, f in r["flows"].items() if not f.get("out") and not f.get("dead"))
    return dv, dp


@pytest.mark.parametrize("scn", [{"T": 25, "load": 60}, {"T": 40, "load": 50}, {"T": 25, "load": 60, "linesOut": ["AT2"]}])
def test_core_matches_pandapower_with_trafos(net, scn):
    r = run_power_flow(scn, net)
    assert r["status"] == "ok"
    p = pp_model.run_pp(scn, net, r["islands"])
    dv, dp = _cmp(r, p)
    assert dv < 1e-6 and dp < 1e-4, (dv, dp)
    for tid in ("AT1", "T1"):
        assert r["flows"][tid]["loading"] == pytest.approx(p["flows"][tid]["loading"], abs=1e-5)


def test_trafo_losses_and_parallel_split(net):
    r = run_power_flow({"T": 25, "load": 60}, net)
    at1, at2 = r["flows"]["AT1"], r["flows"]["AT2"]
    assert at1["loss"] > 0.065 * 0.99  # не меньше потерь холостого хода
    assert at1["P"] > at2["P"] > 0     # одинаковое uk — делят нагрузку по мощности
    assert at1["P"] / at2["P"] == pytest.approx(125 / 63, rel=0.05)


def test_line_across_voltage_levels_rejected(tmp_path):
    with pytest.raises(ValueError):
        _net(tmp_path, [], [_b(1, 220, 0), _b(2, 110, 1)],
             [{"id": "X", "from": 1, "to": 2, "km": 1, "conductor": "АС-185", "name": "x", "status": M, "basis": "т"}],
             [_g("G1", 1, True)])


def test_base_network_unchanged():
    """network.yaml без трансформаторов и vn_kv — поведение прежнее (сверка с JS — в test_network_data)."""
    n = load_network()
    assert n.trafos == [] and all(n.bus_kv(b["id"]) == 110 for b in n.buses)
