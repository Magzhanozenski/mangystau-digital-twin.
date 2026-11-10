"""Данные сети в YAML совпадают с JS-ядром, и каждое значение помечено источник/допущение."""
import pytest


def test_every_item_has_status(net):
    for section in (net.buses, net.lines, net.gens):
        for item in section:
            assert item["status"] in ("источник", "допущение"), item["id"]
            assert item.get("source") or item.get("basis"), item["id"]
    for name, c in net.conductors.items():
        assert c["status"] in ("источник", "допущение"), name
    for key, val in net.model.items():
        assert val["status"] in ("источник", "допущение"), key


def test_shares_sum_to_one(net):
    assert sum(b["share"] for b in net.buses) == pytest.approx(1.0, abs=1e-12)


def test_yaml_matches_js(net, js_data):
    jb = {b["id"]: b for b in js_data["BUSES"]}
    for b in net.buses:
        assert b["share"] == pytest.approx(jb[b["id"]]["share"], abs=1e-12)
        assert b["name"] == jb[b["id"]]["name"]
    jl = {l["id"]: l for l in js_data["LINES"]}
    for l in net.lines:
        j = jl[l["id"]]
        assert (l["from"], l["to"], l["km"], l["conductor"]) == (j["f"], j["t"], j["km"], j["c"])
    jg = {g["id"]: g for g in js_data["GENS"]}
    for g in net.gens:
        j = jg[g["id"]]
        assert (g["bus"], g["p_max_mw"], g["q_min_mvar"], g["q_max_mvar"], g["v_pu"]) == \
               (j["bus"], j["Pmax"], j["Qmin"], j["Qmax"], j["V"])
    for name, c in net.conductors.items():
        j = js_data["COND"][name]
        assert (c["r_ohm_km"], c["x_ohm_km"], c["b_s_km"], c["i_rated_a"]) == (j["r"], j["x"], j["b"], j["I"])
