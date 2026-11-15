"""Суточный график агрегатов (шаг 3.1). Без сети; сверка MILP с полным перебором и независимой проверкой ограничений."""
import random
import sys
from pathlib import Path

import pytest

from twin import dispatch as D
from twin import load_network

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "validation"))

import check_dispatch as C  # noqa: E402


@pytest.fixture(scope="module")
def net():
    return load_network(ROOT / "data" / "network_real.yaml")


def _u(uid, **kw):
    base = dict(p_min=20, p_max=100, ramp=50, start_h=0, min_up=1, min_down=1, heat_rate=10, startup_gj=0)
    base.update(kw)
    return D.Unit(uid, **base)


def test_milp_matches_brute_force():
    rng = random.Random(7)
    units, load = C.random_case(rng)
    sch = D.solve(units, load, reserve_mw=10, loss=0, cyclic=False)
    bf = C.brute_force(units, load, 10, 0)
    assert sch.ok and bf is not None
    assert sch.fuel_gj == pytest.approx(bf, rel=1e-9)
    assert max(C.check_constraints(units, sch, cyclic=False).values()) < 1e-6


def test_min_load_forces_shutdown_or_infeasible():
    """Два агрегата по p_min 60 не могут вместе покрыть 80 МВт — один должен быть выключен."""
    units = [_u("A", p_min=60), _u("B", p_min=60, heat_rate=11)]
    sch = D.solve(units, [80] * 4, reserve_mw=0, loss=0, cyclic=True)
    assert sch.ok
    assert all(sch.u["A"][t] + sch.u["B"][t] == 1 for t in range(4))


def test_ramp_limits_are_respected():
    units = [_u("A", ramp=10, heat_rate=9), _u("B", heat_rate=12)]
    load = [50, 90, 90, 50]
    sch = D.solve(units, load, reserve_mw=0, loss=0, cyclic=True)
    assert sch.ok
    for t in range(4):
        assert abs(sch.p["A"][t] - sch.p["A"][t - 1]) <= 10 + 1e-6
    assert max(C.check_constraints(units, sch).values()) < 1e-6


def test_min_up_down_and_start_time():
    units = [_u("A", min_up=3, startup_gj=0, heat_rate=9, on0=False, hours0=99, start_h=2),
             _u("B", heat_rate=15, on0=True, p0=50, hours0=99)]
    sch = D.solve(units, [50, 50, 50, 50, 50, 50], reserve_mw=0, loss=0, cyclic=False)
    assert sch.ok
    assert sch.u["A"][0] == sch.u["A"][1] == 0          # пуск не раньше 2-го часа
    assert sch.u["A"][2] == 1                            # дешёвый агрегат включается сразу, как может
    assert max(C.check_constraints(units, sch, cyclic=False).values()) < 1e-6


def test_infeasible_reported():
    sch = D.solve([_u("A", p_max=50)], [80, 80], reserve_mw=0, loss=0)
    assert not sch.ok


def test_network_schedule_2023_and_2026(net):
    u23 = D.units_from_network(net, None)
    u26 = D.units_from_network(net, "2026")
    assert not any(u.id.startswith("GPES") for u in u23) and sum(u.id.startswith("GPES") for u in u26) == 6
    assert not any(u.id == "ATYRAU" for u in u26)  # эквивалент с фиксированной P в графике не участвует
    load = D.daily_load("average")
    assert sum(load) / 24 == pytest.approx(392.1, abs=0.5)
    s23, s26 = D.solve(u23, load), D.solve(u26, load)
    assert s23.ok and s26.ok
    for units, s in ((u23, s23), (u26, s26)):
        assert max(C.check_constraints(units, s).values()) < 1e-6
        for un in units:
            if un.must_run:
                assert all(s.u[un.id])
    assert s26.fuel_gj < s23.fuel_gj  # ГПЭС экономичнее блоков МАЭК (по принятым удельным расходам)


def test_hourly_power_flow_converges(net):
    units = D.units_from_network(net, None)
    load = D.daily_load("peak")
    sch = D.solve(units, load)
    pf = D.hourly_power_flow(net, sch, load)
    assert all(h["status"] == "ok" for h in pf)
    assert max(abs(h["slack_pf"] - h["slack_plan"]) for h in pf) < 10
