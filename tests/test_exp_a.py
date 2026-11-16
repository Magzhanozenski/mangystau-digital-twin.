"""Эксперимент A: интерполяция токов, нагрузка, порядок осторожности методов. Без сети."""
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import exp_a_load  # noqa: E402
import exp_a_ratings as A  # noqa: E402


def test_interp_exact_on_linear():
    levels = np.array([100.0, 200.0, 300.0])
    grid = np.array([[10.0, 1.0], [20.0, 3.0], [30.0, 5.0]])
    out = A.interp_hours(grid, levels, np.array([150.0, 250.0, 300.0]))
    assert out == pytest.approx(np.array([[15.0, 2.0], [25.0, 4.0], [30.0, 5.0]]))


def test_hourly_load_matches_facts():
    d = exp_a_load.hourly_load(2025)
    assert len(d) == 8760
    assert d["load_mw"].mean() == pytest.approx(392.1, abs=0.1)
    assert d["load_mw"].max() == pytest.approx(546.0, abs=0.1)
    assert d.attrs["b"] > 0  # нагрузка растёт в жару и холод


@pytest.fixture(scope="module")
def rat():
    raw = yaml.safe_load((ROOT / "data" / "network_real.yaml").read_text(encoding="utf-8"))
    return A.ratings(raw)


def test_conservative_dlr_not_above_median(rat):
    """M4 (ветер P10, T и солнце P90) почти всегда не выше M3 (P50)."""
    r, _, _ = rat
    share = np.mean(np.concatenate([r["M4"][k] <= r["M3"][k] + 1e-6 for k in r["M3"]]))
    assert share > 0.97


def test_static_methods(rat):
    r, info, _ = rat
    for k, v in info.items():
        assert np.all(r["M1"][k] == {"АС-185": 510.0, "АС-300": 710.0}[v["conductor"]])
        assert np.all(r["M2"][k] > 0)


def test_results_file_consistent():
    p = ROOT / "experiments" / "exp_a_results.json"
    if not p.exists():
        pytest.skip("нет результатов эксперимента")
    import json
    d = json.loads(p.read_text(encoding="utf-8"))
    for run in d["runs"]:
        for m, h in run["hour"].items():
            assert h["TP"] + h["FN"] == run["hours_true"]
            assert sum(h.values()) == 8760
