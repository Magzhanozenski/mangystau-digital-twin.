"""DLR по IEEE 738 (шаг 3.3). Без сети."""
import numpy as np
import pytest
from scipy.optimize import brentq

from twin import dlr as L


def test_pue_table_consistent():
    """Табличные токи ПУЭ обоих проводов — при одном эквивалентном ветре (±5 %)."""
    eq = [brentq(lambda v: float(L.ampacity(c, 25, v, 90, 0)) - p["I_pue"], 0.001, 10) for c, p in L.CONDUCTORS.items()]
    assert (max(eq) - min(eq)) / np.mean(eq) < 0.05


def test_pue_temperature_correction_reproduced():
    c = "АС-185"
    v = brentq(lambda w: float(L.ampacity(c, 25, w, 90, 0)) - 510, 0.001, 10)
    for t, k in L.PUE_CORR.items():
        assert float(L.ampacity(c, t, v, 90, 0) / L.ampacity(c, 25, v, 90, 0)) == pytest.approx(k, abs=0.02)


def test_k_angle_and_wind_angle():
    assert float(L.k_angle(90.0)) == pytest.approx(1.0, abs=1e-3)
    assert float(L.k_angle(0.0)) == pytest.approx(0.388, abs=1e-3)
    assert list(L.wind_angle([0, 90, 135, 180, 270, 315], 0)) == [0, 90, 45, 0, 90, 45]


def test_physics():
    c = "АС-300"
    assert L.ampacity(c, 40, 1, 90, 0) < L.ampacity(c, 20, 1, 90, 0)
    assert L.ampacity(c, 30, 4, 90, 0) > L.ampacity(c, 30, 1, 90, 0)
    assert L.ampacity(c, 30, 2, 90, 900) < L.ampacity(c, 30, 2, 90, 0)
    assert L.ampacity(c, 30, 3, 90, 0) > L.ampacity(c, 30, 3, 10, 0)


def test_weakest_span():
    """Линия из двух пролётов: ветер вдоль одного — ток линии равен току этого пролёта."""
    az = np.array([0.0, 90.0])
    i = L.line_rating("АС-185", az, [30], [3], [0.0], [0])
    assert float(i[0]) == pytest.approx(float(L.ampacity("АС-185", 30, 3, 0.0, 0)))


def test_span_azimuths():
    az, ln = L.span_azimuths([(45.0, 50.0), (45.0, 51.0), (46.0, 51.0)])
    assert az[0] == pytest.approx(90, abs=0.5) and az[1] == pytest.approx(0, abs=1e-6)
    assert ln[1] == pytest.approx(111.2, abs=0.2)


def test_wind_mapping_monotone_and_nonnegative():
    qe = np.linspace(1, 10, 99)
    qs = np.linspace(0, 8, 99)
    m = L.map_wind([0.0, 1.0, 5.0, 12.0], qe, qs)
    assert (m >= 0).all() and np.all(np.diff(m) >= 0)
    assert m[-1] == pytest.approx(12 - 2)  # за пределами — сдвиг на крайнюю разницу


def test_static_pue_interpolation():
    assert float(L.static_pue("АС-185", 25)) == pytest.approx(510)
    assert float(L.static_pue("АС-185", 42.5)) == pytest.approx(510 * (0.81 + 0.74) / 2)
