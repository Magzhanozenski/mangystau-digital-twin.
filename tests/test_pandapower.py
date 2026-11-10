"""Сверка с pandapower (шаг 1.3). Пропускается, если pandapower не установлен."""
import pytest

pytest.importorskip("pandapower")

from validation.compare_pandapower import TOL_P, TOL_V, compare  # noqa: E402


def test_matches_pandapower(net, js_data):
    n, worst, fails, errors = compare(js_data["cases"], net)
    assert n > 300, f"сравнено слишком мало режимов: {n}"
    assert not errors, errors[:5]
    assert worst["V"] < TOL_V and worst["P"] < TOL_P, (worst, fails[:5])
