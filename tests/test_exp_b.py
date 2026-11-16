"""Эксперимент B: перенос моделей, данные, тест Диболда — Мариано. Без сети."""
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import exp_b_data as BD  # noqa: E402
import exp_b_models as B  # noqa: E402
import exp_b_run as R  # noqa: E402


def test_mulberry32_known_values():
    """Значения mulberry32 совпадают с эталонной JS-реализацией."""
    if not shutil.which("node"):
        pytest.skip("нет Node.js")
    js = ("function mulberry32(a){return function(){a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);"
          "t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}"
          "const r=mulberry32(42);console.log(JSON.stringify([r(),r(),r(),r(),r()]));")
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True).stdout
    g = B.mulberry32(42)
    assert [g() for _ in range(5)] == pytest.approx(eval(out), abs=0)


def test_linear_recovers_exact_model():
    rng = np.random.default_rng(0)
    n = 2000
    X = {"h": rng.integers(0, 24, n), "T": rng.uniform(-10, 40, n), "Tlag": rng.uniform(-10, 40, n),
         "Tmax": rng.uniform(-5, 45, n), "weekend": rng.random(n) < 0.3}
    w = rng.normal(size=B.l1_features(**X).shape[1])
    y = B.l1_features(**X) @ w
    m = B.Linear(B.l1_features).fit(X, y)
    assert np.max(np.abs(m.predict(X) - y)) < 1e-3


def test_dm_test_sign_and_null():
    rng = np.random.default_rng(1)
    e = rng.normal(size=5000)
    assert R.dm_test(e, e * 1.0) == (0.0, 1.0)
    stat, p = R.dm_test(e * 0.5, e)   # первая модель лучше → статистика < 0, p мало
    assert stat < 0 and p < 1e-6


def test_data_shapes_and_truth():
    d = BD.build()
    assert set(d["year"].unique()) == {2023, 2024, 2025}
    assert d["load"].between(80, 220).all()
    assert (d[d.year == 2025]["T_fc"].notna()).all()


def test_results_reproducible():
    p = ROOT / "experiments" / "exp_b_results.json"
    if not p.exists():
        pytest.skip("нет результатов")
    import json
    res = json.loads(p.read_text(encoding="utf-8"))
    d = BD.build()
    tr = d[d.year <= 2024].dropna(subset=["T", "Tlag", "Tmax"])
    te = d[d.year == 2025]
    pred = B.Linear(B.l1_features).fit(R.X(tr), tr["load"].to_numpy()).predict(R.X(te))
    assert np.mean(np.abs(te["load"].to_numpy() - pred)) == pytest.approx(res["metrics"]["act"]["L1"]["все"]["MAE"], abs=1e-9)
