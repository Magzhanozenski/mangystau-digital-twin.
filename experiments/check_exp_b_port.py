"""Сверка переноса моделей прототипа на Python (эксперимент B, пункт 1).

JS-прототип (core_js/day.js + mulberry32 из web_prototype/index.html) строит свою синтетическую историю, обучает
fitLinear и trainLoadModel так же, как web_prototype (первый год — обучение, второй — проверка), и выдаёт прогнозы.
Python-модели (exp_b_models) обучаются на тех же строках; прогнозы и метрики должны совпасть.
Отчёт — experiments/check_exp_b_port_report.md; код выхода 1 при расхождении.

Запуск из корня:  .venv\\Scripts\\python experiments\\check_exp_b_port.py   (нужен Node.js)
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))

import exp_b_models as B  # noqa: E402

REPORT = Path(__file__).resolve().parent / "check_exp_b_port_report.md"
JS = r"""
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
global.mulberry32 = mulberry32;
Object.assign(global, require('./core_js/engine2.js'));
// solveLinear не экспортируется из engine2.js — берём его текст оттуда же, без изменений
const src = require('fs').readFileSync('./core_js/engine2.js', 'utf8');
global.solveLinear = eval('(' + src.match(/function solveLinear[\s\S]*?\r?\n\}\r?\n/)[0] + ')');
const d = require('./core_js/day.js');
const H = d.makeHistory();
const tr = H.filter(r => r.d < 365), te = H.filter(r => r.d >= 365);
const lin = d.fitLinear(tr);
const g = d.trainLoadModel(tr, 6, 16); let x; while (!(x = g.next()).done) {} const mlp = x.value;
const r10 = mulberry32(123); const rs = [r10(), r10(), mulberry32(5)()];
console.log(JSON.stringify({rows: H.map(r => [r.d, r.h, r.T, r.Tlag, r.Tmax, r.weekend ? 1 : 0, r.load]),
  lin: te.map(r => lin.predict(r)), mlp: te.map(r => mlp.predict(r)), rnd: rs,
  tl: H.slice(0, 50).map(r => d.trueLoad(r.h, r.T, r.Tlag, r.weekend))}));
"""


def main() -> int:
    r = subprocess.run(["node", "-e", JS], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        print(r.stderr)
        return 1
    js = json.loads(r.stdout.strip().splitlines()[-1])
    rows = np.array(js["rows"])
    d, h, T, Tlag, Tmax, we, load = (rows[:, i] for i in range(7))
    X = lambda m: {"h": h[m].astype(int), "T": T[m], "Tlag": Tlag[m], "Tmax": Tmax[m], "weekend": we[m] > 0}  # noqa: E731
    tr, te = d < 365, d >= 365
    py_rnd = [B.mulberry32(123)(), None, B.mulberry32(5)()]
    g = B.mulberry32(123)
    py_rnd[0], py_rnd[1] = g(), g()
    lin = B.Linear(B.l0_features).fit(X(tr), load[tr]).predict(X(te))
    mlp = B.MLP().fit(X(tr), load[tr]).predict(X(te))
    tl = B.true_load(h[:50].astype(int), T[:50], Tlag[:50], we[:50] > 0)
    jl, jm = np.array(js["lin"]), np.array(js["mlp"])
    mape = lambda p: 100 * np.mean(np.abs(load[te] - p) / load[te])  # noqa: E731
    res = {
        "mulberry32": float(np.max(np.abs(np.array(py_rnd) - np.array(js["rnd"])))),
        "trueLoad": float(np.max(np.abs(tl - np.array(js["tl"])))),
        "L0, макс. |Δ| прогноза, МВт": float(np.max(np.abs(lin - jl))),
        "MLP, макс. |Δ| прогноза, МВт": float(np.max(np.abs(mlp - jm))),
    }
    out = ["# Сверка переноса моделей прототипа (JS → Python)", "",
           f"Синтетическая история прототипа: {len(rows)} часов, обучение — 1-й год, проверка — 2-й.", ""]
    out += [f"- {k}: {v:.2e}" for k, v in res.items()]
    out += [f"- MAPE проверки: L0 — JS {mape(jl):.4f} %, Python {mape(lin):.4f} %; MLP — JS {mape(jm):.4f} %, "
            f"Python {mape(mlp):.4f} %"]
    ok = res["mulberry32"] == 0 and res["trueLoad"] < 1e-9 and res["L0, макс. |Δ| прогноза, МВт"] < 1e-6 \
        and res["MLP, макс. |Δ| прогноза, МВт"] < 1e-3
    out.append(f"- итог: {'перенос совпадает' if ok else 'РАСХОЖДЕНИЕ'}")
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
