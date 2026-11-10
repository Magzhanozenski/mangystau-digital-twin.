"""Сверка Python-ядра (twin.powerflow) с pandapower на сценариях validation/cases.json.

Запуск из корня репозитория (нужен pandapower: pip install pandapower):
    node validation/export.js          # если cases.json ещё нет
    python validation/compare_pandapower.py

Критерий готовности шага 1.3: |ΔU| < 1e-4 о.е. и |ΔP| < 0.01 МВт на всех сошедшихся режимах.
Сравниваются только режимы со статусом ok (предел P–V pandapower не определяет).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import load_network, run_power_flow  # noqa: E402
from twin.pp_model import run_pp  # noqa: E402

TOL_V, TOL_P = 1e-4, 0.01


def compare(cases, net, limit=None):
    worst = {"V": 0.0, "P": 0.0, "Q": 0.0, "genP": 0.0, "genQ": 0.0}
    fails, n, errors = [], 0, []
    for c in cases[:limit]:
        if c["status"] != "ok":
            continue
        r = run_power_flow(c, net)
        try:
            p = run_pp(c, net, r["islands"])
        except Exception as e:  # расхождение постановки или несходимость pandapower — фиксируем, не скрываем
            errors.append((c["T"], c["load"], c["linesOut"], c["gensOut"], repr(e)[:120]))
            continue
        n += 1
        dv = max(abs(p["V"][b] - r["V"][b]) for b in r["live"])
        dp = max(abs(p["flows"][l]["P"] - f["P"]) for l, f in r["flows"].items() if not f.get("out") and not f.get("dead"))
        dq = max(abs(p["flows"][l]["Q"] - f["Q"]) for l, f in r["flows"].items() if not f.get("out") and not f.get("dead"))
        dgp = max(abs(p["gen"][g]["P"] - x["P"]) for g, x in r["gen"].items() if not x.get("out"))
        dgq = max(abs(p["gen"][g]["Q"] - x["Q"]) for g, x in r["gen"].items() if not x.get("out"))
        for k, v in zip(worst, (dv, dp, dq, dgp, dgq)):
            worst[k] = max(worst[k], v)
        if dv > TOL_V or dp > TOL_P:
            fails.append((round(c["T"], 1), round(c["load"]), c["linesOut"], c["gensOut"], dv, dp))
    return n, worst, fails, errors


if __name__ == "__main__":
    data = json.loads((ROOT / "validation" / "cases.json").read_text(encoding="utf-8"))
    net = load_network()
    n, worst, fails, errors = compare(data["cases"], net)
    print(f"Сравнено режимов: {n}")
    print(f"Макс. |ΔU| = {worst['V']:.2e} о.е., |ΔP| = {worst['P']:.2e} МВт, |ΔQ| = {worst['Q']:.2e} Мвар")
    print(f"Генерация: |ΔP| = {worst['genP']:.2e} МВт, |ΔQ| = {worst['genQ']:.2e} Мвар")
    print(f"Вне допуска (|ΔU| > {TOL_V} или |ΔP| > {TOL_P}): {len(fails)}")
    for f in fails[:10]:
        print("   ", f)
    print(f"Ошибки pandapower: {len(errors)}")
    for e in errors[:10]:
        print("   ", e)
    sys.exit(0 if not fails and not errors else 1)
