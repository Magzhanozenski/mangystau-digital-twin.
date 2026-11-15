"""Проверка модели DLR (шаг 3.3, пункт 5).

1) Согласованность с ПУЭ: табличный ток каждого провода (табл. 1.3.29, +25 °C, без солнца) должен получаться при
   одном и том же «эквивалентном» ветре поперёк линии для обоих проводов (разброс < 5 %).
2) Поправки ПУЭ на температуру воздуха (табл. 1.3.3) против теплового баланса при эквивалентном ветре — расхождение
   записывается (ПУЭ — упрощённая формула, баланс — физика; совпадение не обязано быть точным).
3) Сверка с JS-реализацией прототипа (core_js/day.js, dlrAmp) на сетке условий: та же физика при угле 45°,
   ε = α = 0,5 — расхождение < 0,5 % (другие R20: ГОСТ 0,1591 против 0,159 в прототипе).
4) Физические свойства: ток падает с ростом T воздуха и солнца, растёт с ветром; ветер поперёк охлаждает сильнее,
   чем вдоль.
Контрольного примера IEEE 738 с числовым ответом в открытом доступе не найдено (dlr_facts.yaml → not_found).
Отчёт — validation/check_dlr_report.md. Код выхода 1 — если не выполнен п. 1, 3 или 4.

Запуск из корня:  .venv\\Scripts\\python validation\\check_dlr.py   (для п. 3 нужен Node.js)
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import dlr as L  # noqa: E402

REPORT = Path(__file__).resolve().parent / "check_dlr_report.md"

JS = r"""
Object.assign(global, require('./core_js/engine2.js'));
global.mulberry32 = a => () => 0.5;
const d = require('./core_js/day.js');
const grid = JSON.parse(process.argv[1]);
console.log(JSON.stringify(grid.map(g => d.dlrAmp('АС-185', g[0], g[1], g[2]))));
"""


def main() -> int:
    out, bad = [], 0

    def say(s=""):
        print(s)
        out.append(s)
    say("# Проверка модели DLR (twin/dlr.py)")
    say("")
    # 1
    say("## 1. Согласованность с ПУЭ (табл. 1.3.29 при +25 °C, без солнца, ветер поперёк)")
    eq = {}
    for c, p in L.CONDUCTORS.items():
        eq[c] = brentq(lambda v: float(L.ampacity(c, 25, v, 90, 0)) - p["I_pue"], 0.001, 10)
        say(f"- {c}: ПУЭ {p['I_pue']:.0f} А получается при ветре {eq[c]:.3f} м/с")
    spread = (max(eq.values()) - min(eq.values())) / np.mean(list(eq.values()))
    ok1 = spread < 0.05
    bad += 0 if ok1 else 1
    say(f"- разброс эквивалентного ветра {100 * spread:.1f} % — {'согласовано' if ok1 else 'НЕ согласовано'} "
        f"(ПУЭ не указывает ветер; по расчёту ≈ {np.mean(list(eq.values())):.2f} м/с поперёк, без солнца, ε = 0,5)")
    # 2
    say("")
    say("## 2. Поправка ПУЭ на температуру воздуха (табл. 1.3.3) против теплового баланса")
    say("")
    say("| T воздуха, °C | ПУЭ | баланс АС-185 | баланс АС-300 |")
    say("|---|---|---|---|")
    dev = 0.0
    for t, k in sorted(L.PUE_CORR.items()):
        r = [float(L.ampacity(c, t, eq[c], 90, 0) / L.ampacity(c, 25, eq[c], 90, 0)) for c in L.CONDUCTORS]
        dev = max(dev, max(abs(x - k) for x in r))
        say(f"| {t:g} | {k:.2f} | {r[0]:.3f} | {r[1]:.3f} |")
    say(f"- макс. расхождение {dev:.3f} — записывается, не критерий")
    # 3
    say("")
    say("## 3. Сверка с JS-прототипом (dlrAmp, АС-185, угол 45°, ε = α = 0,5)")
    grid = [[ta, w, s] for ta in (-10, 10, 25, 40, 48) for w in (0, 0.5, 2, 6) for s in (0, 500, 1000)]
    if shutil.which("node"):
        r = subprocess.run(["node", "-e", JS, json.dumps(grid)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            say(f"- ошибка JS: {r.stderr.strip().splitlines()[-1]}")
            bad += 1
        else:
            js = np.array(json.loads(r.stdout.strip().splitlines()[-1]))
            py = np.array([float(L.ampacity("АС-185", ta, w, 45.0, s, elevation_m=30)) for ta, w, s in grid])
            # прототип: R20 = 0,159 Ом/км; здесь — ГОСТ 0,1591 → поправка √(0,1591/0,159)
            py_adj = py * np.sqrt(L.CONDUCTORS["АС-185"]["R20"] / 0.159e-3)
            rel = np.abs(py_adj - js) / js
            ok3 = rel.max() < 0.005
            bad += 0 if ok3 else 1
            say(f"- {len(grid)} условий: макс. расхождение {100 * rel.max():.3f} %, среднее {100 * rel.mean():.3f} % — "
                f"{'совпадает' if ok3 else 'РАСХОЖДЕНИЕ'} (после поправки на R20 и высоту 30 м, как в прототипе)")
    else:
        say("- Node.js не найден — сверка пропущена")
    # 4
    say("")
    say("## 4. Физические свойства")
    c = "АС-185"
    checks = {
        "ток падает с ростом T воздуха": bool(np.all(np.diff(L.ampacity(c, np.arange(-20, 50, 5), 2, 90, 300)) < 0)),
        "ток растёт с ветром": bool(np.all(np.diff(L.ampacity(c, 30, np.arange(0, 10, 0.5), 90, 300)) > 0)),
        "солнце снижает ток": bool(L.ampacity(c, 30, 2, 90, 1000) < L.ampacity(c, 30, 2, 90, 0)),
        "ветер поперёк охлаждает сильнее, чем вдоль": bool(L.ampacity(c, 30, 3, 90, 0) > L.ampacity(c, 30, 3, 0, 0)),
        "угол ветра 0…90 при любом направлении": bool(np.all((L.wind_angle(np.arange(0, 360, 7), 33) >= 0)
                                                           & (L.wind_angle(np.arange(0, 360, 7), 33) <= 90))),
        "при T воздуха = T провода ток 0": bool(float(L.ampacity(c, 70, 3, 90, 0)) < 1e-6),
    }
    for k, v in checks.items():
        say(f"- {k}: {'да' if v else 'НЕТ'}")
    bad += sum(not v for v in checks.values())
    say("")
    say("Контрольный пример IEEE 738 с числовым ответом не найден в открытом доступе — сверка с ним не выполнена.")
    say("")
    say("Итог: " + ("все проверки пройдены" if not bad else f"{bad} проверок не пройдено"))
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
