"""Нагрузка: реальные данные против заложенных допущений (шаг 2.3, пункт 4).

Сравнивается (ничего не подгоняется, расхождения только записываются):
  1) условная сеть data/network.yaml — «160 МВт» в basis узлов и доли по ПС;
  2) синтетическая история core_js/day.js (trueLoad, makeHistory) — уровень, коэффициент заполнения,
     доля постоянной (промышленной) составляющей; генератор случайных чисел mulberry32 — тот же, что
     в web_prototype/index.html (в core_js он не определён);
  3) диапазон нагрузки сценариев сверки validation/export.js (90–260 МВт);
  4) оценки нагрузки network_real.yaml (load_estimates.yaml) — согласованность с фактами load_facts.yaml.
Пункты 1–3 — справочные; пункт 4 — проверки, при нарушении код выхода 1.
Отчёт печатается и сохраняется в validation/check_load_report.md.

Запуск из корня:  .venv\\Scripts\\python validation\\check_load.py   (для пункта 2 нужен Node.js)
"""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "network_sources"
REPORT = Path(__file__).resolve().parent / "check_load_report.md"

JS = r"""
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
global.mulberry32 = mulberry32;
Object.assign(global, require('./core_js/engine2.js'));
const d = require('./core_js/day.js');
const rows = d.makeHistory();
const L = rows.map(r => r.load).sort((a, b) => a - b);
const mean = L.reduce((a, b) => a + b, 0) / L.length;
const q = p => L[Math.floor(p * (L.length - 1))];
const ind = rows.map(r => 88 * (r.weekend ? 0.97 : 1)).reduce((a, b) => a + b, 0) / rows.length;
console.log(JSON.stringify({n: rows.length, min: L[0], p05: q(.05), p50: q(.5), mean, p95: q(.95), max: L[L.length - 1], ind_mean: ind}));
"""


def synthetic():
    if not shutil.which("node"):
        return None, "Node.js не найден — синтетическая история не посчитана"
    r = subprocess.run(["node", "-e", JS], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    if r.returncode:
        return None, "ошибка запуска day.js: " + r.stderr.strip().splitlines()[-1]
    return json.loads(r.stdout.strip().splitlines()[-1]), ""


def fact(facts, name_part, field_part=""):
    for f in facts["facts"]:
        if name_part in f["object"]["name"] and field_part in f["field"]:
            return f
    raise KeyError(name_part)


def main() -> int:
    facts = yaml.safe_load((SRC / "load_facts.yaml").read_text(encoding="utf-8"))
    est = yaml.safe_load((SRC / "load_estimates.yaml").read_text(encoding="utf-8"))
    cond = yaml.safe_load((ROOT / "data" / "network.yaml").read_text(encoding="utf-8"))
    real = yaml.safe_load((ROOT / "data" / "network_real.yaml").read_text(encoding="utf-8"))

    peak = fact(facts, "АО «МРЭК»", "максимальная нагрузка")["value"]
    energy = fact(facts, "АО «МРЭК»", "объём передачи")["value"]
    cons = fact(facts, "Крупные потребители МРЭК (2023)")["value"]
    restr = fact(facts, "Ограничение потребителей 03.07.2023")["value"]
    node = fact(facts, "Мангистауский энергоузел")["value"]
    aktau = fact(facts, "г. Актау")["value"]
    avg23 = energy["y2023"] * 1000 / 8760
    lf23 = avg23 / peak["y2023"]
    oil = sum(v for k, v in cons.items() if "мунайгаз" in k or "мунай" in k or "Buzachi" in k)
    ls = real["meta"]["load_scenarios"]

    out = ["# Нагрузка: реальные данные против допущений", "",
           "Факты — data/network_sources/load_facts.yaml; оценки — load_estimates.yaml. Ничего не подгонялось.", "",
           "## Опорные факты (сеть АО «МРЭК»)", "",
           f"- максимум нагрузки: {peak['y2019']} (2019) → {peak['y2023']} (2023) → {peak['y2024']} МВт (2024);",
           f"- передача 2023: {energy['y2023']} млн кВт·ч → средняя {avg23:.0f} МВт, коэффициент заполнения {lf23:.2f};",
           f"- нефтедобыча (Озенмунайгаз, Мангистаумунайгаз, Каражанбасмунай, Каракудукмунай, Buzachi Operating): "
           f"{oil} млн кВт·ч = {100 * oil / energy['y2023']:.0f} % передачи 2023;",
           f"- энергоузел: максимум {node['без_передачи']} МВт ({node['с_передачей_за_пределы']} с передачей), дата не указана; "
           f"пик Актау {aktau} МВт.", "",
           "## 1. Условная сеть network.yaml", ""]
    total_cond = None
    m = re.search(r"из (\d+) МВт", cond["buses"][1]["basis"])
    if m:
        total_cond = int(m.group(1))
    beyneu = next(b for b in cond["buses"] if "Бейнеу" in b["name"])
    out += [f"- суммарная нагрузка в basis узлов: {total_cond} МВт — это {100 * total_cond / peak['y2023']:.0f} % "
            f"максимума МРЭК 2023 и {100 * total_cond / avg23:.0f} % средней нагрузки;",
            f"- ПС Бейнеу: {beyneu['share'] * total_cond:.0f} МВт; по оценке шага 2.3 остров Бейнеу в среднем "
            f"{ls['by_island_avg_mw']['Бейнеу']:.1f} МВт (прочая нагрузка — допущение 10–40 МВт).", "",
            "## 2. Синтетическая история core_js/day.js", ""]
    syn, why = synthetic()
    if syn:
        lf_syn = syn["mean"] / syn["max"]
        out += [f"- {syn['n']} часов: мин {syn['min']:.0f}, p5 {syn['p05']:.0f}, медиана {syn['p50']:.0f}, "
                f"средняя {syn['mean']:.0f}, p95 {syn['p95']:.0f}, макс {syn['max']:.0f} МВт;",
                f"- коэффициент заполнения {lf_syn:.2f} (МРЭК 2023 — {lf23:.2f});",
                f"- постоянная «промышленная» часть {syn['ind_mean']:.0f} МВт = {100 * syn['ind_mean'] / syn['mean']:.0f} % "
                f"средней (нефтедобыча в МРЭК 2023 — {100 * oil / energy['y2023']:.0f} % энергии);",
                f"- уровень: максимум синтетики — {100 * syn['max'] / peak['y2023']:.0f} % реального максимума МРЭК 2023."]
    else:
        out.append(f"- не посчитано: {why}")
    out += ["", "## 3. Сценарии сверки validation/export.js", "",
            f"- нагрузка 90–260 МВт: верх — {100 * 260 / peak['y2024']:.0f} % максимума МРЭК 2024. "
            "Для сверки решателей уровень не важен; для выводов о реальной сети — занижен.", "",
            "## 4. Оценки нагрузки network_real.yaml — проверки согласованности", ""]

    checks = []
    used = sum(c["energy_mln_kwh"] for c in est["consumers"] if "energy_mln_kwh" in c)
    checks.append(("потребление, взятое в оценки, не больше передачи МРЭК 2023",
                   used <= energy["y2023"], f"{used} ≤ {energy['y2023']} млн кВт·ч"))
    checks.append(("пик модели не больше максимума МРЭК 2023", ls["peak"] <= peak["y2023"],
                   f"{ls['peak']:.1f} ≤ {peak['y2023']} МВт"))
    kz = next(c for c in est["consumers"] if c["id"] == "karazhanbas")["energy_mln_kwh"] * 1000 / 8760
    bo = next(c for c in est["consumers"] if c["id"] == "buzachi_operating")["energy_mln_kwh"] * 1000 / 8760
    pk = est["scenarios"]["peak"]["factor"]
    checks.append(("ограничение Каражанбасмунай 03.07.2023 (27 МВт) не больше его пиковой оценки",
                   restr["Каражанбасмунай"] <= kz * pk, f"27 ≤ {kz * pk:.1f} МВт (средняя {kz:.1f})"))
    checks.append(("ограничение Buzachi Operating 03.07.2023 (17 МВт) не больше его пиковой оценки",
                   restr["Buzachi Operating"] <= bo * pk, f"17 ≤ {bo * pk:.1f} МВт (средняя {bo:.1f})"))
    checks.append(("сумма долей нагрузки ПС = 1", abs(sum(b["share"] for b in real["buses"]) - 1) < 1e-4,
                   f"{sum(b['share'] for b in real['buses']):.6f}"))
    checks.append(("средняя нагрузка модели = сумма load_mw_avg ПС",
                   abs(ls["base"] - sum(b.get("load_mw_avg", 0) for b in real["buses"])) < 1e-2, f"{ls['base']:.3f} МВт"))
    for name, ok, detail in checks:
        out.append(f"- [{'x' if ok else ' '}] {name}: {detail}")
    share_src = real["summary"]["load_energy_by_mw"]["share"]
    out += ["", f"Модель покрывает {ls['base']:.1f} МВт средней нагрузки = {100 * ls['base'] / avg23:.0f} % средней нагрузки МРЭК 2023; "
            f"{100 * share_src:.0f} % из неё опирается на годовое потребление из документов, остальное — допущение (Бейнеу).",
            "Раскладка по ПС и коэффициенты режимов — допущения. Почасовых данных нет: связь нагрузки с погодой "
            "на реальных данных не проверена."]
    text = "\n".join(out) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
