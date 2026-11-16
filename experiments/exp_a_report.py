"""Отчёт и рисунки эксперимента A (пункт 5): experiments/exp_a_report.md, experiments/fig_a*.png.

Читает experiments/exp_a_results.json (exp_a_ratings.py) и пересчитывает допустимые токи для рисунка 1.
Запуск из корня:  .venv\\Scripts\\python experiments\\exp_a_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import exp_a_ratings as A  # noqa: E402

HERE = Path(__file__).resolve().parent
COLORS = {"truth": "#222222", "M1": "#c0392b", "M2": "#e67e22", "M3": "#2980b9", "M4": "#27ae60"}


def main() -> int:
    res = json.loads((HERE / "exp_a_results.json").read_text(encoding="utf-8"))
    runs = {r["stress"]: r for r in res["runs"]}
    raw = yaml.safe_load((ROOT / "data" / "network_real.yaml").read_text(encoding="utf-8"))
    rat, info, cond = A.ratings(raw)

    # рисунок 1: распределение DLR («правда») / ПУЭ по проводам
    fig, ax = plt.subplots(figsize=(7, 4))
    for c, col in (("АС-185", "#2980b9"), ("АС-300", "#c0392b")):
        r = np.concatenate([rat["truth"][k] / rat["M1"][k] for k, v in info.items() if v["conductor"] == c])
        ax.hist(r, bins=80, range=(0.3, 3.0), histtype="step", lw=1.8, color=col, density=True, label=f"{c} (ВЛ {'110' if c == 'АС-185' else '220'} кВ)")
    ax.axvline(1, color="k", ls="--", lw=1)
    ax.set_xlabel("DLR по фактической погоде / ток ПУЭ (510 / 710 А)")
    ax.set_ylabel("плотность (линия·часов)")
    ax.set_title("Рис. 1. DLR относительно справочного тока, 2025 г., 52 линии")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "fig_a1_dlr_ratio.png", dpi=150)
    plt.close(fig)

    # рисунок 2: найденные перегрузки и ложные тревоги (уровень «час») по методам и нагрузке
    st = sorted(runs)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for m, name in A.METHODS.items():
        tp = [runs[s]["hour"][m]["TP"] for s in st]
        tot = [runs[s]["hours_true"] for s in st]
        rec = [100 * a / b if b else np.nan for a, b in zip(tp, tot)]
        fp = [runs[s]["hour"][m]["FP"] for s in st]
        axes[0].plot(st, rec, "o-", color=COLORS[m], label=f"{m}: {name}")
        axes[1].plot(st, fp, "o-", color=COLORS[m], label=m)
    axes[0].set_xlabel("нагрузка, × 2025 г.")
    axes[0].set_ylabel("найдено часов с перегрузкой, %")
    axes[0].set_title("Обнаружение (N-1, уровень «час»)")
    axes[1].set_xlabel("нагрузка, × 2025 г.")
    axes[1].set_ylabel("часов с ложной тревогой")
    axes[1].set_title("Ложные тревоги")
    axes[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Рис. 2. Справочный ток против DLR при одинаковой нагрузке")
    fig.tight_layout()
    fig.savefig(HERE / "fig_a2_detection.png", dpi=150)
    plt.close(fig)

    # отчёт
    out = ["# Эксперимент A: справочный допустимый ток против DLR при одинаковой нагрузке", "",
           "**Вопрос.** Насколько часто справочный ток ошибается относительно реального допустимого тока: ложные тревоги "
           "(метод видит перегрузку, которой нет) и скрытые риски (перегрузка есть, метод её не видит).", "",
           "**Постановка.** Сеть `network_real.yaml` (110–220 кВ, состав генерации 2023 г.), 2025 г. по часам "
           f"({res['lines']} цепей, нормальная схема и N-1 — {res['contingencies']} схем). «Правда» — DLR по фактической "
           "погоде (ERA5, ветер приведён к станциям, угол ветра по пролётам; шаг 3.3). Методы — по прогнозу на сутки:", ""]
    out += [f"- **{m}** — {n}" for m, n in A.METHODS.items()]
    out += ["", "Нагрузка 2025 г. — модель с зависимостью от температуры (допущение, `exp_a_load.py`: средняя 392, "
            "максимум 546 МВт); ×1,2–1,6 — условное утяжеление для проверки методов при перегрузках. Токи — на сетке "
            "нагрузки с интерполяцией; проверка без интерполяции — 0 расхождений из 12,4 млн решений "
            "(`check_exp_a_report.md`).", "",
            "## Результаты (уровень «час»: была ли перегрузка в любой цепи при любой схеме N-1)", "",
            "| нагрузка | часов с перегрузкой | метод | найдено | пропущено (скрытый риск) | ложных тревог |",
            "|---|---|---|---|---|---|"]
    for s in st:
        r = runs[s]
        for m in A.METHODS:
            h = r["hour"][m]
            out.append(f"| ×{s} | {r['hours_true']} | {m} | {h['TP']} | {h['FN']} | {h['FP']} |")
    out += ["", "## Наибольшая загрузка относительно допустимого тока метода (худшая цепь, худший час, N-1)", "",
            "| нагрузка | «правда» | M1 | M2 | M3 | M4 |", "|---|---|---|---|---|---|"]
    for s in st:
        ml = runs[s]["max_loading"]
        out.append(f"| ×{s} | {ml['truth']:.2f} | {ml['M1']:.2f} | {ml['M2']:.2f} | {ml['M3']:.2f} | {ml['M4']:.2f} |")
    out += ["", "## Уровень «линия·схема·час» при ×1,6 (в скобках — жаркие часы, ERA5 ≥ 35 °C; штиль < 1 м/с)", "",
            "| метод | найдено | ложных | скрытых | скрытых в жару | скрытых в штиль |", "|---|---|---|---|---|---|"]
    r = runs[max(st)]
    for m in A.METHODS:
        e, eh, ec = r["element"][m], r["element_hot"][m], r["element_calm"][m]
        out.append(f"| {m} | {e['TP']} | {e['FP']} | {e['FN']} | {eh['FN']} | {ec['FN']} |")
    out += ["", "![Рис. 1](fig_a1_dlr_ratio.png)", "", "![Рис. 2](fig_a2_detection.png)", "",
            "## Выводы", "",
            "1. При нагрузке 2025 г. перегрузок нет ни по одному методу и ни по «правде» (сеть `network_real` загружена "
            "слабо). Но **справочный ток почти вдвое переоценивает запас**: наибольшая загрузка по ПУЭ — 0,53 (0,60 "
            "с поправкой на температуру) при 0,80 относительно фактического допустимого тока.",
            "2. При утяжелении справочный ток (M1, M2) **систематически пропускает реальные перегрузки**: при ×1,4 — все 21 "
            "час, при ×1,6 находит 8 (M1) и 34 (M2) из 107. Ложных тревог у него мало — его ошибка односторонняя "
            "(скрытый риск), потому что при штиле и жаре реальный допустимый ток ниже табличного.",
            "3. DLR по медиане прогноза (M3) тоже пропускает большую часть (2 из 21; 39 из 107): ошибка прогноза ветра и "
            "температуры симметрична, а перегрузка возникает в худших часах.",
            "4. Осторожный DLR (M4: ветер P10, температура и солнце P90) находит почти всё (14 из 21; 95 из 107), но ценой "
            "ложных тревог (95 и 250 часов) — это плата за неопределённость прогноза; её можно уменьшать уровнем квантиля.",
            "", "## Ограничения", "",
            "- «Правда» — DLR по реанализу ERA5 с поправкой ветра к станциям, а не измерения на линиях; ERA5 занижает жару "
            "в Актау (шаг 2.1) — реальных опасных часов, вероятно, больше.",
            "- Нагрузка 2025 г. — модель (почасовых данных нет); утяжеление ×1,2–1,6 — условное.",
            "- Сеть — только линии с тегом 110/220 кВ в OSM; провода — допущения (АС-185 / АС-300), ε и α — допущения.",
            "- Погода — три точки; до части линий 80–190 км.",
            "- Токи посчитаны при +25 °C в проводе (влияние температуры на сопротивление на токи мало)."]
    (HERE / "exp_a_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("ok:", HERE / "exp_a_report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
