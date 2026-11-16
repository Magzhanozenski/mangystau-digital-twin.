"""Отчёт и рисунки эксперимента B: experiments/exp_b_report.md, fig_b1_error_by_T.png, fig_b2_mae.png.

Читает experiments/exp_b_results.json (exp_b_run.py); для рисунка 1 модели переобучаются теми же функциями.
Запуск из корня:  .venv\\Scripts\\python experiments\\exp_b_report.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp_b_data as BD  # noqa: E402
import exp_b_models as B  # noqa: E402
import exp_b_run as R  # noqa: E402

COL = {"L0": "#c0392b", "L1": "#2980b9", "MLP": "#27ae60"}
NAME = {"L0": "L0 — линейная прототипа (час, выходной, T)", "L1": "L1 — честная линейная (признаки НС + T-члены)",
        "MLP": "MLP — нейросеть прототипа"}


def main() -> int:
    res = json.loads((HERE / "exp_b_results.json").read_text(encoding="utf-8"))
    d = BD.build()
    tr = d[d.year <= 2024].dropna(subset=["T", "Tlag", "Tmax"])
    te = d[d.year == 2025].dropna(subset=["T_fc", "Tlag_fc", "Tmax_fc"])
    y = te["load"].to_numpy()
    models = {"L0": B.Linear(B.l0_features), "L1": B.Linear(B.l1_features), "MLP": B.MLP()}
    pr = {k: m.fit(R.X(tr), tr["load"].to_numpy()).predict(R.X(te, fc=True)) for k, m in models.items()}

    # рис. 1: MAE по диапазонам фактической температуры (признаки — по прогнозу)
    bins = np.arange(-20, 45, 5)
    T = te["T"].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4))
    mid = (bins[:-1] + bins[1:]) / 2
    for k, p in pr.items():
        e = np.abs(y - p)
        mae = [e[(T >= a) & (T < b)].mean() if ((T >= a) & (T < b)).sum() >= 20 else np.nan for a, b in zip(bins[:-1], bins[1:])]
        ax.plot(mid, mae, "o-", color=COL[k], label=NAME[k])
    ax2 = ax.twinx()
    ax2.hist(T, bins=bins, color="#999999", alpha=0.25)
    ax2.set_ylabel("часов в диапазоне", color="#777777")
    ax.set_xlabel("фактическая температура, °C (среднее трёх точек ERA5)")
    ax.set_ylabel("MAE, МВт")
    ax.set_yscale("log")
    ax.set_title("Рис. 1. Ошибка прогноза на сутки по температуре, 2025 г.")
    ax.legend(fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(HERE / "fig_b1_error_by_T.png", dpi=150)
    plt.close(fig)

    # рис. 2: MAE все часы / жара, по факту и прогнозу; для MLP — разброс по инициализациям
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    for ax, inp, title in ((axes[0], "act", "признаки по факту"), (axes[1], "fc", "признаки по прогнозу")):
        m = res["metrics"][inp]
        x = np.arange(2)
        for i, k in enumerate(("L1", "MLP")):
            vals = [m[k]["все"]["MAE"], m[k]["жара"]["MAE"]]
            ax.bar(x + (i - 0.5) * 0.35, vals, 0.35, color=COL[k], label=k)
        sp = res["mlp_spread"][inp]
        ax.errorbar(x + 0.175, [m["MLP"]["все"]["MAE"], m["MLP"]["жара"]["MAE"]],
                    yerr=[[m["MLP"]["все"]["MAE"] - sp["MAE_min"], m["MLP"]["жара"]["MAE"] - sp["hot_MAE_min"]],
                          [sp["MAE_max"] - m["MLP"]["все"]["MAE"], sp["hot_MAE_max"] - m["MLP"]["жара"]["MAE"]]],
                    fmt="none", ecolor="k", capsize=4, label="MLP: 10 инициализаций")
        ax.set_xticks(x, ["все часы", f"жара (≥ 35 °C, {res['hot_hours'][inp]} ч)"])
        ax.set_ylabel("MAE, МВт")
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.suptitle("Рис. 2. Нейросеть против честной линейной модели, проверка 2025 г. (L0 — вне шкалы: 7,1 / 27,0 МВт)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_b2_mae.png", dpi=150)
    plt.close(fig)

    out = ["# Эксперимент B: нейросеть против линейной модели на независимом периоде", "",
           "**Вопрос.** Даёт ли нейросеть выигрыш перед линейной моделью прогноза нагрузки на независимом периоде — во всех "
           "часах и в жаркие часы?", "",
           "**«Правда».** Реальной почасовой нагрузки нет. Нагрузка — генератор прототипа `trueLoad` (промышленность, быт "
           "с профилем, коммерция, кондиционирование с квадратичным ростом выше 30 °C, отопление, шум AR(1)) на **реальной "
           "погоде** ERA5 2023–2025 (среднее трёх точек). Это синтетика: выводы касаются методов, а не нагрузки Мангистау.", "",
           f"**Обучение** — 2023–2024 ({res['train'][2]} ч, признаки по факту), **проверка** — 2025: по факту и по прогнозу "
           "на сутки (Open-Meteo, выпуск накануне). Жара — фактическая T ≥ 35 °C "
           f"({res['hot_hours']['act']} ч). Перенос моделей на Python совпадает с JS-прототипом "
           "(`check_exp_b_port_report.md`).", ""]
    out += [f"- **{k}** — {v}" for k, v in NAME.items()]
    for inp, title in (("act", "Признаки по факту (качество модели)"), ("fc", "Признаки по прогнозу на сутки (эксплуатация)")):
        out += ["", f"## {title}", "", "| модель | MAE, МВт | MAPE, % | RMSE | жара: MAE | жара: MAPE, % | жара: смещение |",
                "|---|---|---|---|---|---|---|"]
        for k, m in res["metrics"][inp].items():
            a, h = m["все"], m["жара"]
            out.append(f"| {k} | {a['MAE']:.2f} | {a['MAPE']:.2f} | {a['RMSE']:.2f} | {h['MAE']:.2f} | {h['MAPE']:.2f} | {h['bias']:+.2f} |")
        sp = res["mlp_spread"][inp]
        out.append("")
        out.append(f"MLP по 10 инициализациям: MAE {sp['MAE_min']:.2f}…{sp['MAE_max']:.2f}; в жару {sp['hot_MAE_min']:.2f}…{sp['hot_MAE_max']:.2f} МВт.")
        out += ["", "| сравнение | ΔMAE (все), МВт [95 % ДИ] | DM p | ΔMAE (жара) [95 % ДИ] | p |", "|---|---|---|---|---|"]
        for k, c in res["compare"][inp].items():
            out.append(f"| {k} | {c['dMAE']:+.3f} [{c['ci'][0]:+.3f}; {c['ci'][1]:+.3f}] | {c['p']:.2g} | "
                       f"{c['dMAE_hot']:+.3f} [{c['ci_hot'][0]:+.3f}; {c['ci_hot'][1]:+.3f}] | {c['p_hot']:.2g} |")
    out += ["", "![Рис. 1](fig_b1_error_by_T.png)", "", "![Рис. 2](fig_b2_mae.png)", "", "## Выводы", "",
            "1. **Преимущество нейросети в прототипе — в основном заслуга слабой линейной модели.** Линейная модель прототипа "
            "(L0) почти не зависит от температуры (наклон 0,02 МВт/°C: рост в жару и в холод гасят друг друга) — в жару её "
            "ошибка 27 МВт (15 %). Честная линейная модель (L1) с теми же признаками и простыми температурными членами "
            "сокращает ошибку в 2,4 раза по всем часам и в 9 раз в жару.",
            "2. **Против честной линейной модели нейросеть выигрывает немного:** 0,11 МВт (4 %) по факту и 0,17 МВт (5 %) "
            "по прогнозу; разница статистически значима, но сравнима с разбросом самой нейросети по инициализациям "
            "(0,10–0,15 МВт).",
            "3. **В жару устойчивого выигрыша нет:** нейросеть с инициализацией прототипа хуже L1 на 0,2…0,5 МВт (незначимо) "
            "и систематически занижает нагрузку в жару (смещение +3,1…3,5 МВт), а линейная модель с квадратичным членом — "
            "почти без смещения. По 10 инициализациям MAE нейросети в жару 2,50…3,56 МВт (по факту) против 3,01 у L1 — "
            "то лучше, то хуже: в жару результат определяется случайной инициализацией. Для задачи «риск в жаркие часы» "
            "это главное.",
            "4. Ошибка прогноза погоды добавляет 0,1–0,8 МВт к MAE (больше всего — в жару): вклад погоды сравним с разницей "
            "между моделями.", "", "## Ограничения", "",
            "- «Правда» — генератор с известной структурой; честная линейная модель получила члены, похожие на генератор "
            "(порог 24 °C, квадрат выше 30 °C) — на реальной нагрузке структура неизвестна, и выигрыш гибкой модели может быть "
            "больше. Вывод «нейросеть не нужна» на реальные данные переносить нельзя.",
            "- Один год проверки; жарких часов 175; в 2025 г. нет температур выше максимума обучения (39,9 °C) — проверки "
            "экстраполяции нет.",
            "- Температура — среднее трёх точек; нейросеть — архитектура прототипа без подбора."]
    (HERE / "exp_b_report.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
