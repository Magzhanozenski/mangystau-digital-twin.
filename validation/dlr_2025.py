"""Почасовой DLR линий network_real за 2025 г. против справочного тока ПУЭ (шаг 3.3, пункт 4).

Погода — ERA5 ближайшей из трёх точек (Актау, Жанаозен, Бейнеу), ветер — с поправкой к станциям (wind_mapping.csv).
Справочный ток — ПУЭ табл. 1.3.29 × поправка табл. 1.3.3 по температуре воздуха.
Показывается отношение DLR / ПУЭ и доля часов, когда DLR ниже справочного (справочный ток в эти часы завышен —
скрытый риск); отдельно — жаркие часы (ERA5 ≥ 35 °C) и влияние двух уточнений: поправки ветра и угла по пролётам
(против прототипа: ветер ERA5 без поправки, угол 45° для всех линий).
Отчёт — validation/dlr_2025_report.md.

Запуск из корня:  .venv\\Scripts\\python validation\\dlr_2025.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import dlr as L  # noqa: E402

REPORT = Path(__file__).resolve().parent / "dlr_2025_report.md"


def md_table(df: pd.DataFrame) -> str:
    """Таблица Markdown без внешних зависимостей."""
    def f(x):
        return f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(f(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def summary(df, info, temps):
    rows = []
    for key, inf in info.items():
        T = temps[inf["point"]]
        dlr = df[key].to_numpy()
        st = L.static_pue(inf["conductor"], T)
        r = dlr / st
        hot = T >= 35
        rows.append({"линия": key, "провод": inf["conductor"], "км": round(inf["km"], 1), "погода": inf["point"],
                     "до точки, км": inf["dist_km"], "пролётов": inf["spans"],
                     "DLR/ПУЭ P5": np.quantile(r, 0.05), "P50": np.median(r), "P95": np.quantile(r, 0.95),
                     "часов DLR < ПУЭ, %": 100 * np.mean(r < 1), "жара: часов": int(hot.sum()),
                     "жара: DLR < ПУЭ, %": 100 * np.mean(r[hot] < 1) if hot.any() else np.nan,
                     "мин. DLR, А": float(dlr.min())})
    return pd.DataFrame(rows)


def main() -> int:
    raw = yaml.safe_load((ROOT / "data" / "network_real.yaml").read_text(encoding="utf-8"))
    base, info, temps = L.line_series(raw)
    nocorr, _, _ = L.line_series(raw, correct_wind=False)
    proto, _, _ = L.line_series(raw, correct_wind=False, fixed_angle=45.0)
    s = summary(base, info, temps)
    out = ["# DLR линий network_real за 2025 г. против справочного тока ПУЭ", "",
           "DLR — IEEE 738 (T провода 70 °C, ε = α = 0,5), погода ERA5 ближайшей точки, ветер с поправкой к станциям, "
           "угол ветра — по каждому пролёту OSM, ток линии — по самому слабому пролёту. ПУЭ — табл. 1.3.29 × табл. 1.3.3.", ""]
    allr = np.concatenate([base[k].to_numpy() / L.static_pue(info[k]["conductor"], temps[info[k]["point"]]) for k in info])
    hot = np.concatenate([temps[info[k]["point"]] >= 35 for k in info])
    out += ["## Сводка по всем линиям и часам", "",
            f"- отношение DLR / ПУЭ: P5 {np.quantile(allr, 0.05):.2f}, медиана {np.median(allr):.2f}, P95 {np.quantile(allr, 0.95):.2f}",
            f"- часов, когда DLR ниже справочного тока ПУЭ: {100 * np.mean(allr < 1):.1f} % (линия·часов {allr.size})",
            f"- в жару (ERA5 ≥ 35 °C): {100 * np.mean(allr[hot] < 1):.1f} % из {int(hot.sum())} линия·часов",
            ""]
    out += ["## Влияние уточнений (доля линия·часов с DLR < ПУЭ и медиана DLR/ПУЭ)", "",
            "| вариант | DLR < ПУЭ, % | медиана DLR/ПУЭ |", "|---|---|---|"]
    for name, df in (("шаг 3.3: ветер к станциям, угол по пролётам", base),
                     ("ветер ERA5 без поправки, угол по пролётам", nocorr),
                     ("как в прототипе: ветер ERA5, угол 45° для всех", proto)):
        r = np.concatenate([df[k].to_numpy() / L.static_pue(info[k]["conductor"], temps[info[k]["point"]]) for k in info])
        out.append(f"| {name} | {100 * np.mean(r < 1):.1f} | {np.median(r):.2f} |")
    out += ["", "## По линиям (отсортировано по доле часов DLR < ПУЭ)", "",
            md_table(s.sort_values("часов DLR < ПУЭ, %", ascending=False)), "",
            "Ограничения: погода — одна из трёх точек, до части линий 150–190 км (остров Бузачи — по Актау); "
            "ветер на 10 м без поправки по высоте подвеса; солнце — оценка сверху (вся GHI перпендикулярно проводу); "
            "ε и α — допущения; ERA5 занижает жару в Актау на 1,5–1,8 °C (шаг 2.1) — доля опасных часов, вероятно, занижена."]
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
