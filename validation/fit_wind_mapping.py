"""Квантильное сопоставление ветра ERA5 → станции NOAA ISD (шаг 3.3, пункт 3) и его проверка на 2025 г.

Калибровка — 2023–2024 (часы, где есть и ERA5, и сводка станции), проверка — 2025 (ряды станций до 08.2025).
Жанаозен — станции нет: соответствие по объединённой выборке Актау и Бейнеу (допущение).
Сохраняет data/weather/wind_mapping.csv; отчёт — validation/wind_mapping_report.md.

Запуск из корня:  .venv\\Scripts\\python validation\\fit_wind_mapping.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation"))

import check_weather as cw  # noqa: E402
from twin import dlr as L  # noqa: E402

REPORT = Path(__file__).resolve().parent / "wind_mapping_report.md"


def pairs(point: str, sid: str) -> pd.DataFrame:
    era = pd.concat([pd.read_csv(ROOT / "data" / "weather" / "archive" / f"{point}_{y}.csv", parse_dates=["time"])
                     for y in (2023, 2024, 2025)])
    st = cw.station(sid)
    j = era[["time", "wind_ms"]].merge(st[["time", "wind_ms"]].rename(columns={"wind_ms": "st"}), on="time").dropna()
    j["year"] = j["time"].dt.year
    return j


def stats(x, y):
    return {"bias": float(np.mean(x - y)), "mae": float(np.mean(np.abs(x - y))),
            "calm": float(np.mean(x < 1.0)), "calm_obs": float(np.mean(y < 1.0))}


def main() -> int:
    out, rows = [], []

    def say(s=""):
        print(s)
        out.append(s)
    say("# Поправка ветра ERA5 к станциям (квантильное сопоставление)")
    say("")
    say("Калибровка 2023–2024, проверка 2025. «Штиль» — доля часов с ветром < 1 м/с (важно для DLR).")
    say("")
    say("| точка | станция | часов (калибр./пров.) | смещение ERA5 − ст. до → после | MAE до → после | штиль: станция / ERA5 / после |")
    say("|---|---|---|---|---|---|")
    data = {p: pairs(p, sid) for p, sid in L.WIND_STATIONS.items()}
    for p, j in data.items():
        cal, tst = j[j["year"] <= 2024], j[j["year"] == 2025]
        qe, qs = L.fit_wind_mapping(cal["wind_ms"].to_numpy(), cal["st"].to_numpy())
        rows += [{"point": p, "prob": pr, "era5": a, "station": b} for pr, a, b in zip(L.PROBS, qe, qs)]
        m = L.map_wind(tst["wind_ms"].to_numpy(), qe, qs)
        b0, b1 = stats(tst["wind_ms"].to_numpy(), tst["st"].to_numpy()), stats(m, tst["st"].to_numpy())
        say(f"| {p} | {L.WIND_STATIONS[p]} | {len(cal)} / {len(tst)} | {b0['bias']:+.2f} → {b1['bias']:+.2f} м/с | "
            f"{b0['mae']:.2f} → {b1['mae']:.2f} м/с | {100 * b0['calm_obs']:.1f} % / {100 * b0['calm']:.1f} % / {100 * b1['calm']:.1f} % |")
    # Жанаозен — объединённая выборка
    for p, srcs in L.WIND_POOLED.items():
        cal = pd.concat([data[s][data[s]["year"] <= 2024] for s in srcs])
        qe, qs = L.fit_wind_mapping(cal["wind_ms"].to_numpy(), cal["st"].to_numpy())
        rows += [{"point": p, "prob": pr, "era5": a, "station": b} for pr, a, b in zip(L.PROBS, qe, qs)]
        say(f"| {p} | нет (объединено: {', '.join(srcs)}) | {len(cal)} / — | проверить нельзя | — | — |")
    pd.DataFrame(rows).to_csv(L.WIND_MAP, index=False, float_format="%.4f")
    say("")
    say("Ограничения: станции измеряют ветер на 10 м, провода висят выше (10–25 м) — поправка по высоте не делается "
        "(оценка осторожная); сводки SYNOP округлены до целых м/с; Жанаозен без станции.")
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
