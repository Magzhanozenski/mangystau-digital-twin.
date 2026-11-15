"""Динамический допустимый ток провода (DLR) по тепловому балансу IEEE 738 (шаг 3.3).

Установившийся режим:  q_c + q_r = q_s + I²·R(T_c)  →  I = √((q_c + q_r − q_s) / R(T_c)).
  q_c — конвекция: max(вынужденная при малых и больших Re (с поправкой на угол ветра к оси провода), естественная);
  q_r — излучение; q_s — нагрев солнцем; R(T_c) — сопротивление при температуре провода.
Свойства воздуха (вязкость, плотность с учётом высоты, теплопроводность) — формулы IEEE 738.

Упрощения (ДОПУЩЕНИЯ, записаны в decisions.md):
  - солнце: q_s = α·GHI·D — вся глобальная горизонтальная радиация как перпендикулярная проводу (оценка сверху);
  - сопротивление: R20 по ГОСТ 839-80 (постоянный ток) × (1 + 0,00403·(T_c − 20)), без поправки на переменный ток;
  - ε = α = 0,5 (провода неизвестного возраста), T_c max = 70 °C (ПУЭ, решение пользователя).
Допустимый ток линии — по самому слабому пролёту: угол ветра считается для каждого отрезка трассы OSM.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
FACTS = ROOT / "data" / "network_sources" / "dlr_facts.yaml"

TC_MAX = 70.0       # ПУЭ п. 1.3.22 (источник)
EMISSIVITY = 0.5    # ДОПУЩЕНИЕ
ABSORPTIVITY = 0.5  # ДОПУЩЕНИЕ
ALPHA_AL = 0.00403  # температурный коэффициент сопротивления алюминия, 1/°C


def _conductors() -> dict:
    d = yaml.safe_load(FACTS.read_text(encoding="utf-8"))
    out = {}
    for f in d["facts"]:
        if f["object"]["type"] == "провод":
            v = f["value"]
            out[f["object"]["name"].split("/")[0]] = {"D": v["diameter_mm"] / 1000, "R20": v["r20_ohm_km"] / 1000}
    pue = next(f for f in d["facts"] if f["object"]["name"] == "ПУЭ, табл. 1.3.29")["value"]
    for k, i in pue.items():
        out[k.split("/")[0]]["I_pue"] = float(i)
    corr = next(f for f in d["facts"] if f["object"]["name"] == "ПУЭ, табл. 1.3.3")["value"]
    return out, {float(k): float(v) for k, v in corr.items()}


CONDUCTORS, PUE_CORR = _conductors()   # {"АС-185": {D, R20, I_pue}, "АС-300": {...}}


def k_angle(phi_deg):
    """Поправка IEEE 738 на угол φ между ветром и осью провода (φ = 90° — поперёк, 1,0)."""
    p = np.radians(phi_deg)
    return 1.194 - np.cos(p) + 0.194 * np.cos(2 * p) + 0.368 * np.sin(2 * p)


def ampacity(cond: str, Ta, wind, phi_deg=90.0, solar=0.0, Tc: float = TC_MAX, elevation_m: float = 0.0,
             eps: float = EMISSIVITY, alpha: float = ABSORPTIVITY):
    """Допустимый ток, А. Ta — °C, wind — м/с, phi_deg — угол ветра к оси провода, solar — Вт/м² (горизонтальная)."""
    c = CONDUCTORS[cond]
    D = c["D"]
    Ta = np.asarray(Ta, dtype=float)
    V = np.maximum(np.asarray(wind, dtype=float), 0.0)
    Tf = (Tc + Ta) / 2
    mu = 1.458e-6 * (Tf + 273) ** 1.5 / (Tf + 383.4)
    rho = (1.293 - 1.525e-4 * elevation_m + 6.379e-9 * elevation_m ** 2) / (1 + 0.00367 * Tf)
    kf = 2.424e-2 + 7.477e-5 * Tf - 4.407e-9 * Tf ** 2
    dT = np.maximum(Tc - Ta, 0.0)
    Re = D * rho * V / mu
    ka = k_angle(phi_deg)
    qc1 = ka * (1.01 + 1.35 * Re ** 0.52) * kf * dT
    qc2 = ka * 0.754 * Re ** 0.6 * kf * dT
    qcn = 3.645 * np.sqrt(rho) * D ** 0.75 * dT ** 1.25
    qc = np.maximum(np.maximum(qc1, qc2), qcn)
    qr = 17.8 * D * eps * (((Tc + 273) / 100) ** 4 - ((Ta + 273) / 100) ** 4)
    qs = alpha * np.maximum(np.asarray(solar, dtype=float), 0.0) * D
    R = c["R20"] * (1 + ALPHA_AL * (Tc - 20))
    return np.sqrt(np.maximum(qc + qr - qs, 0.0) / R)


def static_pue(cond: str, Ta) -> np.ndarray:
    """Справочный ток по ПУЭ: табл. 1.3.29 × поправка табл. 1.3.3 (линейная интерполяция по температуре воздуха)."""
    t = sorted(PUE_CORR)
    return CONDUCTORS[cond]["I_pue"] * np.interp(np.asarray(Ta, dtype=float), t, [PUE_CORR[x] for x in t])


def wind_angle(wind_dir_deg, span_az_deg):
    """Угол (0…90°) между направлением ветра (откуда дует, метеорологическое) и осью пролёта (азимут 0…180)."""
    d = np.abs((np.asarray(wind_dir_deg, dtype=float) - span_az_deg) % 180.0)
    return np.where(d > 90, 180 - d, d)


def span_azimuths(coords: list) -> tuple[np.ndarray, np.ndarray]:
    """Азимуты (0…180°) и длины (км) отрезков трассы [(lat, lon), …]."""
    az, ln = [], []
    for (la1, lo1), (la2, lo2) in zip(coords, coords[1:]):
        p1, p2 = math.radians(la1), math.radians(la2)
        dl = math.radians(lo2 - lo1)
        y = math.sin(dl) * math.cos(p2)
        x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
        az.append(math.degrees(math.atan2(y, x)) % 180.0)
        a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        ln.append(2 * 6371.0088 * math.asin(math.sqrt(a)))
    return np.array(az), np.array(ln)


def line_rating(cond: str, az: np.ndarray, Ta, wind, wind_dir, solar, elevation_m: float = 0.0,
                bin_deg: float = 1.0) -> np.ndarray:
    """Допустимый ток линии по самому слабому пролёту для рядов погоды (по часам). az — азимуты пролётов."""
    uniq = np.unique(np.round(az / bin_deg) * bin_deg)
    Ta, wind, wind_dir, solar = (np.asarray(x, dtype=float) for x in (Ta, wind, wind_dir, solar))
    out = np.full(Ta.shape, np.inf)
    for a in uniq:
        out = np.minimum(out, ampacity(cond, Ta, wind, wind_angle(wind_dir, a), solar, elevation_m=elevation_m))
    return out
