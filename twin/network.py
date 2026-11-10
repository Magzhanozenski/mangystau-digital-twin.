"""Загрузка модели сети из data/network.yaml и расчёт параметров элементов."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "network.yaml"


@dataclass
class Network:
    base_mva: float
    base_kv: float
    freq: float
    conductors: dict
    buses: list
    lines: list
    gens: list
    model: dict
    raw: dict = field(repr=False, default_factory=dict)

    @property
    def zb(self) -> float:
        """Базисное сопротивление, Ом."""
        return self.base_kv ** 2 / self.base_mva

    def bus(self, bid: int) -> dict:
        return next(b for b in self.buses if b["id"] == bid)

    def line(self, lid: str) -> dict:
        return next(l for l in self.lines if l["id"] == lid)

    def gen(self, gid: str) -> dict:
        return next(g for g in self.gens if g["id"] == gid)

    # --- параметры линий ---
    def resistance_factor(self, T: float) -> float:
        """Поправка сопротивления на нагрев провода: (1 + α·(T_провода − 20))."""
        m = self.model
        return 1 + m["alpha_al_per_c"]["value"] * (T + m["conductor_overheat_c"]["value"] - 20)

    def line_pu(self, line: dict, T: float) -> tuple[float, float, float]:
        """R, X, B (полная ёмкостная проводимость) линии в о.е. при температуре воздуха T."""
        c = self.conductors[line["conductor"]]
        R = c["r_ohm_km"] * self.resistance_factor(T) * line["km"] / self.zb
        X = c["x_ohm_km"] * line["km"] / self.zb
        B = c["b_s_km"] * line["km"] * self.zb
        return R, X, B

    def static_rating_a(self, line: dict, T: float) -> float:
        """Справочный допустимый ток с поправкой на температуру воздуха."""
        s = self.model["static_rating"]
        k = math.sqrt(max(0.05, (s["conductor_t_max_c"] - T) / (s["conductor_t_max_c"] - s["reference_air_c"])))
        return self.conductors[line["conductor"]]["i_rated_a"] * min(s["max_factor"], k)

    def tan_phi(self, T: float) -> float:
        r = self.model["reactive_load"]
        return r["tan_phi_base"] + r["tan_phi_per_c_above_25"] * max(0.0, T - 25)

    def default_ctrl(self) -> dict:
        return {
            "gP": {g["id"]: g["p_mw"] for g in self.gens if not g["slack"]},
            "vset": {g["id"]: g["v_pu"] for g in self.gens},
            "bsk": {},
            "shed": {},
        }


def load_network(path: str | Path = DEFAULT_PATH) -> Network:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for section in ("buses", "lines", "generators"):
        for item in raw[section]:
            if item.get("status") not in ("источник", "допущение"):
                raise ValueError(f"{section}: у элемента {item.get('id')} нет пометки status (источник/допущение)")
            if item["status"] == "источник" and not item.get("source"):
                raise ValueError(f"{section}: у элемента {item['id']} status=источник, но нет поля source")
    m = raw["meta"]
    return Network(
        base_mva=m["base_mva"], base_kv=m["base_kv"], freq=m["frequency_hz"],
        conductors=raw["conductors"], buses=raw["buses"], lines=raw["lines"],
        gens=raw["generators"], model=raw["model"], raw=raw,
    )
