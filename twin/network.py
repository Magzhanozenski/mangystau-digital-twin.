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
    trafos: list = field(default_factory=list)

    @property
    def zb(self) -> float:
        """Базисное сопротивление, Ом."""
        return self.base_kv ** 2 / self.base_mva

    def bus(self, bid: int) -> dict:
        return next(b for b in self.buses if b["id"] == bid)

    def bus_kv(self, bid: int) -> float:
        """Номинальное напряжение узла, кВ (поле vn_kv; по умолчанию — базисное напряжение сети)."""
        return float(self.bus(bid).get("vn_kv", self.base_kv))

    def zb_kv(self, kv: float) -> float:
        """Базисное сопротивление ступени напряжения kv, Ом."""
        return kv ** 2 / self.base_mva

    def trafo(self, tid: str) -> dict:
        return next(t for t in self.trafos if t["id"] == tid)

    def trafo_pu(self, t: dict) -> tuple[complex, complex]:
        """Двухобмоточный трансформатор, П-схема, о.е. на базе сети: (продольная проводимость, поперечная
        проводимость намагничивания целиком). Номинальные напряжения обмоток равны напряжениям шин
        (коэффициент трансформации 1 о.е., РПН не моделируется)."""
        sn, k = t["sn_mva"], self.base_mva / t["sn_mva"]
        z = t["uk_pct"] / 100 * k
        r = t["pk_kw"] / 1000 / sn * k
        x = (z * z - r * r) ** 0.5
        g = t.get("p0_kw", 0.0) / 1000 / sn
        ym = t.get("i0_pct", 0.0) / 100
        b = -max(ym * ym - g * g, 0.0) ** 0.5
        return 1 / complex(r, x), complex(g, b) / k

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
        zb = self.zb_kv(self.bus_kv(line["from"]))
        R = c["r_ohm_km"] * self.resistance_factor(T) * line["km"] / zb
        X = c["x_ohm_km"] * line["km"] / zb
        B = c["b_s_km"] * line["km"] * zb
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
    for section in ("buses", "lines", "generators", "transformers"):
        for item in raw.get(section) or []:
            if item.get("status") not in ("источник", "допущение"):
                raise ValueError(f"{section}: у элемента {item.get('id')} нет пометки status (источник/допущение)")
            if item["status"] == "источник" and not item.get("source"):
                raise ValueError(f"{section}: у элемента {item['id']} status=источник, но нет поля source")
    m = raw["meta"]
    net = Network(
        base_mva=m["base_mva"], base_kv=m["base_kv"], freq=m["frequency_hz"],
        conductors=raw["conductors"], buses=raw["buses"], lines=raw["lines"],
        gens=raw["generators"], model=raw["model"], raw=raw, trafos=raw.get("transformers") or [],
    )
    for l in net.lines:
        if net.bus_kv(l["from"]) != net.bus_kv(l["to"]):
            raise ValueError(f"линия {l['id']} соединяет узлы разного напряжения — нужен трансформатор")
    for t in net.trafos:
        if net.bus_kv(t["from"]) <= net.bus_kv(t["to"]):
            raise ValueError(f"трансформатор {t['id']}: from — сторона ВН, to — НН")
    return net
