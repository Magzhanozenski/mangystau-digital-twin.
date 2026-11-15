"""Проверка сети по открытым источникам data/network_real.yaml (шаг 2.2, пункт 5).

Что проверяется:
  1) файл грузится load_network() (пометки status/source у всех элементов);
  2) ядро twin.powerflow сходится в базовом режиме и в нескольких уровнях нагрузки;
  3) pandapower на той же постановке совпадает с ядром: |ΔU| < 1e-4 о.е., |ΔP| < 0,01 МВт.
Если не сходится или не совпадает — печатаем причину и выходим с кодом 1 (не подгоняем).
  Отчёт (напряжения по классам, зарядная мощность, загрузка трансформаторов) — validation/check_network_real_report.md.

Запуск из корня:  .venv\\Scripts\\python validation\\check_network_real.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import load_network, run_power_flow  # noqa: E402
from twin.pp_model import run_pp  # noqa: E402

PATH = ROOT / "data" / "network_real.yaml"
REPORT = Path(__file__).resolve().parent / "check_network_real_report.md"
TOL_V, TOL_P = 1e-4, 0.01
# Опорные режимы нагрузки — из meta.load_scenarios (шаг 2.3, load_estimates.yaml); температуры — ДОПУЩЕНИЕ:
# минимум — ночь +10 °C, база +25 °C, пик — летняя жара +40 °C.
SCENARIO_T = {"min": 10, "base": 25, "peak": 40}


def scenarios(net):
    """Режимы min/base/peak. Загрузка агрегатов (p_mw задана при средней нагрузке) масштабируется под нагрузку
    режима — это статическая проверка сети, не суточный график (его строит twin/dispatch.py)."""
    ls = net.raw["meta"]["load_scenarios"]
    out = []
    for k, t in SCENARIO_T.items():
        ctrl = net.default_ctrl()
        ctrl["gP"] = {g: p * ls[k] / ls["base"] for g, p in ctrl["gP"].items()}
        # состав 2023 г.: агрегаты сценариев (ГПЭС Жанаозен, 2026) отключены
        out.append({"name": k, "T": t, "load": ls[k], "ctrl": ctrl,
                    "gensOut": [g["id"] for g in net.gens if g.get("scenario")]})
    return out


def main() -> int:
    net = load_network(PATH)
    out = []

    def say(s=""):
        print(s)
        out.append(s)

    say("# Проверка network_real.yaml: ядро против pandapower")
    say("")
    say(f"{PATH.name}: шин {len(net.buses)}, линий (цепей) {len(net.lines)}, трансформаторов {len(net.trafos)}, "
        f"генераторов {len(net.gens)}. Допуск: |ΔU| < {TOL_V} о.е., |ΔP| < {TOL_P} МВт.")
    lim = net.model["voltage_limits_pu"]
    bad = 0
    for scn in scenarios(net):
        r = run_power_flow(scn, net)
        live = r["live"]
        say("")
        say(f"## Режим «{scn['name']}»: T = {scn['T']} °C, нагрузка {scn['load']:.1f} МВт")
        say(f"- островов {len(r['islands'])}, живых шин {len(live)} из {len(net.buses)}, "
            f"непокрыто {r['unserved']:.3f} МВт, АЧР {r['achr']:.3f} МВт")
        if len(live) < len(net.buses):
            say("- ВНИМАНИЕ: не все шины под напряжением")
        loss = sum(f.get("loss", 0) for f in r["flows"].values() if not f.get("out"))
        say(f"- U: {min(r['V'][b] for b in live):.4f} … {max(r['V'][b] for b in live):.4f} о.е.; потери {loss:.3f} МВт; "
            f"нарушений {len(r['violations'])}")
        by_kv = {}
        for b in live:
            v, kv = r["V"][b], net.bus_kv(b)
            d = by_kv.setdefault(kv, [0, 0, 0, 1e9, 0])
            d[0] += 1
            d[1] += v > lim["crit_high"]
            d[2] += v < lim["crit_low"]
            d[3], d[4] = min(d[3], v), max(d[4], v)
        for kv in sorted(by_kv, reverse=True):
            n, hi, lo, vmin, vmax = by_kv[kv]
            say(f"  - {kv:g} кВ: шин {n}, U {vmin:.3f} … {vmax:.3f} о.е., выше {lim['crit_high']}: {hi}, ниже {lim['crit_low']}: {lo}")
        charging = {}
        for l in net.lines:
            f = r["flows"][l["id"]]
            if f.get("out") or f.get("dead"):
                continue
            _, _, B = net.line_pu(l, scn["T"])
            charging[net.bus_kv(l["from"])] = charging.get(net.bus_kv(l["from"]), 0) + \
                B / 2 * (r["V"][l["from"]] ** 2 + r["V"][l["to"]] ** 2) * net.base_mva
        say("- зарядная мощность линий: " + ", ".join(f"{kv:g} кВ — {q:.0f} Мвар" for kv, q in sorted(charging.items(), reverse=True)))
        for g, x in r["gen"].items():
            say(f"- {g}: P = {x['P']:.1f} МВт, Q = {x['Q']:.1f} Мвар" + (" (на пределе Q)" if x.get("atQlim") else ""))
        tr = sorted(((f["loading"], k) for k, f in r["flows"].items() if f.get("kind") == "trafo" and not f.get("out")), reverse=True)
        ln = sorted(((f["loading"], k) for k, f in r["flows"].items() if f.get("kind") != "trafo" and not f.get("out") and not f.get("dead")), reverse=True)
        say("- загрузка трансформаторов (макс.): " + ", ".join(f"{k} {100 * v:.0f} %" for v, k in tr[:4]))
        say("- загрузка линий (макс.): " + ", ".join(f"{k} {100 * v:.0f} %" for v, k in ln[:3]))
        try:
            p = run_pp(scn, net, r["islands"])
        except Exception as e:  # не скрываем несходимость pandapower
            say(f"- pandapower: ОШИБКА {e!r}")
            bad += 1
            continue
        dv = max(abs(p["V"][b] - r["V"][b]) for b in live)
        dp = max(abs(p["flows"][l]["P"] - f["P"]) for l, f in r["flows"].items() if not f.get("out") and not f.get("dead"))
        dl = max([abs(p["flows"][k]["loading"] - f["loading"]) for k, f in r["flows"].items()
                  if f.get("kind") == "trafo" and not f.get("out") and not f.get("dead")] or [0.0])
        ok = dv < TOL_V and dp < TOL_P
        say(f"- ядро vs pandapower: |ΔU| = {dv:.2e} о.е., |ΔP| = {dp:.2e} МВт, загрузка трансформаторов |Δ| = {dl:.1e} — "
            f"{'совпадает' if ok else 'ВНЕ ДОПУСКА'}")
        bad += 0 if ok else 1
    say("")
    say("Итог: " + ("все режимы совпадают" if not bad else f"{bad} режим(ов) с ошибкой или вне допуска"))
    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
