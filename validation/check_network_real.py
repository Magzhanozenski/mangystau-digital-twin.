"""Проверка сети по открытым источникам data/network_real.yaml (шаг 2.2, пункт 5).

Что проверяется:
  1) файл грузится load_network() (пометки status/source у всех элементов);
  2) ядро twin.powerflow сходится в базовом режиме и в нескольких уровнях нагрузки;
  3) pandapower на той же постановке совпадает с ядром: |ΔU| < 1e-4 о.е., |ΔP| < 0,01 МВт.
Если не сходится или не совпадает — печатаем причину и выходим с кодом 1 (не подгоняем).

Запуск из корня:  .venv\\Scripts\\python validation\\check_network_real.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import load_network, run_power_flow  # noqa: E402
from twin.pp_model import run_pp  # noqa: E402

PATH = ROOT / "data" / "network_real.yaml"
TOL_V, TOL_P = 1e-4, 0.01
BASE_LOAD_MW = 60  # ДОПУЩЕНИЕ: суммарная нагрузка ПС модели в базовом режиме (реальная нагрузка — шаг 2.3)
SCENARIOS = [{"T": 25, "load": BASE_LOAD_MW}, {"T": 25, "load": 30}, {"T": 40, "load": 100}]


def main() -> int:
    net = load_network(PATH)
    print(f"{PATH.name}: шин {len(net.buses)}, линий (цепей) {len(net.lines)}, эквивалентов {len(net.gens)}")
    bad = 0
    for scn in SCENARIOS:
        r = run_power_flow(scn, net)
        live = r["live"]
        print(f"\nT={scn['T']} °C, нагрузка {scn['load']} МВт: островов {len(r['islands'])}, "
              f"живых шин {len(live)} из {len(net.buses)}, непокрыто {r['unserved']:.3f} МВт, АЧР {r['achr']:.3f} МВт")
        for isl in r["islands"]:
            print(f"   остров: балансирующий {isl['slack']}, шин {len(isl['ids'])}")
        if len(live) < len(net.buses):
            print("   ВНИМАНИЕ: не все шины под напряжением — см. острова")
        vmin = min(r["V"][b] for b in live)
        vmax = max(r["V"][b] for b in live)
        loss = sum(f.get("loss", 0) for f in r["flows"].values() if not f.get("out"))
        print(f"   U: {vmin:.4f} … {vmax:.4f} о.е.; потери {loss:.3f} МВт; нарушений {len(r['violations'])}")
        for g, x in r["gen"].items():
            print(f"   {g}: P={x['P']:.3f} МВт, Q={x['Q']:.3f} Мвар")
        try:
            p = run_pp(scn, net, r["islands"])
        except Exception as e:  # не скрываем несходимость pandapower
            print(f"   pandapower: ОШИБКА {e!r}")
            bad += 1
            continue
        dv = max(abs(p["V"][b] - r["V"][b]) for b in live)
        dp = max(abs(p["flows"][l]["P"] - f["P"]) for l, f in r["flows"].items() if not f.get("out") and not f.get("dead"))
        ok = dv < TOL_V and dp < TOL_P
        print(f"   ядро vs pandapower: |ΔU| = {dv:.2e} о.е., |ΔP| = {dp:.2e} МВт — {'совпадает' if ok else 'ВНЕ ДОПУСКА'}")
        bad += 0 if ok else 1
    print("\nИтог:", "все режимы совпадают" if not bad else f"{bad} режим(ов) с ошибкой или вне допуска")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
