"""Сравнение условной сети (data/network.yaml) с сетью по открытым источникам (шаг 2.2, пункт 7).

1) ПС условной сети (8): есть ли у названного места ПС в OSM и вошла ли она в data/network_real.yaml.
2) Линии условной сети (11): есть ли связь между этими местами
     а) в модели network_real.yaml (только линии с тегом 110 кВ);
     б) в OSM по линиям 110 кВ и линиям без тега напряжения (35 и 220 кВ не берутся) — кратчайший путь,
        его длина и доля без тега. Путь б) — ориентир: линии без тега могут оказаться 35 кВ.
   Разница длин: путь − условная длина.
3) Доли «источник» / «допущение» по полям: длины, провода, цепи, нагрузки, генерация — для обеих сетей.

Сопоставление «узел условной сети → место» — ДОПУЩЕНИЕ (таблица PLACES ниже, основание у каждой строки).
Отчёт печатается и сохраняется в validation/compare_network_report.md.

Запуск из корня:  .venv\\Scripts\\python validation\\compare_network.py
"""
import csv
import heapq
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from twin import build_real as br  # noqa: E402

SRC = ROOT / "data" / "network_sources"
REPORT = Path(__file__).resolve().parent / "compare_network_report.md"

# узел network.yaml → (места OSM, радиус км, конкретные ПС OSM, основание). ДОПУЩЕНИЕ.
PLACES = {
    1: ([], 0, ["way/224369643"], "МАЭК = ТЭС МАЭК (ТЭЦ-3): имя ТЭЦ-3 в OSM"),
    2: (["Актау"], 5, [], "ПС города — ПС OSM, ближайший посёлок которых Актау, не дальше 5 км"),
    3: ([], 0, [], "«Промзона» — место не определено: ПС 110/6 «Промзона» есть в отчёте ККС 2019, адреса нет"),
    4: (["Форт-Шевченко"], 5, [], "ПС у Форт-Шевченко, не дальше 5 км"),
    5: (["Жетыбай", "Мунайшы"], 15, ["way/279523559"], "ПС у Жетыбая (15 км) и «РЭС Жетыбай» (имя в OSM, у Мунайшы)"),
    6: (["Жанаозен"], 15, [], "ПС у Жанаозена, не дальше 15 км (в т. ч. ПС 220/110 «Узень», опознание — средняя)"),
    7: (["Шетпе"], 15, [], "ПС у Шетпе, не дальше 15 км"),
    8: (["Бейнеу"], 5, [], "ПС у Бейнеу, не дальше 5 км (ПС 220 «Бейнеу» и 110/35/6 «Бейнеу», опознание — средняя)"),
}


def _csv(name):
    with open(SRC / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def candidates(bid: int, subs: list[dict]) -> list[str]:
    places, km, explicit, _ = PLACES[bid]
    out = list(explicit)
    for s in subs:
        if s["nearest_place"] in places and float(s["nearest_place_km"]) <= km and s["osm"] not in out:
            out.append(s["osm"])
    return out


def shortest(edges: list[dict], src: set, dst: set):
    """Дейкстра по рёбрам графа: (км, длина по классам, число рёбер) или None."""
    adj = defaultdict(list)
    for e in edges:
        adj[e["p"]].append((e["q"], e))
        adj[e["q"]].append((e["p"], e))
    start = [("s", x) for x in src if ("s", x) in adj]
    goal = {("s", x) for x in dst}
    dist, prev, heap = {}, {}, []
    for s in start:
        dist[s] = 0.0
        heapq.heappush(heap, (0.0, str(s), s))
    while heap:
        d, _, u = heapq.heappop(heap)
        if d > dist.get(u, float("inf")):
            continue
        if u in goal:
            kv, n, x = defaultdict(float), 0, u
            while x in prev:
                x, e = prev[x]
                n += 1
                for k, v in e["kv_km"].items():
                    kv[k] += v
            return d, dict(kv), n
        for v, e in adj[u]:
            nd = d + e["km"]
            if nd < dist.get(v, float("inf")):
                dist[v], prev[v] = nd, (u, e)
                heapq.heappush(heap, (nd, str(v), v))
    return None


def model_edges(real: dict) -> list[dict]:
    """Рёбра модели network_real.yaml (одна запись на линию, без дублей цепей)."""
    osm_of = {b["id"]: b["osm"] for b in real["buses"]}
    seen, out = set(), []
    for l in real["lines"]:
        key = l["id"].rsplit("-", 1)[0]
        if key in seen:
            continue
        seen.add(key)
        p = ("s", osm_of[l["from"]]) if not osm_of[l["from"]].startswith("node/") else ("n", osm_of[l["from"]])
        q = ("s", osm_of[l["to"]]) if not osm_of[l["to"]].startswith("node/") else ("n", osm_of[l["to"]])
        out.append({"p": p, "q": q, "km": l["km"], "kv_km": {"110.0": l["km"]}})
    return out


def field_shares(net: dict) -> dict:
    """Доли «источник» по полям. Для network.yaml пометка одна на элемент — она и берётся для всех полей."""
    def st(item, field):
        return item.get("marks", {}).get(field, {}).get("status", item["status"])
    lines, gens = net["lines"], net["generators"]
    loads = [b for b in net["buses"] if b["share"] > 0]
    km = sum(l["km"] for l in lines)

    def by_km(field):
        return sum(l["km"] for l in lines if st(l, field) == "источник") / km if km else None

    def by_cnt(items, field):
        return sum(1 for x in items if st(x, field) == "источник") / len(items) if items else None
    return {"длины (по км)": by_km("km"), "провода (по км)": by_km("conductor"),
            "цепи (по км)": by_km("circuits") if any("marks" in l for l in lines) else None,  # в network.yaml поля нет
            "нагрузки (по ПС)": by_cnt(loads, "share"),
            "генерация (по МВт)": (sum(g["p_max_mw"] for g in gens if st(g, "p_max_mw") == "источник")
                                   / sum(g["p_max_mw"] for g in gens)) if gens else None}


def pct(x):
    return "—" if x is None else f"{100 * x:.0f} %"


def main() -> int:
    cond = yaml.safe_load((ROOT / "data" / "network.yaml").read_text(encoding="utf-8"))
    real = yaml.safe_load((ROOT / "data" / "network_real.yaml").read_text(encoding="utf-8"))
    subs, segs, ids = _csv("osm_substations.csv"), _csv("osm_segments.csv"), {r["osm"]: r for r in _csv("substation_ids.csv")}
    snap = json.loads(br.latest_snapshot().read_text(encoding="utf-8"))
    in_model = {b["osm"] for b in real["buses"]}
    osm_edges, _ = br.graph_110(snap, segs, keep=lambda s: s["kv"] in ("110.0", ""))
    all_edges, _ = br.graph_110(snap, segs, keep=lambda s: True)  # все классы, вкл. 35 и 220 кВ
    m_edges = model_edges(real)

    out = ["# Сравнение условной сети с сетью по открытым источникам", "",
           f"Условная: data/network.yaml ({len(cond['buses'])} ПС, {len(cond['lines'])} линий). "
           f"Реальная: data/network_real.yaml (снимок {real['meta']['osm_snapshot']}, вариант A — только тег 110 кВ).",
           "Сопоставление узлов с местами — допущение (PLACES в скрипте).", "",
           "## 1. Подстанции", "",
           "| № | Условная ПС | ПС в OSM у места | в модели network_real | опознание (лучшее) | основание сопоставления |",
           "|---|---|---|---|---|---|"]
    cand = {}
    rank = {"высокая": 0, "средняя": 1, "низкая": 2, "не опознана": 3}
    n_found = n_model = 0
    for b in cond["buses"]:
        c = candidates(b["id"], subs)
        cand[b["id"]] = c
        inm = [x for x in c if x in in_model]
        best = min((ids[x] for x in c), key=lambda r: rank[r["confidence"]], default=None)
        n_found += bool(c)
        n_model += bool(inm)
        out.append(f"| {b['id']} | {b['name']} | {len(c) or 'нет'} | {'да (' + str(len(inm)) + ')' if inm else 'нет'} | "
                   f"{(best['match_name'] + ' — ' + best['confidence']) if best else '—'} | {PLACES[b['id']][3]} |")
    out += ["", f"Итог: ПС у места найдены в OSM для {n_found} из {len(cond['buses'])}; в модель вошли {n_model}.", "",
            "## 2. Линии", "",
            "Путь «OSM» — кратчайший по линиям 110 кВ и линиям без тега напряжения; это ориентир, не подтверждённая связь.", "",
            "| Линия | Условная, км | в модели: путь, км | OSM 110 + без тега: путь, км | из них без тега | Δ (OSM − условная), км | OSM все классы (вкл. 35/220) |",
            "|---|---|---|---|---|---|---|"]
    n_model_l = n_osm_l = 0
    rows = []
    for l in cond["lines"]:
        a, b = set(cand[l["from"]]), set(cand[l["to"]])
        pm = shortest(m_edges, a, b) if a and b else None
        po = shortest(osm_edges, a, b) if a and b else None
        pa = shortest(all_edges, a, b) if a and b else None
        pa_txt = "нет" if not pa else f"{pa[0]:.1f} км: " + ", ".join(
            f"{'без тега' if k == '' else k.split('.')[0] + ' кВ'} {v:.0f}" for k, v in sorted(pa[1].items()))
        n_model_l += bool(pm)
        n_osm_l += bool(po)
        unk = po[1].get("", 0.0) if po else None
        rows.append({"id": l["id"], "cond_km": l["km"], "model_km": pm and round(pm[0], 1),
                     "osm_km": po and round(po[0], 1), "osm_unknown_km": unk and round(unk, 1)})
        why = "" if a and b else " (место не определено)"
        out.append(f"| {l['id']} {l['name']}{why} | {l['km']} | {f'{pm[0]:.1f}' if pm else 'нет'} | "
                   f"{f'{po[0]:.1f}' if po else 'нет'} | {f'{unk:.1f} км ({100 * unk / po[0]:.0f} %)' if po and po[0] else '—'} | "
                   f"{f'{po[0] - l['km']:+.1f}' if po else '—'} | {pa_txt if a and b else '—'} |")
    out += ["", f"Итог: связь в модели есть у {n_model_l} из {len(cond['lines'])} условных линий; "
            f"путь в OSM (с линиями без тега) — у {n_osm_l}.", "",
            "## 3. Доли «источник» по полям", "", "| Поле | Условная сеть | Реальная сеть |", "|---|---|---|"]
    fc, fr = field_shares(cond), field_shares(real)
    for k in fc:
        out.append(f"| {k} | {pct(fc[k])} | {pct(fr[k])} |")
    out += ["", "Допущения сравнения: места узлов (PLACES), радиусы 5/15 км, путь через линии без тега напряжения.",
            "Генерация реальной сети — эквиваленты 220/110 (допущение); станций МАЭК, ГПЭС в модели нет — они вне островов 110 кВ."]
    text = "\n".join(out) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
