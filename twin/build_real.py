"""Сборка data/network_real.yaml из снимка OSM, таблиц опознания ПС и документов (шаг 2.2, пункт 5).

Решения пользователя: в расчёт идут только линии с тегом voltage=110000 (вариант A);
220 кВ — только в инвентаризацию; питание от ПС 220/110 — эквивалентные источники (допущение).

Пометки: у элемента общий status (как требует load_network): «источник» — только если все его
поля из источников, иначе «допущение». Подробно по полям — в marks: {поле: {status, source|basis}}.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "network_sources"
OUT = ROOT / "data" / "network_real.yaml"
OSM_WAY = "https://www.openstreetmap.org/way/"

CONDUCTOR = "АС-185"  # ДОПУЩЕНИЕ: марка провода ВЛ 110 кВ в документах не найдена
EQ_COS_PHI = 0.92     # ДОПУЩЕНИЕ: пересчёт мощности АТ (МВА) в предел эквивалента (МВт)
EQ_UNLIMITED_MW = 500  # ДОПУЩЕНИЕ: предел эквивалента, когда мощность АТ неизвестна
EQ_V_PU = 1.0          # ДОПУЩЕНИЕ: напряжение на шине 110 кВ ПС 220/110
EQ_Q_FACTOR = 0.6      # ДОПУЩЕНИЕ: пределы Q эквивалента = ±0,6·Pmax

# Мощность АТ из документов (documents.yaml): osm-id ПС → (МВА, ключ источника, где)
AT_MVA = {"way/279492273": (150, "kus_2019", "стр. PDF 4, 20")}

HEADER = """\
# Модель сети 110 кВ Мангистауской области по ОТКРЫТЫМ ИСТОЧНИКАМ (шаг 2.2) — собрана скриптом twin/build_real.py.
# Не редактировать вручную: пересобрать командой  .venv\\Scripts\\python -m twin.build_real
#
# Лицензия данных OpenStreetMap: трассы, длины, цепи и положение ПС взяты из OpenStreetMap.
#   © участники OpenStreetMap. Данные доступны по лицензии Open Database License (ODbL) 1.0:
#   https://opendatacommons.org/licenses/odbl/1-0/ ; авторство: https://www.openstreetmap.org/copyright
#   Снимок: {snapshot} (метка времени базы OSM: {osm_base}). Этот файл — производная база данных
#   и распространяется на тех же условиях ODbL 1.0.
#
# Пометки: status у элемента — «источник» только если ВСЕ его поля из источников, иначе «допущение».
#   Подробно по полям — в marks: {{поле: {{status, source | basis}}}}. Доли источников — в разделе summary.
# В расчёт входят только линии с тегом voltage=110000 (решение пользователя, вариант A).
#   Линии без тега напряжения — в разделе not_modelled, 220 кВ — в inventory_220kv (в расчёте не участвуют).
# Питание от ПС 220/110 кВ — эквивалентные источники (допущение): сеть 220 кВ и АТ не моделируются (шаг 2.4).
"""


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _csv(name):
    with open(SRC / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def latest_snapshot() -> Path:
    files = sorted((ROOT / "data" / "osm").glob("power_*.json"))
    if not files:
        raise FileNotFoundError("нет снимка data/osm/power_*.json")
    return files[-1]


def circuits_of(tags: dict) -> tuple[int, str, str]:
    """Число цепей участка: (n, статус, основание)."""
    c, k = tags.get("circuits", ""), tags.get("cables", "")
    if c.isdigit() and int(c) > 0:
        return int(c), "источник", f"тег circuits={c}"
    if k.isdigit() and int(k) in (3, 6):
        return int(k) // 3, "допущение", f"тега circuits нет; cables={k} → {int(k) // 3} цепь(и) по числу проводов"
    return 1, "допущение", "нет тегов circuits и cables — принята 1 цепь"


def graph_110(snap: dict, segs: list[dict]) -> tuple[list[dict], dict]:
    """Рёбра между узлами-терминалами (ПС, ответвления, обрывы) по линиям с тегом 110 кВ.
    Связность — по всем общим узлам OSM (в т. ч. внутренним), не только по концам участков."""
    seg = {int(s["way"]): s for s in segs}
    ways = [e for e in snap["elements"]
            if e.get("tags", {}).get("power") in ("line", "cable") and seg.get(e["id"], {}).get("kv") == "110.0"
            and seg[e["id"]]["kv_basis"] == "тег voltage"]
    inc = Counter()
    for e in ways:
        n = e["nodes"]
        inc[n[0]] += 1
        inc[n[-1]] += 1
        for x in n[1:-1]:
            inc[x] += 2
    sub_of = {}  # узел конца участка → ПС (привязка из osm_segments.csv, порог SNAP_KM в twin/osm.py)
    for e in ways:
        s = seg[e["id"]]
        if s["sub1"]:
            sub_of[e["nodes"][0]] = s["sub1"]
        if s["sub2"]:
            sub_of[e["nodes"][-1]] = s["sub2"]

    def point(n):
        return ("s", sub_of[n]) if n in sub_of else ("n", n)

    pieces = []
    for e in ways:
        n, g = e["nodes"], e["geometry"]
        cut = [0] + [i for i in range(1, len(n) - 1) if inc[n[i]] >= 3 or n[i] in sub_of] + [len(n) - 1]
        circ = circuits_of(e["tags"])
        for a, b in zip(cut, cut[1:]):
            km = sum(haversine_km(g[i]["lat"], g[i]["lon"], g[i + 1]["lat"], g[i + 1]["lon"]) for i in range(a, b))
            pieces.append({"p": point(n[a]), "q": point(n[b]), "km": km, "ways": [e["id"]], "circ": [circ],
                           "flags": seg[e["id"]]["flags"]})

    # склейка цепочек: терминал — ПС или узел, где сходится не ровно два куска;
    # от каждого терминала идём по кускам через узлы-продолжения до следующего терминала
    at = defaultdict(list)
    for i, pc in enumerate(pieces):
        at[pc["p"]].append(i)
        at[pc["q"]].append(i)

    def terminal(pt):
        return pt[0] == "s" or len(at[pt]) != 2

    used, edges = set(), []

    def walk(start, first):
        chain, pt, i = [], start, first
        while True:
            used.add(i)
            chain.append(pieces[i])
            pt = pieces[i]["q"] if pieces[i]["p"] == pt else pieces[i]["p"]
            if terminal(pt):
                break
            nxt = [k for k in at[pt] if k != i and k not in used]
            if not nxt:
                break
            i = nxt[0]
        edges.append({"p": start, "q": pt, "km": sum(c["km"] for c in chain),
                      "ways": [w for c in chain for w in c["ways"]], "circ": [x for c in chain for x in c["circ"]],
                      "flags": ";".join(sorted({c["flags"] for c in chain if c["flags"]}))})

    for pt in list(at):
        if terminal(pt):
            for i in at[pt]:
                if i not in used:
                    walk(pt, i)
    for i in range(len(pieces)):  # замкнутые кольца без терминалов
        if i not in used:
            walk(pieces[i]["p"], i)
    return edges, {"ways": len(ways), "km": sum(float(seg[e["id"]]["km"]) for e in ways)}


def build(snapshot: Path | None = None) -> tuple[dict, dict]:
    snap_path = Path(snapshot or latest_snapshot())
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    segs, subs, ids = _csv("osm_segments.csv"), _csv("osm_substations.csv"), _csv("substation_ids.csv")
    docs = yaml.safe_load((SRC / "documents.yaml").read_text(encoding="utf-8"))
    dsrc = docs["sources"]
    sub = {s["osm"]: s for s in subs}
    ident = {r["osm"]: r for r in ids}
    base = yaml.safe_load((ROOT / "data" / "network.yaml").read_text(encoding="utf-8"))

    edges, stat110 = graph_110(snap, segs)
    edges = [e for e in edges if e["p"] != e["q"]]  # петли внутри одной ПС не нужны

    # узлы-терминалы и острова
    pts = sorted({e["p"] for e in edges} | {e["q"] for e in edges}, key=str)
    par = {p: p for p in pts}

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for e in edges:
        par[f(e["p"])] = f(e["q"])
    island_of = {p: f(p) for p in pts}
    feeds = {}  # остров → ПС 220/110
    for p in pts:
        if p[0] == "s" and {220, 110} <= {int(v) // 1000 for v in sub[p[1]]["voltage_tag"].split(";") if v.isdigit()}:
            feeds.setdefault(island_of[p], []).append(p[1])
    isl_km = Counter()
    for e in edges:
        isl_km[island_of[e["p"]]] += e["km"]
    modelled = {i for i in isl_km if len(feeds.get(i, [])) == 1}
    excluded = [{"island_km": round(isl_km[i], 1), "reason": "нет ПС 220/110 и генерации в данных OSM"
                 if not feeds.get(i) else f"несколько ПС 220/110: {feeds[i]}"} for i in isl_km if i not in modelled]

    # шины
    bus_id, buses = {}, []
    deg = Counter()
    for e in edges:
        deg[e["p"]] += 1
        deg[e["q"]] += 1
    load_pts = [p for p in pts if island_of[p] in modelled and p[0] == "s" and p[1] not in
                {x for v in feeds.values() for x in v}]
    share = round(1.0 / len(load_pts), 6) if load_pts else 0.0
    for p in pts:
        if island_of[p] not in modelled:
            continue
        bid = len(buses) + 1
        bus_id[p] = bid
        if p[0] == "s":
            r = ident.get(p[1], {})
            nm = r.get("match_name") or f"ПС {p[1]}"
            s = sub[p[1]]
            is_feed = p[1] in {x for v in feeds.values() for x in v}
            name_mark = ({"status": "источник", "source": r["source"], "confidence": r["confidence"], "basis": r["basis"]}
                         if r.get("status") == "источник" else
                         {"status": "допущение", "confidence": r.get("confidence", "не опознана"),
                          "basis": r.get("basis", "нет строки опознания")})
            buses.append({
                "id": bid, "name": nm, "short": nm[:18], "share": 0.0 if is_feed else share,
                "osm": p[1], "lat": float(s["lat"]), "lon": float(s["lon"]), "kind": "ПС 220/110 (питание)" if is_feed else "ПС",
                "status": "допущение",
                "basis": "положение — OSM (источник); доля нагрузки — допущение",
                "marks": {
                    "position": {"status": "источник", "source": f"https://www.openstreetmap.org/{p[1]}"},
                    "name": name_mark,
                    "share": {"status": "допущение", "basis": "питающий узел: нагрузка 110 кВ не задаётся" if is_feed
                              else f"поровну между {len(load_pts)} ПС модели; реальная нагрузка — шаг 2.3"},
                }})
        else:
            kind = "ответвление" if deg[p] >= 3 else "обрыв линии"
            buses.append({
                "id": bid, "name": f"{kind} (узел OSM {p[1]})", "short": kind[:18], "share": 0.0,
                "osm": f"node/{p[1]}", "kind": kind, "status": "допущение",
                "basis": ("точка ответвления линии в OSM, ПС нет" if kind == "ответвление" else
                          "конец линии 110 кВ в OSM не доходит до ПС (ПС не нанесена или вне порога привязки 0,5 км)"),
                "marks": {
                    "position": {"status": "источник", "source": f"https://www.openstreetmap.org/node/{p[1]}"},
                    "share": {"status": "допущение", "basis": "нагрузка в узле без ПС не задаётся (0)"},
                }})

    # линии (каждая цепь — отдельная линия)
    lines = []
    for k, e in enumerate(sorted((e for e in edges if island_of[e["p"]] in modelled), key=lambda e: -e["km"])):
        a, b = bus_id[e["p"]], bus_id[e["q"]]
        ns = {c[0] for c in e["circ"]}
        n = min(ns)
        cst = "источник" if all(c[1] == "источник" for c in e["circ"]) and len(ns) == 1 else "допущение"
        cbasis = "; ".join(sorted({c[2] for c in e["circ"]}))
        if len(ns) > 1:
            cbasis += f"; вдоль линии разное число цепей {sorted(ns)} — принято меньшее ({n})"
        ways = ";".join(str(w) for w in e["ways"])
        nm = f"{buses[a - 1]['short']} — {buses[b - 1]['short']}"
        for c in range(1, n + 1):
            lines.append({
                "id": f"R{k + 1:02d}-{c}", "from": a, "to": b, "km": round(e["km"], 3), "conductor": CONDUCTOR,
                "name": nm + (f" (цепь {c})" if n > 1 else ""), "status": "допущение",
                "basis": "длина и трасса — OSM (источник); марка провода — допущение",
                "osm_ways": ways,
                "marks": {
                    "km": {"status": "источник", "source": OSM_WAY + str(e["ways"][0]),
                           "basis": f"сумма длин участков OSM по геометрии (гаверсинус): ways {ways}"},
                    "conductor": {"status": "допущение", "basis": f"марка провода ВЛ 110 кВ в документах не найдена — "
                                  f"принята {CONDUCTOR}"},
                    "circuits": {"status": cst, "value": n, "basis": cbasis,
                                 **({"source": OSM_WAY + str(e["ways"][0])} if cst == "источник" else {})},
                    **({"flags": {"status": "допущение", "basis": f"теги OSM: {e['flags']}"}} if e["flags"] else {}),
                }})

    # эквиваленты 220/110
    gens = []
    feed_list = sorted(((i, v[0]) for i, v in feeds.items() if i in modelled), key=lambda t: -isl_km[t[0]])
    for j, (isl, sid) in enumerate(feed_list):
        r = ident.get(sid, {})
        at = AT_MVA.get(sid)
        pmax = round(at[0] * EQ_COS_PHI, 1) if at else EQ_UNLIMITED_MW
        pbasis = (f"АТ {at[0]} МВА (источник: {dsrc[at[1]]}, {at[2]}) × cos φ {EQ_COS_PHI} (допущение)" if at else
                  f"мощность АТ не найдена — принят предел {EQ_UNLIMITED_MW} МВт (без ограничения)")
        gens.append({
            "id": f"EQ{j + 1}", "bus": bus_id[("s", sid)], "name": f"Эквивалент 220 кВ: {r.get('match_name') or sid}",
            "slack": j == 0, "p_mw": None if j == 0 else 0.0, "p_max_mw": pmax,
            "q_min_mvar": -round(EQ_Q_FACTOR * pmax, 1), "q_max_mvar": round(EQ_Q_FACTOR * pmax, 1),
            "v_pu": EQ_V_PU, "ramp_mw_h": round(pmax, 1), "status": "допущение",
            "basis": "питание из сети 220 кВ через АТ 220/110; сеть 220 кВ и АТ не моделируются (шаг 2.4)",
            "marks": {
                "location": {"status": r.get("status", "допущение"), "source": r.get("source", ""), "basis": r.get("basis", "")},
                "p_max_mw": {"status": "допущение", "basis": pbasis},
                "v_pu": {"status": "допущение", "basis": f"напряжение шины 110 кВ ПС 220/110 принято {EQ_V_PU} о.е."},
                "q_limits": {"status": "допущение", "basis": f"±{EQ_Q_FACTOR}·Pmax"},
                "slack": {"status": "допущение", "basis": "балансирующий — эквивалент самого большого острова" if j == 0
                          else "в своём острове ядро делает его балансирующим автоматически"},
            }})

    # инвентаризация 220 кВ (в расчёте не участвует)
    lines220 = [{"route": r["route"], "km": float(r["km"]), "ways": r["ways"], "ends": r["ends"],
                 "status": "источник", "source": OSM_WAY + r["ways"].split(";")[0], "basis": "трасса и длина — OSM"}
                for r in _csv("osm_lines.csv") if r["kv"] == "220.0"]
    subs220 = [{"osm": s["osm"], "name": ident[s["osm"]]["match_name"], "voltage_tag": s["voltage_tag"],
                "confidence": ident[s["osm"]]["confidence"], "status": ident[s["osm"]]["status"],
                "basis": ident[s["osm"]]["basis"], "source": ident[s["osm"]]["source"]}
               for s in subs if "220000" in s["voltage_tag"] or s["osm"] == "way/224369643"]
    doc220 = [f for f in docs["facts"] if "220" in str(f["object"].get("name", "")) + str(f["object"].get("voltage", ""))
              or "220" in str(f["field"])]

    unk = [s for s in segs if s["kv"] == ""]
    v35 = [s for s in segs if s["kv"] == "35.0"]

    # доли источников
    km_tot = sum(l["km"] for l in lines)
    def share_of(items, key, weight=lambda x: 1.0):
        tot = sum(weight(x) for x in items)
        src = sum(weight(x) for x in items if x["marks"][key]["status"] == "источник")
        return {"sources": round(src, 3), "total": round(tot, 3), "share": round(src / tot, 3) if tot else None}
    summary = {
        "lengths_km": share_of(lines, "km", lambda l: l["km"]),
        "conductors_by_km": share_of(lines, "conductor", lambda l: l["km"]),
        "conductors_by_count": share_of(lines, "conductor"),
        "circuits_by_km": share_of(lines, "circuits", lambda l: l["km"]),
        "loads_by_bus": share_of([b for b in buses if b["share"] > 0], "share"),
        "generation_by_mw": share_of(gens, "p_max_mw", lambda g: g["p_max_mw"]),
        "generation_location_by_count": share_of(gens, "location"),
        "substation_names_by_count": {
            "sources": sum(1 for b in buses if b.get("marks", {}).get("name", {}).get("status") == "источник"),
            "total": sum(1 for b in buses if "name" in b.get("marks", {}))},
        "note": "доли — по полям marks; «источник» в нагрузках и генерации отсутствует: реальная нагрузка — шаг 2.3",
    }

    osm_base = snap.get("osm3s", {}).get("timestamp_osm_base", "?")
    out = {
        "meta": {"base_mva": base["meta"]["base_mva"], "base_kv": 110, "frequency_hz": 50,
                 "built": date.today().isoformat(), "builder": "twin/build_real.py",
                 "osm_snapshot": snap_path.name, "osm_base_timestamp": osm_base,
                 "license": "ODbL 1.0 (© участники OpenStreetMap)",
                 "scope": "только линии с тегом voltage=110000 (вариант A)"},
        "conductors": base["conductors"],
        "buses": buses, "lines": lines, "generators": gens,
        "model": base["model"],
        "summary": summary,
        "islands": [{"feed": sid, "feed_name": ident[sid]["match_name"], "km": round(isl_km[i], 1),
                     "buses": sum(1 for p in pts if island_of[p] == i), "slack": gens[j]["id"]}
                    for j, (i, sid) in enumerate(feed_list)],
        "not_modelled": {
            "unknown_voltage": {"segments": len(unk), "km": round(sum(float(s["km"]) for s in unk), 1),
                                "basis": "в OSM нет тега voltage; по решению пользователя (вариант A) в расчёт не входят. "
                                         "Здесь сети Актау, Жетыбая, Жанаозена — в OSM у них нет линий с тегом 110 кВ"},
            "kv35": {"segments": len(v35), "km": round(sum(float(s["km"]) for s in v35), 1)},
            "islands_without_feed": excluded,
            "osm_110kv_total_km": round(stat110["km"], 1),
            "mrek_110kv_total_km": {"value": 1526.01, "status": "источник",
                                    "source": dsrc["mrek_2024"], "basis": "стр. PDF 4, «Компания в цифрах», 2024 г."},
        },
        "inventory_220kv": {"note": "в расчёте не участвует (шаг 2.4)", "substations": subs220,
                            "lines_osm": lines220, "facts_from_documents": [
                                {k: f[k] for k in ("object", "field", "value", "unit", "source", "location")} for f in doc220]},
    }
    return out, {"snapshot": snap_path.name, "osm_base": osm_base}


def write(path: Path = OUT) -> dict:
    data, hdr = build()
    text = HEADER.format(snapshot=hdr["snapshot"], osm_base=hdr["osm_base"]) + "\n" + \
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=140)
    path.write_text(text, encoding="utf-8")
    return data


if __name__ == "__main__":
    d = write()
    print("шин", len(d["buses"]), "линий (цепей)", len(d["lines"]), "эквивалентов", len(d["generators"]))
    print(json.dumps(d["islands"], ensure_ascii=False))
    print(json.dumps(d["summary"], ensure_ascii=False, indent=1))
    print(json.dumps(d["not_modelled"]["islands_without_feed"], ensure_ascii=False))
