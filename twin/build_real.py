"""Сборка data/network_real.yaml из снимка OSM, таблиц опознания ПС и документов (шаги 2.2–2.4).

Решения пользователя: линии 110 кВ — только с тегом voltage=110000 (вариант A); с шага 2.4 в расчёт
входят линии 220 кВ, АТ 220/110, типовые трансформаторы 110/10, МАЭК, эквивалент Атырау и эквивалентные
нагрузки районов вне сети 110 кВ.

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

CONDUCTOR = "АС-185"      # ДОПУЩЕНИЕ: марка провода ВЛ 110 кВ в документах не найдена
CONDUCTOR_220 = "АС-300"  # ДОПУЩЕНИЕ: марка провода ВЛ 220 кВ в документах не найдена
MAEK_V = 1.02             # ДОПУЩЕНИЕ: напряжение шин 220 кВ МАЭК, о.е.
COS_PHI = 0.92            # ДОПУЩЕНИЕ: cos φ нагрузки для выбора мощности трансформаторов 110/10 (≈ tan φ 0,42 в model)
DLR_FACTS = SRC / "dlr_facts.yaml"


def real_conductors(base_conductors: dict) -> dict:
    """Провода network_real: r, диаметр — ГОСТ 839-80, длительный ток — ПУЭ табл. 1.3.29 (источники, шаг 3.3);
    x и b — типовые (допущение). Пометки по полям — в marks."""
    d = yaml.safe_load(DLR_FACTS.read_text(encoding="utf-8"))
    src = d["sources"]
    gost = {f["object"]["name"].split("/")[0]: f for f in d["facts"] if f["object"]["type"] == "провод"}
    pue = next(f for f in d["facts"] if f["object"]["name"] == "ПУЭ, табл. 1.3.29")["value"]
    typical = {"АС-185": (0.413, 2.75e-6), "АС-300": (0.429, 2.64e-6)}
    out = dict(base_conductors)
    for name, (x, b) in typical.items():
        g = gost[name]["value"]
        i = next(v for k, v in pue.items() if k.split("/")[0] == name)
        out[name] = {"r_ohm_km": g["r20_ohm_km"], "x_ohm_km": x, "b_s_km": b, "i_rated_a": i,
                     "diameter_m": g["diameter_mm"] / 1000, "status": "допущение",
                     "basis": "r и диаметр — ГОСТ 839-80, ток — ПУЭ табл. 1.3.29; x и b — типовые",
                     "marks": {"r_ohm_km": {"status": "источник", "source": src["gost_839_t4"],
                                            "basis": f"ГОСТ 839-80, табл. 4, {gost[name]['object']['name']}: при 20 °C"},
                               "diameter_m": {"status": "источник", "source": src["gost_839_t4"], "basis": "ГОСТ 839-80, табл. 4"},
                               "i_rated_a": {"status": "источник", "source": src["pue_13"],
                                             "basis": "ПУЭ табл. 1.3.29 (вне помещений; +70 °C провод, +25 °C воздух)"},
                               "x_ohm_km": {"status": "допущение", "basis": "типовое для одноцепной ВЛ; геометрия опор не найдена"},
                               "b_s_km": {"status": "допущение", "basis": "типовое; геометрия опор не найдена"}}}
    return out

HEADER = """\
# Модель сети 110–220 кВ Мангистауской области по ОТКРЫТЫМ ИСТОЧНИКАМ (шаги 2.2–2.4) — собрана скриптом twin/build_real.py.
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
# В расчёт входят линии с тегом voltage=110000 (вариант A) и 220000; линии без тега — в not_modelled.
# Трансформаторы: АТ 220/110 (мощность — источник, где найдена; параметры — типовые ГОСТ, допущение) и типовые
#   110/10 на ПС с нагрузкой (допущение). МАЭК — агрегаты ТЭЦ-1, ТЭЦ-2, ТЭС на шинах 220 кВ ТЭЦ-3 (шаг 3.1;
#   балансирует MAEK-TEC2-1); ГПЭС Жанаозен 6×20 МВт — сценарий 2026 (p_mw = 0); Атырау — эквивалент.
# Районы вне сети 110 кВ (Актау, Жанаозен) — эквивалентные нагрузки на шинах 220 кВ ТЭЦ-3 и 110 кВ ПС «Узень».
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


def tagged_110(s: dict) -> bool:
    """Участок с тегом voltage=110000 (вариант A)."""
    return s.get("kv") == "110.0" and s.get("kv_basis") == "тег voltage"


def graph_110(snap: dict, segs: list[dict], keep=tagged_110) -> tuple[list[dict], dict]:
    """Рёбра между узлами-терминалами (ПС, ответвления, обрывы) по участкам, для которых keep(seg) истинно
    (по умолчанию — линии с тегом 110 кВ). Связность — по всем общим узлам OSM (в т. ч. внутренним).
    У ребра kv_km — длина по классам напряжения участков ('' — без тега)."""
    seg = {int(s["way"]): s for s in segs}
    ways = [e for e in snap["elements"]
            if e.get("tags", {}).get("power") in ("line", "cable") and e["id"] in seg and keep(seg[e["id"]])]
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
                           "flags": seg[e["id"]]["flags"], "kv": seg[e["id"]]["kv"]})

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
        kv_km = Counter()
        for c in chain:
            kv_km[c["kv"]] += c["km"]
        edges.append({"p": start, "q": pt, "km": sum(c["km"] for c in chain), "kv_km": dict(kv_km),
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


def bus_loads(est: dict) -> dict:
    """Средняя нагрузка по шинам из load_estimates.yaml: {(osm, кВ): {"mw", "parts": [...], "sourced": bool}}."""
    out = {}
    for c in est["consumers"]:
        mw = c["energy_mln_kwh"] * 1000 / est["hours"] if "energy_mln_kwh" in c else c["avg_mw"]
        if abs(sum(a["frac"] for a in c["allocation"]) - 1) > 1e-3:
            raise ValueError(f"load_estimates: доли раскладки {c['id']} не дают 1")
        for a in c["allocation"]:
            b = out.setdefault((a["osm"], a.get("kv", 110)), {"mw": 0.0, "parts": [], "sourced": True})
            b["mw"] += mw * a["frac"]
            b["sourced"] &= c["status"] == "источник"
            b["parts"].append({"consumer": c["name"], "mw": round(mw * a["frac"], 3), "frac": a["frac"],
                               "energy_status": c["status"], "energy_source": c.get("source", ""),
                               "energy_basis": c.get("location", c.get("basis", "")), "allocation_basis": a["basis"]})
    return out


def _name_mark(r: dict) -> dict:
    if r.get("status") == "источник":
        return {"status": "источник", "source": r["source"], "confidence": r["confidence"], "basis": r["basis"]}
    return {"status": "допущение", "confidence": r.get("confidence", "не опознана"), "basis": r.get("basis", "нет строки опознания")}


def _circuit_lines(prefix: str, a: int, b: int, e: dict, buses: list, conductor: str, kv: str) -> list:
    """Линии по ребру графа: каждая цепь — отдельная линия; пометки по полям."""
    ns = {c[0] for c in e["circ"]}
    n = min(ns)
    cst = "источник" if all(c[1] == "источник" for c in e["circ"]) and len(ns) == 1 else "допущение"
    cbasis = "; ".join(sorted({c[2] for c in e["circ"]}))
    if len(ns) > 1:
        cbasis += f"; вдоль линии разное число цепей {sorted(ns)} — принято меньшее ({n})"
    ways = ";".join(str(w) for w in e["ways"])
    nm = f"{buses[a - 1]['short']} — {buses[b - 1]['short']}"
    out = []
    for c in range(1, n + 1):
        out.append({
            "id": f"{prefix}-{c}", "from": a, "to": b, "km": round(e["km"], 3), "conductor": conductor,
            "name": nm + (f" (цепь {c})" if n > 1 else ""), "status": "допущение",
            "basis": "длина и трасса — OSM (источник); марка провода — допущение",
            "osm_ways": ways,
            "marks": {
                "km": {"status": "источник", "source": OSM_WAY + str(e["ways"][0]),
                       "basis": f"сумма длин участков OSM по геометрии (гаверсинус): ways {ways}"},
                "conductor": {"status": "допущение", "basis": f"марка провода ВЛ {kv} кВ в документах не найдена — "
                              f"принята {conductor}"},
                "circuits": {"status": cst, "value": n, "basis": cbasis,
                             **({"source": OSM_WAY + str(e["ways"][0])} if cst == "источник" else {})},
                **({"flags": {"status": "допущение", "basis": f"теги OSM: {e['flags']}"}} if e["flags"] else {}),
            }})
    return out


UNIT_FIELDS = ("p_max_mw", "p_min_mw", "ramp_mw_h", "start_h", "min_up_h", "min_down_h", "heat_rate_gj_mwh", "startup_gj")


def load_units(path: Path = SRC / "unit_params.yaml") -> tuple[list, dict]:
    """Агрегаты из unit_params.yaml с раскрытием copy_of. Возвращает (список, раздел base_dispatch)."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    by_id, out = {}, []
    for u in raw["units"]:
        full = {**by_id[u["copy_of"]], **{k: v for k, v in u.items() if k != "copy_of"}} if "copy_of" in u else dict(u)
        by_id[u["id"]] = full
        out.append(full)
    return out, raw["base_dispatch"]


def units_to_gens(units: list, base_dispatch: dict, key_of: dict, load_avg: float) -> list:
    """Генераторы network_real из агрегатов: значения параметров + пометки по полям."""
    def val(x):
        return x["value"] if isinstance(x, dict) else x
    base_units = [u for u in units if not u.get("scenario") and not u.get("slack")]
    total = sum(val(u["p_max_mw"]) for u in base_units) + sum(val(u["p_max_mw"]) for u in units if u.get("slack"))
    gens = []
    for u in units:
        pmax = val(u["p_max_mw"])
        gpes = u["station"].startswith("ГПЭС")
        q = round((0.5 if gpes else 0.6) * pmax, 1)
        marks = {k: {"status": u[k]["status"], "basis": u[k]["basis"], **({"source": u[k]["source"]} if u[k].get("source") else {})}
                 for k in UNIT_FIELDS}
        mr = u.get("must_run", False)
        marks["must_run"] = ({"status": mr["status"], "basis": mr["basis"]} if isinstance(mr, dict)
                             else {"status": "допущение", "basis": "агрегат не обязан работать"})
        marks["location"] = {"status": "допущение", "basis": u.get("bus_basis", "все станции МАЭК — на шинах 220 кВ ТЭЦ-3; "
                             "ГПЭС — на шинах 110 кВ ПС «Узень» (подключение — generator_facts.yaml, выведено)")}
        marks["p_mw"] = {"status": "допущение", "basis": base_dispatch["basis"]}
        marks["q_limits"] = {"status": "допущение", "basis": f"±{0.5 if gpes else 0.6}·Pmax"}
        gens.append({
            "id": u["id"], "bus": key_of[(u["bus"]["osm"], u["bus"]["kv"])], "name": u["name"], "station": u["station"],
            **({"scenario": u["scenario"]} if u.get("scenario") else {}),
            "slack": bool(u.get("slack")),
            "p_mw": None if u.get("slack") else (0.0 if u.get("scenario") else round(load_avg * pmax / total, 1)),
            **{k: val(u[k]) for k in UNIT_FIELDS}, "must_run": bool(val(mr)),
            "q_min_mvar": -q, "q_max_mvar": q, "v_pu": 1.01 if gpes else MAEK_V,
            "status": "источник" if all(m["status"] == "источник" for m in marks.values()) else "допущение",
            "basis": "агрегат для суточного графика (шаг 3.1); параметры — data/network_sources/unit_params.yaml",
            "marks": marks})
    return gens


def assign_loads(buses: list, trafos: list, loads: dict, est: dict, tdn: dict) -> float:
    """Раскладка средней нагрузки по шинам. Нагрузка ПС из островов 110 кВ переносится на шину НН (10 кВ) за двумя
    типовыми трансформаторами 110/10 (допущение); эквивалентные нагрузки (kv ≠ 110 у ПС островов) остаются на своей шине.
    Возвращает сумму средней нагрузки, МВт."""
    total = sum(v["mw"] for v in loads.values())
    key = {(b["osm"], b["vn_kv"]): b for b in buses}
    missing = sorted(k for k in loads if k not in key)
    if missing:
        raise ValueError(f"load_estimates: шины {missing} нет в модели")
    peak = est["scenarios"]["peak"]["factor"]
    sizes = sorted(int(k.split("-")[1].split("/")[0]) // 1000 for k in tdn["value"])
    for k, ld in loads.items():
        b = key[k]
        energy_mark = {
            "status": "источник" if ld["sourced"] else "допущение",
            "source": "; ".join(sorted({x["energy_source"] for x in ld["parts"] if x["energy_source"]})),
            "basis": "; ".join(f"{x['consumer']}: {x['mw']} МВт ({'годовое потребление ' + str(est['year']) + ', ' + x['energy_basis'] if x['energy_status'] == 'источник' else x['energy_basis']})" for x in ld["parts"])}
        share_mark = {"status": "допущение", "basis": "доля = средняя нагрузка шины / сумма по модели; раскладка — "
                      "допущение: " + "; ".join(x["allocation_basis"] for x in ld["parts"])}
        target = b
        if b["kind"] == "ПС":  # ПС острова 110 кВ: шина НН и два типовых трансформатора 110/10
            s_peak = ld["mw"] * peak / COS_PHI
            need = 0.7 * s_peak
            sn = next((x for x in sizes if x >= need), None)
            if sn is None:
                raise ValueError(f"нет типового трансформатора для {b['name']}: нужно ≥ {need:.1f} МВА")
            row = tdn["value"][f"ТДН-{sn * 1000}/110"]
            lv_id = len(buses) + 1
            target = {"id": lv_id, "name": f"{b['name']} — шины НН 10 кВ (условно)", "short": (b["short"] + " НН")[:18],
                      "vn_kv": 10, "share": 0.0, "osm": b["osm"], "kind": "шины НН (условно)", "status": "допущение",
                      "basis": "условная шина НН: класс 10 кВ принят для всех ПС (у части ПС 35/6 кВ) — допущение",
                      "marks": {}}
            buses.append(target)
            b["marks"]["share"] = {"status": "допущение", "basis": f"нагрузка ПС перенесена на шину НН {lv_id}"}
            for u in (1, 2):
                trafos.append({"id": f"T-{b['id']}-{u}", "from": b["id"], "to": lv_id, "sn_mva": sn,
                               "uk_pct": row["uk"], "pk_kw": row["pk_kw"], "p0_kw": row["p0_kw"], "i0_pct": row["i0_pct"],
                               "name": f"Т-{u} 110/10 кВ {b['name']}", "status": "допущение",
                               "basis": "типовой трансформатор (решение пользователя: число и мощность не найдены)",
                               "marks": {"sn_mva": {"status": "допущение",
                                                    "basis": f"2 трансформатора, каждый ≥ 0,7 × пиковой нагрузки ПС "
                                                             f"({ld['mw']:.1f} МВт × {peak} / cos φ {COS_PHI} = {s_peak:.1f} МВА) → "
                                                             f"ближайший типовой ТДН-{sn * 1000}/110"},
                                         "params": {"status": "допущение",
                                                    "basis": f"параметры ТДН-{sn * 1000}/110 по {tdn['location']} ({tdn['source']}); "
                                                             "применение к ПС — допущение"},
                                         "ratio": {"status": "допущение", "basis": "номинальный, РПН не моделируется"}}})
        target["share"] = round(ld["mw"] / total, 6)
        target["load_mw_avg"] = round(ld["mw"], 3)
        target["marks"]["share"] = share_mark
        target["marks"]["load_energy"] = energy_mark
    return total


def build(snapshot: Path | None = None) -> tuple[dict, dict]:
    snap_path = Path(snapshot or latest_snapshot())
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    segs, subs, ids = _csv("osm_segments.csv"), _csv("osm_substations.csv"), _csv("substation_ids.csv")
    docs = yaml.safe_load((SRC / "documents.yaml").read_text(encoding="utf-8"))
    est = yaml.safe_load((SRC / "load_estimates.yaml").read_text(encoding="utf-8"))
    loads = bus_loads(est)
    load_total = sum(v["mw"] for v in loads.values())
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

    # шины 110 кВ островов (нагрузка раскладывается в конце, assign_loads)
    bus_id, buses = {}, []
    deg = Counter()
    for e in edges:
        deg[e["p"]] += 1
        deg[e["q"]] += 1
    feed_ids = {x for v in feeds.values() for x in v}
    for p in pts:
        if island_of[p] not in modelled:
            continue
        bid = len(buses) + 1
        bus_id[p] = bid
        if p[0] == "s":
            r = ident.get(p[1], {})
            nm = r.get("match_name") or f"ПС {p[1]}"
            s = sub[p[1]]
            is_feed = p[1] in feed_ids
            buses.append({
                "id": bid, "name": nm + (" — шины 110 кВ" if is_feed else ""), "short": nm[:18], "vn_kv": 110,
                "share": 0.0, "osm": p[1], "lat": float(s["lat"]), "lon": float(s["lon"]),
                "kind": "ПС 220/110 — шины 110 кВ" if is_feed else "ПС", "status": "допущение",
                "basis": "положение — OSM (источник); нагрузка — допущение",
                "marks": {
                    "position": {"status": "источник", "source": f"https://www.openstreetmap.org/{p[1]}"},
                    "name": _name_mark(r),
                    "share": {"status": "допущение", "basis": "нагрузка не задана"},
                }})
        else:
            kind = "ответвление" if deg[p] >= 3 else "обрыв линии"
            buses.append({
                "id": bid, "name": f"{kind} (узел OSM {p[1]})", "short": kind[:18], "vn_kv": 110, "share": 0.0,
                "osm": f"node/{p[1]}", "kind": kind, "status": "допущение",
                "basis": ("точка ответвления линии в OSM, ПС нет" if kind == "ответвление" else
                          "конец линии 110 кВ в OSM не доходит до ПС (ПС не нанесена или вне порога привязки 0,5 км)"),
                "marks": {
                    "position": {"status": "источник", "source": f"https://www.openstreetmap.org/node/{p[1]}"},
                    "share": {"status": "допущение", "basis": "нагрузка в узле без ПС не задаётся (0)"},
                }})

    # линии 110 кВ (каждая цепь — отдельная линия)
    lines = []
    for k, e in enumerate(sorted((e for e in edges if island_of[e["p"]] in modelled), key=lambda e: -e["km"])):
        lines += _circuit_lines(f"R{k + 1:02d}", bus_id[e["p"]], bus_id[e["q"]], e, buses, CONDUCTOR, "110")

    # ---------------- уровень 220 кВ (шаг 2.4) ----------------
    tf = yaml.safe_load((SRC / "transformer_facts.yaml").read_text(encoding="utf-8"))
    tsrc = tf["sources"]
    key_of = {}  # (osm, kv) → id шины
    for b in buses:
        if b["kind"] == "ПС 220/110 — шины 110 кВ":
            key_of[(b["osm"], 110)] = b["id"]

    def add_bus(osm, kv, name, kind, basis, marks, lat=None, lon=None):
        bid = len(buses) + 1
        buses.append({"id": bid, "name": name, "short": name[:18], "vn_kv": kv, "share": 0.0, "osm": osm,
                      **({"lat": lat, "lon": lon} if lat is not None else {}), "kind": kind,
                      "status": "допущение", "basis": basis, "marks": marks})
        key_of[(osm, kv)] = bid
        return bid

    subs220 = ["way/224369643", "way/181270259", "way/279492273", "way/226831318"]
    for osm in subs220:
        r, s = ident.get(osm, {}), sub[osm]
        add_bus(osm, 220, (r.get("match_name") or osm) + " — шины 220 кВ", "ПС/станция — шины 220 кВ",
                "положение — OSM (источник)",
                {"position": {"status": "источник", "source": f"https://www.openstreetmap.org/{osm}"},
                 "name": _name_mark(r), "share": {"status": "допущение", "basis": "нагрузка не задана"}},
                float(s["lat"]), float(s["lon"]))
    s = sub["way/226831318"]
    add_bus("way/226831318", 110, "ПС 220/110/10 кВ «Узень» — шины 110 кВ", "ПС 220/110 — шины 110 кВ",
            "шины 110 кВ ПС «Узень»: сеть 110 кВ Жанаозена в модели не представлена (нет тега 110 кВ в OSM) — "
            "на этой шине эквивалентная нагрузка",
            {"position": {"status": "источник", "source": "https://www.openstreetmap.org/way/226831318"},
             "name": _name_mark(ident.get("way/226831318", {})),
             "share": {"status": "допущение", "basis": "нагрузка не задана"}}, float(s["lat"]), float(s["lon"]))
    atyrau = add_bus("external/atyrau", 220, "Атырау 220 кВ (эквивалент за краем снимка OSM)", "внешняя сеть",
                     "концы двух ВЛ 220 кВ от ПС «Бейнеу», уходящих за пределы снимка; по документам — Л-2135/2145 "
                     "до ПС 220 кВ «Атырау» (mrek_tru_2021, стр. PDF 84)",
                     {"position": {"status": "допущение", "basis": "оба конца объединены в один узел; положение условное"},
                      "share": {"status": "допущение", "basis": "нагрузка не задаётся"}})

    # линии 220 кВ из OSM
    seg = {s_["way"]: s_ for s_ in segs}
    k220 = 0
    for r in _csv("osm_lines.csv"):
        if r["kv"] != "220.0":
            continue
        ends = r["ends"].split(";")
        if len(set(ends)) < 2 or any(e_ not in subs220 and e_ != "обрыв" for e_ in ends):
            continue  # петли внутри ТЭЦ-3
        a = key_of[(ends[0], 220)] if ends[0] != "обрыв" else atyrau
        b = key_of[(ends[1], 220)] if ends[1] != "обрыв" else atyrau
        ways = [int(w) for w in r["ways"].split(";")]
        circ = [circuits_of({"circuits": seg[str(w)]["circuits"], "cables": seg[str(w)]["cables"]}) for w in ways]
        k220 += 1
        new_l = _circuit_lines(f"H{k220:02d}", a, b, {"km": float(r["km"]), "ways": ways, "circ": circ, "flags": ""},
                               buses, CONDUCTOR_220, "220")
        if "обрыв" in ends:
            for l in new_l:
                l["marks"]["km"] = {"status": "допущение", "source": OSM_WAY + str(ways[0]),
                                    "basis": "длина участка до края снимка OSM; полная длина до ПС «Атырау» больше — "
                                             "эквивалент Атырау стоит на этом конце"}
        lines += new_l

    # автотрансформаторы 220/110
    trafos = []
    at_types = {f["object"]["name"]: f for f in tf["facts"] if f["object"]["type"] == "тип"}

    def at_params(type_name):
        f = at_types[type_name]
        v = f["value"]
        return ({"uk_pct": v["uk_vn_sn"], "pk_kw": v["pk_kw"], "p0_kw": v["p0_kw"], "i0_pct": v["i0_pct"]},
                f"типовые параметры {type_name}: {f['location']} ({f['source']}); паспорт аппарата не найден")

    at_list = (
        ("way/181270259", "Бейнеу", 2, 63,
         {"status": "источник", "source": tsrc["energybase_beyneu"], "basis": "2×63 МВА — карточка ПС (агрегатор energybase, вторичный)"},
         "АТДЦТН-63000/220/110"),
        ("way/279492273", "Каражанбас", 1, 150,
         {"status": "источник", "source": tsrc["kus_2019"], "basis": "1×150 МВА — отчёт ККС 2019, стр. PDF 4, 20"},
         "АТДЦТН-125000/220/110"),
        ("way/226831318", "Узень", 2, 125,
         {"status": "допущение", "basis": "два АТ (АТ-1, АТ-2 — mrek_2021, стр. PDF 42), мощность не найдена — "
                                          "принят распространённый тип 125 МВА"},
         "АТДЦТН-125000/220/110"),
    )
    for osm, short, n, sn, sn_mark, ptype in at_list:
        prm, pbasis = at_params(ptype)
        if sn != int(ptype.split("-")[1].split("/")[0]) // 1000:
            pbasis += f"; применены к АТ {sn} МВА (такого типа в таблице ГОСТ нет)"
        nm = ident.get(osm, {}).get("match_name", osm)
        for u in range(1, n + 1):
            trafos.append({"id": f"AT-{short}-{u}", "from": key_of[(osm, 220)], "to": key_of[(osm, 110)],
                           "sn_mva": sn, **prm, "name": f"АТ-{u} 220/110 кВ {nm}", "status": "допущение",
                           "basis": "мощность — см. marks.sn_mva; параметры — типовые по ГОСТ 17544-85 (допущение)",
                           "marks": {"sn_mva": sn_mark,
                                     "params": {"status": "допущение", "basis": pbasis},
                                     "ratio": {"status": "допущение",
                                               "basis": "коэффициент 220/110 номинальный, РПН не моделируется"}}})

    # генерация
    units, base_dispatch = load_units()
    load_avg_est = sum(v["mw"] for v in loads.values())
    gens = units_to_gens(units, base_dispatch, key_of, load_avg_est) + [
        {"id": "ATYRAU", "bus": atyrau, "name": "Атырауский энергоузел — эквивалент по ВЛ 220 кВ",
         "slack": False, "p_mw": 0.0, "p_max_mw": 80, "q_min_mvar": -100, "q_max_mvar": 100,
         "v_pu": 1.0, "ramp_mw_h": 80, "p_min_mw": 0.0, "start_h": 0, "min_up_h": 0, "min_down_h": 0,
         "heat_rate_gj_mwh": 0.0, "startup_gj": 0.0, "must_run": False, "fixed_p": True, "status": "допущение",
         "basis": "решение пользователя: эквивалент с P = 0, держит напряжение; переток — отдельным сценарием",
         "marks": {"p_max_mw": {"status": "допущение",
                                "basis": "переток по слабой связи ~80 МВт (dknews.kz, 14.11.2025, дата измерения не указана)"},
                   "location": {"status": "допущение", "basis": "конец ВЛ 220 кВ у края снимка OSM"},
                   "v_pu": {"status": "допущение", "basis": "1,0 о.е."},
                   "q_limits": {"status": "допущение", "basis": "±100 Мвар"},
                   "slack": {"status": "допущение", "basis": "PV-узел, P = 0 (в суточном графике не участвует)"}}},
    ]

    # нагрузка по шинам и типовые трансформаторы 110/10 на ПС с нагрузкой
    tdn = next(f for f in tf["facts"] if f["object"]["name"].startswith("ТДН-"))
    load_total = assign_loads(buses, trafos, loads, est, tdn)

    # острова (по линиям и трансформаторам)
    par2 = {b["id"]: b["id"] for b in buses}

    def f2(x):
        while par2[x] != x:
            par2[x] = par2[par2[x]]
            x = par2[x]
        return x
    for br_ in lines + trafos:
        par2[f2(br_["from"])] = f2(br_["to"])
    comp = defaultdict(list)
    for b in buses:
        comp[f2(b["id"])].append(b["id"])
    islands_out = []
    for ids_ in comp.values():
        islands_out.append({"buses": len(ids_), "generators": [g["id"] for g in gens if g["bus"] in ids_],
                            "km": round(sum(l["km"] for l in lines if l["from"] in ids_ and l["id"].endswith("-1")), 1)})

    # инвентаризация 220 кВ (в расчёте не участвует)
    lines220 = [{"route": r["route"], "km": float(r["km"]), "ways": r["ways"], "ends": r["ends"],
                 "status": "источник", "source": OSM_WAY + r["ways"].split(";")[0], "basis": "трасса и длина — OSM"}
                for r in _csv("osm_lines.csv") if r["kv"] == "220.0"]
    subs220_inv = [{"osm": s["osm"], "name": ident[s["osm"]]["match_name"], "voltage_tag": s["voltage_tag"],
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
        "lengths_220_km": share_of([l for l in lines if l["id"].startswith("H")], "km", lambda l: l["km"]),
        "transformer_ratings_by_mva": share_of(trafos, "sn_mva", lambda t: t["sn_mva"]),
        "transformer_params_by_count": share_of(trafos, "params"),
        "loads_by_bus": share_of([b for b in buses if b["share"] > 0], "share"),
        "load_energy_by_mw": share_of([b for b in buses if b["share"] > 0], "load_energy", lambda b: b["load_mw_avg"]),
        "generation_by_mw": share_of(gens, "p_max_mw", lambda g: g["p_max_mw"]),
        "generation_location_by_count": share_of(gens, "location"),
        "substation_names_by_count": {
            "sources": sum(1 for b in buses if b.get("marks", {}).get("name", {}).get("status") == "источник"),
            "total": sum(1 for b in buses if "name" in b.get("marks", {}))},
        "note": ("доли — по полям marks. loads_by_bus — раскладка по ПС (всегда допущение); load_energy_by_mw — доля "
                 "средней нагрузки, опирающейся на годовое потребление из документов; generation_by_mw — Pmax МАЭК "
                 "(Википедия, вторичный) и эквивалента Атырау (допущение)"),
    }

    osm_base = snap.get("osm3s", {}).get("timestamp_osm_base", "?")
    out = {
        "meta": {"base_mva": base["meta"]["base_mva"], "base_kv": 110, "frequency_hz": 50,
                 "built": date.today().isoformat(), "builder": "twin/build_real.py",
                 "osm_snapshot": snap_path.name, "osm_base_timestamp": osm_base,
                 "license": "ODbL 1.0 (© участники OpenStreetMap)",
                 "scope": "линии с тегом voltage=110000 (вариант A) и 220000; трансформаторы 220/110 и 110/10 (шаг 2.4)",
                 "load_scenarios": {
                     "year": est["year"], "avg_total_mw": round(load_total, 3),
                     **{k: round(load_total * v["factor"], 3) for k, v in est["scenarios"].items()},
                     "by_island_avg_mw": {k: round(sum(c["energy_mln_kwh"] * 1000 / est["hours"] if "energy_mln_kwh" in c
                                                       else c["avg_mw"] for c in est["consumers"] if c["island"] == k), 3)
                                          for k in sorted({c["island"] for c in est["consumers"]})},
                     "status": "допущение",
                     "basis": "суммарная нагрузка для scn.load: средняя = энергия/8760 (потребление МРЭК — источник; "
                              "прочая нагрузка Бейнеу и остаток МРЭК — допущение) × коэффициенты режимов (допущение), "
                              "см. data/network_sources/load_estimates.yaml"}},
        "conductors": real_conductors(base["conductors"]),
        "buses": buses, "lines": lines, "transformers": trafos, "generators": gens,
        "model": base["model"],
        "summary": summary,
        "islands": islands_out,
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
        "inventory_220kv": {"note": "с шага 2.4 линии 220 кВ из OSM входят в расчёт (lines H*); здесь — справка",
                            "substations": subs220_inv,
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
    print("шин", len(d["buses"]), "линий (цепей)", len(d["lines"]), "трансформаторов", len(d["transformers"]),
          "генераторов", len(d["generators"]))
    print(json.dumps(d["islands"], ensure_ascii=False))
    print(json.dumps(d["summary"], ensure_ascii=False, indent=1))
    print(json.dumps(d["meta"]["load_scenarios"], ensure_ascii=False))
