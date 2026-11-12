"""Электросеть Мангистауской области по данным OpenStreetMap (шаг 2.2).

Снимок: запрос к Overpass API сохраняется как есть в data/osm/power_<дата>.json;
вся дальнейшая обработка идёт только по снимку (без сети), чтобы результат был воспроизводим.

Данные OSM — © участники OpenStreetMap, лицензия ODbL 1.0: https://www.openstreetmap.org/copyright
"""
from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OSM_DIR = ROOT / "data" / "osm"
OVERPASS = "https://overpass-api.de/api/interpreter"
USER_AGENT = "research-grid-model/0.1"  # Overpass отвергает запросы со стандартным User-Agent

# Граница Мангистауской области: relation 215686 (admin_level=4, ISO3166-2 KZ-47).
# Area id в Overpass = 3600000000 + id relation.
REGION_RELATION = 215686
QUERY = f"""[out:json][timeout:300];
area({3600000000 + REGION_RELATION})->.a;
(
  way["power"~"^(line|cable)$"](area.a);
  nwr["power"~"^(substation|plant)$"](area.a);
  node["place"~"^(city|town|village|hamlet)$"](area.a);
);
out geom;"""


def fetch_snapshot(retries: int = 4) -> Path:
    """Скачать снимок и сохранить в data/osm/power_<дата>.json. Возвращает путь к файлу."""
    body = urllib.parse.urlencode({"data": QUERY}).encode()
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(OVERPASS, data=body, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=400) as r:
                data = json.load(r)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
            time.sleep(30 * (k + 1))  # Overpass при перегрузке отвечает 429/502 — ждём
    else:
        raise RuntimeError(f"Overpass недоступен: {last!r}")
    data["query"] = QUERY
    OSM_DIR.mkdir(parents=True, exist_ok=True)
    path = OSM_DIR / f"power_{date.today().isoformat()}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


def latest_snapshot() -> Path:
    files = sorted(OSM_DIR.glob("power_*.json"))
    if not files:
        raise FileNotFoundError("нет снимка data/osm/power_*.json — запустите fetch_snapshot()")
    return files[-1]


def load_snapshot(path: str | Path | None = None) -> dict:
    return json.loads(Path(path or latest_snapshot()).read_text(encoding="utf-8"))


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    la1, lo1, la2, lo2 = map(math.radians, (lat1, lon1, lat2, lon2))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


# ---------- обработка снимка ----------

SNAP_KM = 0.5  # ДОПУЩЕНИЕ: конец линии ближе 0,5 км к контуру ПС считается подключённым к этой ПС
VOLTAGE_CLASSES = {"110000": 110, "220000": 220, "35000": 35, "500000": 500}
ENDS = ("1", "2")


def _xy(lat: float, lon: float, lat0: float) -> tuple[float, float]:
    """Локальная плоская проекция, км (точности хватает для расстояний до нескольких км)."""
    return lon * 111.32 * math.cos(math.radians(lat0)), lat * 110.57


def _dist_to_polygon_km(lat: float, lon: float, poly: list) -> float:
    if len(poly) == 1:
        return haversine_km(lat, lon, *poly[0])
    px, py = _xy(lat, lon, lat)
    pts = [_xy(a, b, lat) for a, b in poly]
    edges = list(zip(pts, pts[1:] + pts[:1]))
    inside = False
    for (x1, y1), (x2, y2) in edges:
        if (y1 > py) != (y2 > py) and px < x1 + (py - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    if inside:
        return 0.0
    best = float("inf")
    for (x1, y1), (x2, y2) in edges:
        dx, dy = x2 - x1, y2 - y1
        t = 0.0 if dx == dy == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        best = min(best, math.hypot(px - x1 - t * dx, py - y1 - t * dy))
    return best


def substations(data: dict) -> list[dict]:
    """ПС и станции из снимка: центр контура, теги; контур — в поле _poly (в CSV не пишется)."""
    out = []
    for e in data["elements"]:
        t = e.get("tags", {})
        if t.get("power") not in ("substation", "plant"):
            continue
        if e["type"] == "node":
            poly = [(e["lat"], e["lon"])]
        elif e["type"] == "way":
            poly = [(g["lat"], g["lon"]) for g in e["geometry"]]
        else:  # relation: внешние контуры
            poly = [(g["lat"], g["lon"]) for m in e.get("members", []) if m.get("role") == "outer"
                    for g in m.get("geometry", [])]
        if not poly:
            continue
        out.append({"osm": f"{e['type']}/{e['id']}", "power": t["power"],
                    "name": t.get("name:ru") or t.get("name", ""), "voltage_tag": t.get("voltage", ""),
                    "lat": round(sum(p[0] for p in poly) / len(poly), 5),
                    "lon": round(sum(p[1] for p in poly) / len(poly), 5),
                    "plant_source": t.get("plant:source", ""), "plant_output": t.get("plant:output:electricity", ""),
                    "_poly": poly})
    return out


def places(data: dict) -> list[dict]:
    """Населённые пункты из снимка (для опознания безымянных ПС по близости)."""
    return [{"osm": f"node/{e['id']}", "name": e["tags"].get("name:ru") or e["tags"].get("name", ""),
             "place": e["tags"]["place"], "lat": e["lat"], "lon": e["lon"]}
            for e in data["elements"] if e["type"] == "node" and "place" in e.get("tags", {})]


def nearest_place(lat: float, lon: float, pl: list[dict]) -> tuple[dict | None, float]:
    best = min(((haversine_km(lat, lon, p["lat"], p["lon"]), i) for i, p in enumerate(pl)), default=(float("inf"), -1))
    return (pl[best[1]] if best[1] >= 0 else None), best[0]


SOURCES_DIR = ROOT / "data" / "network_sources"


def export_tables(snapshot: str | Path | None = None, out_dir: Path = SOURCES_DIR) -> dict:
    """Записать osm_segments.csv, osm_lines.csv, osm_substations.csv по снимку. Возвращает таблицы."""
    import pandas as pd

    path = Path(snapshot or latest_snapshot())
    data = load_snapshot(path)
    subs, pl = substations(data), places(data)
    segs = segments(data, subs)
    rts = routes(segs)
    attached = {}
    for r in rts:
        for sid in filter(None, r["substations"].split(";")):
            attached.setdefault(sid, set()).add(r["kv"])
    sub_rows = []
    for s in subs:
        p, dist = nearest_place(s["lat"], s["lon"], pl)
        kvs = attached.get(s["osm"], set())
        sub_rows.append({k: v for k, v in s.items() if not k.startswith("_")} | {
            "nearest_place": p["name"] if p else "", "nearest_place_osm": p["osm"] if p else "",
            "nearest_place_km": round(dist, 2),
            "lines_kv": ";".join(sorted("?" if k is None else str(k) for k in kvs))})
    tables = {"osm_segments": pd.DataFrame(segs), "osm_lines": pd.DataFrame(rts), "osm_substations": pd.DataFrame(sub_rows)}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.insert(0, "snapshot", path.name)
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


def _by_node(segs: list[dict]) -> dict:
    at = {}
    for s in segs:
        for k in ENDS:
            at.setdefault(s["n" + k], []).append(s)
    return at


def segments(data: dict, subs: list[dict] | None = None) -> list[dict]:
    """Участки линий (ways): длина, класс напряжения, привязка концов к ПС.

    Класс участка без тега voltage выводится, только если каждый его стыкующийся конец — простое продолжение
    (в узле ровно два участка, конец не на ПС) и все соседи с известным классом — одного класса
    (соседи без класса не учитываются). Иначе — «неизвестно».
    Повторяется до устойчивости, чтобы пройти цепочки участков без тега.
    """
    subs = subs if subs is not None else substations(data)
    segs = []
    for e in data["elements"]:
        t = e.get("tags", {})
        if t.get("power") not in ("line", "cable"):
            continue
        g, vt = e["geometry"], t.get("voltage", "")
        cls = [VOLTAGE_CLASSES.get(v) for v in vt.split(";")] if vt else []
        kv = cls[0] if len(cls) == 1 and cls[0] else None
        seg = {"way": e["id"], "voltage_tag": vt, "kv": kv,
               "kv_basis": "тег voltage" if kv else (f"тег voltage={vt} неоднозначен" if vt else ""),
               "circuits": t.get("circuits", ""), "cables": t.get("cables", ""),
               "flags": ";".join(k for k in ("fixme", "disused", "voltage:design", "abandoned") if k in t),
               "km": round(sum(haversine_km(a["lat"], a["lon"], b["lat"], b["lon"]) for a, b in zip(g, g[1:])), 3),
               "n1": e["nodes"][0], "n2": e["nodes"][-1],
               "lat1": g[0]["lat"], "lon1": g[0]["lon"], "lat2": g[-1]["lat"], "lon2": g[-1]["lon"]}
        for k in ENDS:
            d, sid = min(((_dist_to_polygon_km(seg["lat" + k], seg["lon" + k], s["_poly"]), s["osm"]) for s in subs),
                         default=(float("inf"), ""))
            seg["sub" + k] = sid if d <= SNAP_KM else ""
        segs.append(seg)

    at = _by_node(segs)
    changed = True
    while changed:
        changed = False
        for s in segs:
            if s["kv"] is not None or s["voltage_tag"]:
                continue
            neigh, ok = set(), True
            for k in ENDS:
                others = [o for o in at[s["n" + k]] if o is not s]
                if not others:
                    continue  # висячий конец: не мешает и не помогает
                if s["sub" + k] or len(others) != 1:
                    ok = False  # ответвление или ПС — продолжение неоднозначно
                    break
                if others[0]["kv"] is not None:  # сосед без класса не помогает и не мешает
                    neigh.add(others[0]["kv"])
            if ok and len(neigh) == 1:
                s["kv"] = neigh.pop()
                s["kv_basis"] = "выведено: продолжение участка того же класса"
                changed = True
    for s in segs:
        if s["kv"] is None and not s["kv_basis"]:
            s["kv_basis"] = "неизвестно"
    return segs


def routes(segs: list[dict]) -> list[dict]:
    """Склейка участков одного класса в линии через узлы-продолжения (ровно два участка, не на ПС)."""
    at = _by_node(segs)
    parent = {s["way"]: s["way"] for s in segs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for node, lst in at.items():
        if len(lst) == 2 and lst[0]["kv"] == lst[1]["kv"]:
            on_sub = any(s["sub" + k] for s in lst for k in ENDS if s["n" + k] == node)
            if not on_sub:
                parent[find(lst[0]["way"])] = find(lst[1]["way"])
    groups = {}
    for s in segs:
        groups.setdefault(find(s["way"]), []).append(s)
    out = []
    for i, lst in enumerate(sorted(groups.values(), key=lambda g: -sum(s["km"] for s in g))):
        ids = {s["way"] for s in lst}
        ends = [s["sub" + k] or "обрыв" for s in lst for k in ENDS
                if not [o for o in at[s["n" + k]] if o["way"] in ids and o is not s]]
        out.append({"route": f"R{i + 1:03d}", "kv": lst[0]["kv"], "km": round(sum(s["km"] for s in lst), 2),
                    "n_ways": len(lst), "ways": ";".join(str(s["way"]) for s in lst), "ends": ";".join(ends),
                    "substations": ";".join(sorted({e for e in ends if e != "обрыв"})),
                    "n_open_ends": ends.count("обрыв"),
                    "kv_basis": ";".join(sorted({s["kv_basis"] for s in lst}))})
    return out
