"""Опознание подстанций OSM по названиям из документов (шаг 2.2, пункт 4).

Вход: data/network_sources/osm_substations.csv (ПС и станции из снимка OSM) и
data/network_sources/documents.yaml (факты из документов). Выход: substation_ids.csv.

Уверенность и основание указываются у каждой ПС:
  высокая     — имя стоит в самом OSM (тег name)                         → статус «источник» (OSM);
  средняя     — класс напряжения в OSM совпадает с ПС из документа, документ называет посёлок,
                и это единственный такой кандидат у посёлка              → статус «источник» (документ), вывод описан;
  низкая      — только близость к посёлку; название не установлено       → статус «допущение», расстояние в basis;
  не опознана — оснований нет.
"""
from __future__ import annotations

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = ROOT / "data" / "network_sources"

DOC_PLACE_KM = 15.0  # ДОПУЩЕНИЕ: ПС документа «у посёлка X» ищется не дальше 15 км от него
NEAR_KM = 5.0        # ДОПУЩЕНИЕ: для низкой уверенности ПС должна быть не дальше 5 км от посёлка

# ПС из документов, которые можно привязать к посёлку. kv — классы напряжения по документу;
# need — класс, который обязан стоять в теге voltage у ПС OSM; place — посёлок (как в OSM);
# facts — где в documents.yaml записан факт (ключ источника и страница); why — почему посёлок именно этот.
DOC_SUBSTATIONS = [
    {"name": "ПС 220 кВ «Бейнеу»", "kv": "220/110", "need": 220, "place": "Бейнеу",
     "facts": [("lada_kegoc_2026", "текст статьи"), ("mrek_tru_2021", "стр. PDF 84")],
     "why": "название ПС совпадает с посёлком; в документах — ПС 220 кВ с шинами 110 кВ"},
    {"name": "УРПС 220/110/35/10 кВ «Каражанбас»", "kv": "220/110/35/10", "need": 220, "place": "Каражанбас",
     "facts": [("kus_2019", "стр. PDF 4, 20")],
     "why": "название ПС совпадает с посёлком (месторождением)"},
    {"name": "ПС 220/110/10 кВ «Узень»", "kv": "220/110/10", "need": 220, "place": "Жанаозен",
     "facts": [("mrek_2021", "стр. PDF 42"), ("eco_gpes_uzen", "стр. PDF 2")],
     "why": "ГПЭС в г. Жанаозен подключается к ПС «Узень» (eco_gpes_uzen); Узень — месторождение у Жанаозена"},
    {"name": "ПС 110/35/6 кВ «Бейнеу»", "kv": "110/35/6", "need": 110, "need_also": 35, "place": "Бейнеу",
     "facts": [("mrek_tru_2021", "стр. PDF 44")],
     "why": "название ПС совпадает с посёлком; классы 110 и 35 кВ совпадают с тегом voltage"},
]

FIELDS = ["osm", "power", "voltage_tag", "lat", "lon", "nearest_place", "nearest_place_km",
          "match_name", "doc_voltage", "confidence", "status", "basis", "source"]


def voltage_classes(tag: str) -> set[int]:
    """Классы напряжения (кВ) из тега voltage OSM, например '220000;110000' → {220, 110}."""
    out = set()
    for v in tag.split(";"):
        v = v.strip()
        if v.isdigit():
            out.add(int(v) // 1000)
    return out


def load_doc_sources(path: Path = SOURCES_DIR / "documents.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]


def _doc_candidates(doc: dict, subs: list[dict]) -> list[dict]:
    out = []
    for s in subs:
        cls = voltage_classes(s["voltage_tag"])
        if s["power"] != "substation" or doc["need"] not in cls:
            continue
        if doc.get("need_also") and doc["need_also"] not in cls:
            continue
        if doc.get("need") == 110 and 220 in cls:
            continue  # ПС 110 кВ не путать с ПС 220/110 у того же посёлка
        if s["nearest_place"] == doc["place"] and float(s["nearest_place_km"]) <= DOC_PLACE_KM:
            out.append(s)
    return out


def identify(subs: list[dict], doc_sources: dict) -> list[dict]:
    """Таблица опознания: одна строка на ПС/станцию OSM."""
    rows = {s["osm"]: {k: s.get(k, "") for k in FIELDS} for s in subs}
    for r in rows.values():
        r.update(match_name="", doc_voltage="", confidence="не опознана", status="не опознана", basis="", source="")

    # 1) высокая: имя в OSM
    for s in subs:
        if s["name"]:
            rows[s["osm"]].update(match_name=s["name"], confidence="высокая", status="источник",
                                  basis="имя в теге name OSM", source=f"https://www.openstreetmap.org/{s['osm']}")

    # 2) средняя: ПС из документа у названного посёлка, единственный кандидат
    for doc in DOC_SUBSTATIONS:
        cand = _doc_candidates(doc, subs)
        refs = "; ".join(f"{doc_sources[k]} ({loc})" for k, loc in doc["facts"])
        if len(cand) == 1:
            r = rows[cand[0]["osm"]]
            if r["confidence"] == "высокая":
                continue
            r.update(match_name=doc["name"], doc_voltage=doc["kv"], confidence="средняя", status="источник",
                     basis=(f"единственная ПС OSM с классом {doc['need']} кВ, у которой ближайший посёлок — "
                            f"«{doc['place']}» ({cand[0]['nearest_place_km']} км); {doc['why']}"),
                     source=refs)
        elif len(cand) > 1:
            for c in cand:
                r = rows[c["osm"]]
                if r["confidence"] in ("высокая", "средняя"):
                    continue
                r["basis"] = (f"кандидат на {doc['name']}, но таких ПС у «{doc['place']}» {len(cand)} — "
                              "выбрать нельзя; ")

    # 3) низкая: только близость к посёлку
    for s in subs:
        r = rows[s["osm"]]
        if r["confidence"] != "не опознана":
            continue
        km = float(s["nearest_place_km"]) if s["nearest_place_km"] else float("inf")
        if s["nearest_place"] and km <= NEAR_KM:
            r.update(match_name=f"ПС у «{s['nearest_place']}» (название не установлено)", confidence="низкая",
                     status="допущение",
                     basis=r["basis"] + f"только близость: {km} км до посёлка «{s['nearest_place']}» по OSM")
        else:
            r["basis"] = r["basis"] + (f"оснований нет: ближайший посёлок «{s['nearest_place']}» в {km} км "
                                       f"(> {NEAR_KM} км), имени и совпадения с документом нет")
    return list(rows.values())


def read_substations(path: Path = SOURCES_DIR / "osm_substations.csv") -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def export(out: Path = SOURCES_DIR / "substation_ids.csv") -> list[dict]:
    rows = identify(read_substations(), load_doc_sources())
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


if __name__ == "__main__":
    import collections
    rows = export()
    print(dict(collections.Counter(r["confidence"] for r in rows)))
    for r in rows:
        if r["confidence"] in ("высокая", "средняя"):
            print(r["osm"], r["confidence"], r["match_name"], "|", r["basis"])
