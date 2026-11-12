"""Проверочная карта сети по снимку OSM (шаг 2.2): validation/network_map.html.

Линии 110 и 220 кВ — разными цветами (линии 35 кВ и без напряжения — тонко, слоями, по умолчанию выключены);
подстанции — с подписью и уверенностью опознания из data/network_sources/substation_ids.csv,
неопознанные — отдельным цветом. Линии, вошедшие в data/network_real.yaml, отмечены во всплывающей подсказке.

Запуск из корня:  .venv\\Scripts\\python validation\\make_map.py
Карта открывается в браузере; нужен интернет для подложки OSM и библиотеки Leaflet (unpkg).
"""
import csv
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "network_sources"
OUT = Path(__file__).resolve().parent / "network_map.html"

LINE_STYLE = {  # класс → (подпись, цвет, толщина, слой включён)
    "220": ("ВЛ 220 кВ", "#d62728", 4, True),
    "110": ("ВЛ 110 кВ", "#1f6fd1", 3, True),
    "35": ("ВЛ 35 кВ", "#2ca02c", 1.5, False),
    "?": ("без тега напряжения", "#7f7f7f", 1.5, False),
}
CONF_STYLE = {  # уверенность опознания → (цвет, подпись в легенде)
    "высокая": "#1b7a1b",
    "средняя": "#e69500",
    "низкая": "#7fb3e6",
    "не опознана": "#c2185b",
}


def _csv(name):
    with open(SRC / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def collect():
    snap_path = sorted((ROOT / "data" / "osm").glob("power_*.json"))[-1]
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    seg = {int(s["way"]): s for s in _csv("osm_segments.csv")}
    real = yaml.safe_load((ROOT / "data" / "network_real.yaml").read_text(encoding="utf-8"))
    in_model = {}
    for l in real["lines"]:
        for w in l["osm_ways"].split(";"):
            in_model.setdefault(int(w), l["id"].rsplit("-", 1)[0])
    lines = []
    for e in snap["elements"]:
        if e.get("tags", {}).get("power") not in ("line", "cable") or e["id"] not in seg:
            continue
        s = seg[e["id"]]
        cls = s["kv"].split(".")[0] if s["kv"] else "?"
        if cls not in LINE_STYLE:
            cls = "?"
        lines.append({"w": e["id"], "c": cls, "km": float(s["km"]), "circ": s["circuits"], "cab": s["cables"],
                      "tag": s["voltage_tag"], "m": in_model.get(e["id"], ""),
                      "g": [[round(p["lat"], 5), round(p["lon"], 5)] for p in e["geometry"]]})
    ids = {r["osm"]: r for r in _csv("substation_ids.csv")}
    subs = []
    for s in _csv("osm_substations.csv"):
        r = ids.get(s["osm"], {})
        subs.append({"o": s["osm"], "p": s["power"], "lat": float(s["lat"]), "lon": float(s["lon"]),
                     "v": s["voltage_tag"], "n": r.get("match_name", ""), "c": r.get("confidence", "не опознана"),
                     "st": r.get("status", ""), "b": r.get("basis", ""), "src": r.get("source", ""),
                     "pl": s["nearest_place"], "plkm": s["nearest_place_km"]})
    meta = {"snapshot": snap_path.name, "osm_base": snap.get("osm3s", {}).get("timestamp_osm_base", "?"),
            "islands": real["islands"], "summary": real["summary"]}
    return lines, subs, meta


PAGE = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Карта сети OSM</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;height:100%;font:13px/1.35 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;background:#fff;color:#222}
  #map{position:absolute;inset:0}
  .panel{background:rgba(255,255,255,.95);padding:6px 8px;border-radius:6px;box-shadow:0 1px 4px rgba(0,0,0,.3);max-width:260px;font-size:12px}
  .panel summary{cursor:pointer;font-weight:600;font-size:13px}
  .panel h3{margin:0 0 4px;font-size:14px}
  .row{display:flex;align-items:center;gap:6px;margin:2px 0}
  .ln{display:inline-block;width:22px;height:0;border-top-style:solid}
  .dot{display:inline-block;width:11px;height:11px;border-radius:50%;border:1.5px solid #333}
  .muted{color:#666;font-size:12px}
  .lbl{background:rgba(255,255,255,.85);border:none;box-shadow:none;padding:0 3px;font-size:11px;font-weight:600}
  .leaflet-tooltip.lbl:before{display:none}
  @media (max-width:600px){.panel{max-width:200px;font-size:12px}}
</style></head><body><div id="map"></div>
<script>
const LINES = __LINES__, SUBS = __SUBS__, META = __META__;
const LS = __LINE_STYLE__, CS = __CONF_STYLE__;
const map = L.map('map', {preferCanvas: true});
L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom: 18,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">участники OpenStreetMap</a> (ODbL)'}).addTo(map);
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const overlays = {}, bounds = L.latLngBounds([]);
for (const [cls, [label, color, w, on]] of Object.entries(LS)) {
  const g = L.layerGroup(); let n = 0, km = 0;
  for (const l of LINES.filter(x => x.c === cls)) {
    n++; km += l.km;
    const pl = L.polyline(l.g, {color, weight: w, opacity: cls === '?' || cls === '35' ? .7 : .9,
      dashArray: l.m || cls !== '110' ? null : '6 5'});
    pl.bindTooltip(`<b>${label}</b> · way ${l.w}<br>${l.km.toFixed(2)} км; circuits=${esc(l.circ) || '—'}, cables=${esc(l.cab) || '—'}` +
      `<br>${l.m ? 'в модели network_real.yaml: ' + l.m : (cls === '110' ? 'в модель не вошла (вне островов с питанием)' : 'в расчёт не входит')}`, {sticky: true});
    pl.addTo(g); if (cls === '110' || cls === '220') l.g.forEach(p => bounds.extend(p));
  }
  overlays[`<span class="ln" style="border-top:${w}px solid ${color}"></span> ${label} — ${n} уч., ${km.toFixed(0)} км`] = g;
  if (on) g.addTo(map);
}
const allLabels = L.layerGroup(), counts = {};
for (const [conf, color] of Object.entries(CS)) {
  const g = L.layerGroup(); counts[conf] = 0;
  for (const s of SUBS.filter(x => x.c === conf)) {
    counts[conf]++;
    const m = L.circleMarker([s.lat, s.lon], {radius: s.v.includes('220000') ? 8 : 6, color: '#222', weight: 1.5,
      fillColor: color, fillOpacity: .95});
    const title = s.n || (s.p === 'plant' ? 'станция' : 'ПС') + ' (не опознана)';
    m.bindPopup(`<b>${esc(title)}</b><br><span class="muted">${esc(s.o)} · ${s.p === 'plant' ? 'станция' : 'подстанция'}</span>` +
      `<br>voltage: ${esc(s.v) || '—'}<br>уверенность: <b>${esc(conf)}</b> (${esc(s.st) || '—'})` +
      `<br>основание: ${esc(s.b)}` + (s.src ? `<br>источник: ${s.src.split('; ').map(u => `<a href="${esc(u.split(' ')[0])}" target="_blank">${esc(u)}</a>`).join('<br>')}` : '') +
      `<br><a href="https://www.openstreetmap.org/${esc(s.o)}" target="_blank">открыть в OSM</a>`);
    const short = s.n ? s.n.replace(/ \\(название не установлено\\)/, '') : 'не опознана';
    const tip = `${short} · ${conf}`;
    const dir = s.v.includes('220000') ? 'left' : 'right';  // у ПС 220 подпись слева, чтобы не накрывать соседнюю ПС 110
    if (conf === 'высокая' || conf === 'средняя') m.bindTooltip(tip, {permanent: true, direction: dir, className: 'lbl'});
    else {
      m.bindTooltip(tip, {direction: 'right'});
      L.marker([s.lat, s.lon], {opacity: 0, interactive: false}).bindTooltip(tip, {permanent: true, direction: 'right', className: 'lbl'}).addTo(allLabels);
    }
    m.addTo(g);
  }
  overlays[`<span class="dot" style="background:${color}"></span> ПС: ${conf} — ${counts[conf]}`] = g;
  g.addTo(map);
}
overlays['подписи всех остальных ПС'] = allLabels;
L.control.layers(null, overlays, {collapsed: window.innerWidth < 900}).addTo(map);
const info = L.control({position: 'bottomleft'});
info.onAdd = () => {
  const d = L.DomUtil.create('div', 'panel');
  const sm = META.summary;
  d.innerHTML = `<details ${window.innerWidth >= 900 ? 'open' : ''}><summary>Проверочная карта сети (OSM)</summary>
  <div class="muted">снимок ${esc(META.snapshot)}, база OSM ${esc(META.osm_base)}</div>
  <div class="row"><span class="ln" style="border-top:3px dashed #1f6fd1"></span> 110 кВ вне модели (пунктир)</div>
  <div class="row"><span class="dot" style="width:15px;height:15px"></span> крупный кружок — ПС с 220 кВ</div>
  <div>Модель network_real.yaml: ${META.islands.map(i => `${esc(i.feed_name)} — ${i.km} км`).join('; ')}</div>
  <div class="muted">Доля «источник»: длины ${Math.round(sm.lengths_km.share*100)} %, провода ${Math.round(sm.conductors_by_km.share*100)} %,
  нагрузки ${Math.round(sm.loads_by_bus.share*100)} %, генерация ${Math.round(sm.generation_by_mw.share*100)} %</div></details>`;
  L.DomEvent.disableClickPropagation(d); return d;
};
info.addTo(map);
// выравнивание по сети; повторяем после загрузки и при смене размера окна (контейнер мог быть ещё без размера)
const fit = () => { map.invalidateSize(); map.fitBounds(bounds.isValid() ? bounds : [[41.2, 50], [46, 56.5]], {padding: [20, 20], maxZoom: 10}); };
map.setView([44.3, 53.2], 7);
fit();
window.addEventListener('load', fit);
let userMoved = false; map.on('zoomstart dragstart', e => { if (e.originalEvent || e.type === 'dragstart') userMoved = true; });
window.addEventListener('resize', () => { if (!userMoved) fit(); else map.invalidateSize(); });
</script></body></html>
"""


def main() -> int:
    lines, subs, meta = collect()
    dump = lambda x: json.dumps(x, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    html = (PAGE.replace("__LINES__", dump(lines)).replace("__SUBS__", dump(subs)).replace("__META__", dump(meta))
            .replace("__LINE_STYLE__", dump(LINE_STYLE)).replace("__CONF_STYLE__", dump(CONF_STYLE)))
    OUT.write_text(html, encoding="utf-8")
    by = {}
    for l in lines:
        by[l["c"]] = by.get(l["c"], 0) + 1
    print(f"{OUT.name}: {OUT.stat().st_size // 1024} КБ; участков линий {by}; ПС {len(subs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
