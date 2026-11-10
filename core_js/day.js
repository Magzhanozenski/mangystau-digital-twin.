// ===== РИСК НА ЗАВТРА: погода → нагрузка → допустимый ток → N-1 по часам =====

// --- Динамический допустимый ток провода (упрощённый IEEE 738-2012, SI) ---
const COND_PHYS = {
  'АС-240': { D: 0.0216, R20: 0.118e-3 },
  'АС-185': { D: 0.0188, R20: 0.159e-3 },
  'АС-120': { D: 0.0152, R20: 0.244e-3 },
};
const TC_MAX = 70; // допустимая температура провода, °C
function dlrAmp(condName, Ta, wind, solar, Tc = TC_MAX) {
  const c = COND_PHYS[condName];
  const D = c.D, Tf = (Tc + Ta) / 2, He = 30;
  const mu = 1.458e-6 * Math.pow(Tf + 273, 1.5) / (Tf + 383.4);
  const rho = (1.293 - 1.525e-4 * He + 6.379e-9 * He * He) / (1 + 0.00367 * Tf);
  const kf = 2.424e-2 + 7.477e-5 * Tf - 4.407e-9 * Tf * Tf;
  const dT = Math.max(0.1, Tc - Ta);
  const V = Math.max(0, wind);
  const Re = D * rho * V / mu;
  const Kang = 0.855; // ветер под углом ~45° к линии
  const qc1 = (1.01 + 1.35 * Math.pow(Re, 0.52)) * kf * dT * Kang;
  const qc2 = 0.754 * Math.pow(Re, 0.6) * kf * dT * Kang;
  const qcn = 3.645 * Math.sqrt(rho) * Math.pow(D, 0.75) * Math.pow(dT, 1.25);
  const qc = Math.max(qc1, qc2, qcn);
  const eps = 0.5, alpha = 0.5;
  const qr = 17.8 * D * eps * (Math.pow((Tc + 273) / 100, 4) - Math.pow((Ta + 273) / 100, 4));
  const qs = alpha * solar * D;
  const R = c.R20 * (1 + 0.00403 * (Tc - 20));
  return Math.sqrt(Math.max(0, qc + qr - qs) / R);
}
// Статический допустимый ток «по справочнику»: норматив при +25 °C с поправкой на температуру воздуха
function staticAmp(line, Ta) { return lineRatingA(line, Ta); }

// --- Погода на сутки по сценарию ---
// Ориентировочные типичные значения для Актау по месяцам (Tmax, Tmin, ветер) — не климатическая справка, а стартовая точка
const MONTH_TYPICAL = [[2, -4, 6], [3, -3, 6], [9, 2, 6], [17, 9, 5], [24, 16, 5], [30, 21, 5], [33, 24, 4.5], [32, 23, 4.5], [26, 17, 5], [17, 10, 5], [10, 4, 6], [4, -1, 6]];
const MONTH_NAMES = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября', 'октября', 'ноября', 'декабря'];
function typicalPreset(month) { const [Tmax, Tmin, w] = MONTH_TYPICAL[month - 1]; return { name: 'Типичный день месяца', Tmax, Tmin, windDay: w, windNight: Math.max(1, w - 1.5), cloud: month >= 11 || month <= 2 ? 0.5 : 0.15, month }; }
const WEATHER_PRESETS = {
  summer: { name: 'Обычный летний день', Tmax: 36, Tmin: 24, windDay: 4.5, windNight: 3, cloud: 0.1, month: 7 },
  heat: { name: 'Волна жары, штиль', Tmax: 46, Tmin: 31, windDay: 1.5, windNight: 1, cloud: 0, month: 7 },
  heatwind: { name: 'Жара с ветром', Tmax: 44, Tmin: 30, windDay: 8, windNight: 6, cloud: 0.05, month: 7 },
  autumn: { name: 'Ноябрь (типичная погода)', Tmax: 12, Tmin: 5, windDay: 6, windNight: 5, cloud: 0.5, month: 11 },
  frost: { name: 'Мороз', Tmax: -12, Tmin: -22, windDay: 8, windNight: 7, cloud: 0.3, month: 1 },
};
function dayWeather(p) {
  const out = [];
  const summerness = Math.cos(2 * Math.PI * (p.month - 7) / 12); // 1 летом, -1 зимой
  const dayLen = 12 + 3 * summerness, noon = 12.5;
  const Gmax = 650 + 300 * summerness;
  for (let h = 0; h < 24; h++) {
    // минимум ~05:00, максимум ~15:00
    const ph = h <= 5 ? (h + 24 - 15) / 14 : h <= 15 ? (h - 5) / 10 : (h - 15) / 14;
    const shape = h > 5 && h <= 15 ? (1 - Math.cos(Math.PI * ph)) / 2 : (1 + Math.cos(Math.PI * ph)) / 2;
    const T = p.Tmin + (p.Tmax - p.Tmin) * shape;
    const sun = Math.max(0, Math.cos(Math.PI * (h + 0.5 - noon) / dayLen));
    const solar = Gmax * Math.pow(sun, 1.2) * (1 - 0.75 * p.cloud);
    const dayw = Math.max(0, Math.sin(Math.PI * (h - 7) / 14));
    const wind = p.windNight + (p.windDay - p.windNight) * dayw;
    out.push({ h, T, wind, solar });
  }
  return out;
}

// --- «Истинная» модель нагрузки для синтетической истории ---
// Промышленность (нефтедобыча) ~ постоянна, бытовая — с вечерним пиком, кондиционирование растёт с жарой.
const RES_PROFILE = [0.55, 0.5, 0.47, 0.46, 0.47, 0.52, 0.62, 0.72, 0.78, 0.8, 0.8, 0.82, 0.84, 0.84, 0.83, 0.84, 0.88, 0.95, 1.0, 1.0, 0.97, 0.9, 0.78, 0.65];
const OCC = h => 0.55 + 0.45 * Math.max(0, Math.sin(Math.PI * (h - 8) / 16));
function trueLoad(h, T, Tlag, weekend) {
  const ind = 88;
  const res = 48 * RES_PROFILE[h] * (weekend ? 1.06 : 1);
  const comm = weekend ? 4 : 12 * Math.max(0, Math.sin(Math.PI * (h - 7) / 12));
  const cool = 3.1 * Math.max(0, Tlag - 24) * OCC(h) + 0.06 * Math.pow(Math.max(0, Tlag - 30), 2) * OCC(h);
  const heat = 1.3 * Math.max(0, 8 - Tlag) * (0.7 + 0.3 * RES_PROFILE[h]);
  return ind * (weekend ? 0.97 : 1) + res + comm + cool + heat;
}
function makeHistory(years = 2, seed = 21) {
  const rnd = mulberry32(seed);
  const gauss = () => { let u = 0, v = 0; while (!u) u = rnd(); v = rnd(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  const rows = [];
  let anom = 0, eps = 0, Tprev = [10, 10, 10];
  for (let d = 0; d < 365 * years; d++) {
    const doy = d % 365;
    const summer = Math.max(0, Math.sin(2 * Math.PI * (doy - 105) / 365));
    anom = 0.8 * anom + gauss() * (2.0 + 1.4 * summer);
    const mean = 12 + 15 * Math.sin(2 * Math.PI * (doy - 105) / 365) + anom + (anom > 0 ? 0.2 * anom * summer : 0);
    const amp = 4.5 + 2.5 * Math.sin(2 * Math.PI * (doy - 105) / 365) + rnd() * 2;
    const month = Math.floor(doy / 30.4) + 1;
    const windD = 2 + rnd() * 7, windN = Math.max(0.5, windD - 1 - rnd() * 2);
    const w = dayWeather({ Tmax: mean + amp, Tmin: mean - amp, windDay: windD, windNight: windN, cloud: rnd() * 0.6, month });
    const weekend = (d % 7) >= 5;
    for (const x of w) {
      const Tlag = (x.T + Tprev[0] + Tprev[1]) / 3;
      Tprev = [x.T, Tprev[0], Tprev[1]];
      eps = 0.85 * eps + gauss() * 1.6;
      rows.push({ d, h: x.h, T: x.T, Tlag, Tmax: mean + amp, weekend, load: trueLoad(x.h, x.T, Tlag, weekend) + eps });
    }
  }
  return rows;
}

// --- Признаки и модели прогноза нагрузки ---
function loadFeatures(r) {
  return [
    Math.sin(2 * Math.PI * r.h / 24), Math.cos(2 * Math.PI * r.h / 24),
    Math.sin(4 * Math.PI * r.h / 24), Math.cos(4 * Math.PI * r.h / 24),
    r.weekend ? 1 : 0, (r.T - 12) / 15, (r.Tlag - 12) / 15, (r.Tmax - 15) / 15,
  ];
}
// Базовая модель: линейная регрессия (час + температура) — то, что обычно используют как ориентир
function fitLinear(rows) {
  const X = rows.map(r => { const f = new Array(24).fill(0); f[r.h] = 1; return [...f, r.weekend ? 1 : 0, r.T]; });
  const y = rows.map(r => r.load);
  const n = X[0].length;
  const A = Array.from({ length: n }, () => new Array(n).fill(0)), b = new Array(n).fill(0);
  X.forEach((x, k) => { for (let i = 0; i < n; i++) { b[i] += x[i] * y[k]; for (let j = 0; j < n; j++) A[i][j] += x[i] * x[j]; } });
  for (let i = 0; i < n; i++) A[i][i] += 1e-3;
  const w = solveLinear(A, b);
  return { predict: r => { const f = new Array(24).fill(0); f[r.h] = 1; const x = [...f, r.weekend ? 1 : 0, r.T]; return x.reduce((s, xi, i) => s + xi * w[i], 0); } };
}
class MLPReg {
  constructor(nIn, nH, seed = 5) {
    const rnd = mulberry32(seed), rn = () => rnd() * 2 - 1;
    this.W1 = Array.from({ length: nH }, () => Array.from({ length: nIn }, () => rn() * Math.sqrt(2 / nIn)));
    this.b1 = new Array(nH).fill(0); this.W2 = Array.from({ length: nH }, () => rn() * Math.sqrt(1 / nH)); this.b2 = 0;
    this.nH = nH; this.nIn = nIn; this.m = {}; this.v = {}; this.t = 0;
  }
  fwd(x) { const h = this.W1.map((w, j) => Math.tanh(w.reduce((s, wi, i) => s + wi * x[i], this.b1[j]))); return { h, y: h.reduce((s, hj, j) => s + hj * this.W2[j], this.b2) }; }
  predictRaw(x) { return this.fwd(x).y; }
  step(batch, lr) {
    const { nH, nIn } = this;
    const gW1 = Array.from({ length: nH }, () => new Array(nIn).fill(0)), gb1 = new Array(nH).fill(0), gW2 = new Array(nH).fill(0); let gb2 = 0;
    for (const d of batch) {
      const { h, y } = this.fwd(d.x); const dy = y - d.y; gb2 += dy;
      for (let j = 0; j < nH; j++) { gW2[j] += dy * h[j]; const dh = dy * this.W2[j] * (1 - h[j] * h[j]); gb1[j] += dh; for (let i = 0; i < nIn; i++) gW1[j][i] += dh * d.x[i]; }
    }
    const N = batch.length; this.t++;
    const B1 = 0.9, B2 = 0.999, t = this.t;
    const up = (k, val, g) => { const m = this.m[k] = B1 * (this.m[k] || 0) + (1 - B1) * g; const v = this.v[k] = B2 * (this.v[k] || 0) + (1 - B2) * g * g; return val - lr * (m / (1 - B1 ** t)) / (Math.sqrt(v / (1 - B2 ** t)) + 1e-8); };
    for (let j = 0; j < nH; j++) { for (let i = 0; i < nIn; i++) this.W1[j][i] = up(`a${j}_${i}`, this.W1[j][i], gW1[j][i] / N); this.b1[j] = up(`b${j}`, this.b1[j], gb1[j] / N); this.W2[j] = up(`c${j}`, this.W2[j], gW2[j] / N); }
    this.b2 = up('d', this.b2, gb2 / N);
  }
}
const LOAD_MU = 165, LOAD_SD = 30;
function* trainLoadModel(train, epochs = 6, nH = 16) {
  const data = train.map(r => ({ x: loadFeatures(r), y: (r.load - LOAD_MU) / LOAD_SD }));
  const m = new MLPReg(data[0].x.length, nH);
  const rnd = mulberry32(9);
  for (let e = 0; e < epochs; e++) {
    for (let i = data.length - 1; i > 0; i--) { const j = Math.floor(rnd() * (i + 1)); [data[i], data[j]] = [data[j], data[i]]; }
    const lr = 0.01 * Math.pow(0.6, e);
    for (let k = 0; k < data.length; k += 32) m.step(data.slice(k, k + 32), lr);
    yield { epoch: e + 1, epochs };
  }
  m.predict = r => m.predictRaw(loadFeatures(r)) * LOAD_SD + LOAD_MU;
  return m;
}
function metrics(model, rows) {
  let ae = 0, ape = 0, se = 0; const res = [];
  rows.forEach(r => { const p = model.predict(r); const e = r.load - p; ae += Math.abs(e); ape += Math.abs(e) / r.load; se += e * e; res.push(e); });
  res.sort((a, b) => a - b);
  const q = f => res[Math.floor(f * (res.length - 1))];
  return { mae: ae / rows.length, mape: 100 * ape / rows.length, rmse: Math.sqrt(se / rows.length), q10: q(0.1), q90: q(0.9), q95: q(0.95) };
}
function hotMetrics(model, rows) { return metrics(model, rows.filter(r => r.T >= 35)); }

// --- Суточный анализ N-1 ---
const CONTINGENCIES = [{ id: 'base', label: 'Нормальная схема', linesOut: [], gensOut: [] },
  ...LINES.map(l => ({ id: l.id, label: `${l.id} · ${l.name}`, linesOut: [l.id], gensOut: [] })),
  { id: 'G2', label: 'G2 · ГТЭС Жанаозен', linesOut: [], gensOut: ['G2'] },
  { id: 'G3', label: 'G3 · ГТУ Жетыбай', linesOut: [], gensOut: ['G3'] }];

function forecastDay(weather, model, weekend, q90) {
  const Tm = Math.max(...weather.map(w => w.T));
  let Tprev = [weather[23].T, weather[22].T]; // предыдущие сутки ~ такие же
  return weather.map(w => {
    const Tlag = (w.T + Tprev[0] + Tprev[1]) / 3; Tprev = [w.T, Tprev[0]];
    const r = { h: w.h, T: w.T, Tlag, Tmax: Tm, weekend };
    const p50 = model.predict(r);
    return { ...w, Tlag, p50, p90: p50 + q90, truth: trueLoad(w.h, w.T, Tlag, weekend) };
  });
}
function ratingsFor(w, mode) {
  const out = {};
  LINES.forEach(l => { out[l.id] = mode === 'dlr' ? dlrAmp(l.c, w.T, w.wind, w.solar) : staticAmp(l, w.T); });
  return out;
}
function classify(r) {
  if (r.unknown) return 3;
  if (!r.converged || r.critical) return 2;
  if ((r.warns || []).some(x => !/предел реактивной/.test(x))) return 1;
  return 0;
}
function analyzeDay(fc, mode, useP90) {
  return fc.map(w => {
    const ratingA = ratingsFor(w, mode);
    const load = useP90 ? w.p90 : w.p50;
    const cells = CONTINGENCIES.map(c => {
      const r = runPowerFlow({ T: w.T, load, linesOut: c.linesOut, gensOut: c.gensOut, ratingA });
      return { c, r, st: classify(r), maxL: r.converged ? r.maxLoading : 9, minV: r.converged ? r.minV : 0 };
    });
    return { w, load, ratingA, cells, nCrit: cells.filter(x => x.st >= 2).length, nWarn: cells.filter(x => x.st === 1).length };
  });
}

if (typeof module !== 'undefined') module.exports = { dlrAmp, staticAmp, dayWeather, WEATHER_PRESETS, makeHistory, fitLinear, trainLoadModel, metrics, hotMetrics, forecastDay, analyzeDay, CONTINGENCIES, trueLoad };

// --- Превентивная мера на час: подбор загрузки местной генерации и уставок напряжения (N-1-безопасный режим) ---
function hourScore(cells, ctrl) {
  let s = (Math.abs(ctrl.gP.G2 - 35) + Math.abs(ctrl.gP.G3 - 15)) * 0.3 + (ctrl.vset.G2 !== 1.02 ? 1 : 0);
  cells.forEach(x => { s += x.st >= 2 ? 1000 + 300 * Math.max(0, (x.maxL || 0) - 1) : x.st === 1 ? 10 : 0; });
  return s;
}
function evalHour(hr, ctrl, list) {
  return list.map(c => {
    const r = runPowerFlow({ T: hr.w.T, load: hr.load, linesOut: c.linesOut, gensOut: c.gensOut, ratingA: hr.ratingA, ctrl });
    return { c, r, st: classify(r), maxL: r.converged ? r.maxLoading : 9, minV: r.converged ? r.minV : 0 };
  });
}
function preventiveHour(hr) {
  const base = defaultCtrl();
  const baseScore = hourScore(hr.cells, base);
  let relevantIds = new Set(hr.cells.filter(x => x.st > 0 || x.c.id === 'base').map(x => x.c.id));
  let best = base, bestCells = hr.cells, bestFull = baseScore;
  for (let round = 0; round < 3; round++) {
    const relevant = CONTINGENCIES.filter(c => relevantIds.has(c.id));
    let cand = null, candS = Infinity;
    for (let g2 = 0; g2 <= 50; g2 += 5) for (const g3 of [0, 5, 10, 15, 20, 25, 28]) for (const vs of [1.02, 1.05]) {
      const ctrl = { gP: { G2: g2, G3: g3 }, vset: { G1: vs === 1.05 ? 1.05 : 1.04, G2: vs, G3: vs }, bsk: {}, shed: {} };
      const sc = hourScore(evalHour(hr, ctrl, relevant), ctrl);
      if (sc < candS - 1e-9) { candS = sc; cand = ctrl; }
    }
    const full = evalHour(hr, cand, CONTINGENCIES);
    const fullS = hourScore(full, cand);
    if (fullS < bestFull - 1e-9) { best = cand; bestCells = full; bestFull = fullS; }
    const newBad = full.filter(x => x.st > 0 && !relevantIds.has(x.c.id));
    if (!newBad.length) break;
    newBad.forEach(x => relevantIds.add(x.c.id));
  }
  const changed = best !== base;
  return { ctrl: best, changed, cells: bestCells, nCrit: bestCells.filter(x => x.st === 2).length, nWarn: bestCells.filter(x => x.st === 1).length };
}
if (typeof module !== 'undefined') Object.assign(module.exports, { preventiveHour });

// --- Суточный график местной генерации с ограничением скорости набора/сброса мощности ---
function rampSchedule(req) {
  // req[h] = {G2, G3} требуемые значения (или null — базовая загрузка). Возвращает выполнимый график.
  const base = defaultCtrl().gP, out = {};
  ['G2', 'G3'].forEach(id => {
    const ramp = GENS.find(g => g.id === id).ramp;
    const tgt = req.map(r => r ? r[id] : base[id]);
    const S = tgt.slice();
    const fixed = req.map(r => !!r);
    // заранее набираем мощность перед опасными часами и плавно снижаем после них
    for (let it = 0; it < 3; it++) {
      for (let h = 22; h >= 0; h--) if (!fixed[h]) { if (S[h] < S[h + 1] - ramp) S[h] = S[h + 1] - ramp; if (S[h] > S[h + 1] + ramp) S[h] = S[h + 1] + ramp; }
      for (let h = 1; h < 24; h++) if (!fixed[h]) { if (S[h] < S[h - 1] - ramp) S[h] = S[h - 1] - ramp; if (S[h] > S[h - 1] + ramp) S[h] = S[h - 1] + ramp; }
    }
    out[id] = S;
  });
  let feasible = true;
  ['G2', 'G3'].forEach(id => { const ramp = GENS.find(g => g.id === id).ramp; for (let h = 1; h < 24; h++) if (Math.abs(out[id][h] - out[id][h - 1]) > ramp + 1e-9) feasible = false; });
  return { ...out, feasible };
}
if (typeof module !== 'undefined') Object.assign(module.exports, { rampSchedule, typicalPreset, MONTH_NAMES });
