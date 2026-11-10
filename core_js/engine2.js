// ===== MANGYSTAU DIGITAL TWIN — расчётное ядро v2 =====
// Схема иллюстративная (упрощённая модель 110 кВ), параметры — типовые для проводов АС.
const BASE = 100, KV = 110;
const ZB = KV * KV / BASE;

const COND = {
  'АС-240': { r: 0.118, x: 0.405, b: 2.81e-6, I: 610 },
  'АС-185': { r: 0.159, x: 0.413, b: 2.75e-6, I: 510 },
  'АС-120': { r: 0.244, x: 0.427, b: 2.66e-6, I: 390 },
};
const BUSES = [
  { id: 1, name: 'МАЭК (Актау)', short: 'МАЭК', x: 40, y: 220, share: 0 },
  { id: 2, name: 'ПС Актау-Город', short: 'Актау', x: 100, y: 170, share: 55 / 160 },
  { id: 3, name: 'ПС Промзона', short: 'Промзона', x: 125, y: 245, share: 30 / 160 },
  { id: 4, name: 'ПС Форт-Шевченко', short: 'Форт-Шевч.', x: 55, y: 50, share: 12 / 160 },
  { id: 5, name: 'ПС Жетыбай', short: 'Жетыбай', x: 220, y: 215, share: 18 / 160 },
  { id: 6, name: 'ПС Жанаозен', short: 'Жанаозен', x: 310, y: 262, share: 30 / 160 },
  { id: 7, name: 'ПС Шетпе', short: 'Шетпе', x: 190, y: 110, share: 6 / 160 },
  { id: 8, name: 'ПС Бейнеу', short: 'Бейнеу', x: 360, y: 45, share: 9 / 160 },
];
const LINES = [
  { id: 'L1', f: 1, t: 2, km: 15, c: 'АС-240', name: 'МАЭК — Актау' },
  { id: 'L2', f: 1, t: 3, km: 22, c: 'АС-240', name: 'МАЭК — Промзона' },
  { id: 'L3', f: 2, t: 3, km: 14, c: 'АС-185', name: 'Актау — Промзона' },
  { id: 'L4', f: 1, t: 7, km: 95, c: 'АС-185', name: 'МАЭК — Шетпе' },
  { id: 'L5', f: 7, t: 4, km: 75, c: 'АС-120', name: 'Шетпе — Форт-Шевченко' },
  { id: 'L6', f: 3, t: 5, km: 70, c: 'АС-185', name: 'Промзона — Жетыбай' },
  { id: 'L7', f: 5, t: 6, km: 60, c: 'АС-185', name: 'Жетыбай — Жанаозен' },
  { id: 'L8', f: 2, t: 5, km: 85, c: 'АС-185', name: 'Актау — Жетыбай' },
  { id: 'L9', f: 6, t: 8, km: 150, c: 'АС-120', name: 'Жанаозен — Бейнеу' },
  { id: 'L10', f: 7, t: 8, km: 175, c: 'АС-120', name: 'Шетпе — Бейнеу' },
  { id: 'L11', f: 2, t: 4, km: 110, c: 'АС-120', name: 'Актау — Форт-Шевченко' },
];
const GENS = [
  { id: 'G1', bus: 1, name: 'МАЭК (ТЭЦ/ТЭС)', P: null, Pmax: 330, Qmin: -150, Qmax: 220, V: 1.04, slack: true, ramp: 60 },
  { id: 'G2', bus: 6, name: 'ГТЭС Жанаозен', P: 35, Pmax: 50, Qmin: -20, Qmax: 30, V: 1.02, ramp: 20 },
  { id: 'G3', bus: 5, name: 'ГТУ Жетыбай', P: 15, Pmax: 28, Qmin: -10, Qmax: 18, V: 1.02, ramp: 12 },
];
const VMIN_CRIT = 0.90, VMIN_WARN = 0.95, VMAX_WARN = 1.05, VMAX_CRIT = 1.10;

function defaultLoadForTemp(T) { return 150 * (1 + 0.011 * Math.max(0, T - 22) + 0.006 * Math.max(0, 5 - T)); }
function ratingFactor(T) { return Math.min(1.25, Math.sqrt(Math.max(0.05, (70 - T) / 45))); }
function lineRatingA(line, T) { return COND[line.c].I * ratingFactor(T); }
function defaultCtrl() { return { gP: { G2: 35, G3: 15 }, vset: { G1: 1.04, G2: 1.02, G3: 1.02 }, bsk: {}, shed: {} }; }
function cloneCtrl(c) { return JSON.parse(JSON.stringify(c)); }
function normScn(s) {
  return { T: s.T ?? 25, load: s.load ?? defaultLoadForTemp(s.T ?? 25), linesOut: s.linesOut || [], gensOut: s.gensOut || [], ctrl: s.ctrl || defaultCtrl(), ratingA: s.ratingA || null };
}
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

function solveLinear(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    if (Math.abs(M[p][c]) < 1e-12) return null;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) { const f = M[r][c] / M[c][c]; if (f) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k]; }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) { let s = M[r][n]; for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k]; x[r] = s / M[r][r]; }
  return x;
}

// ---------- построение острова: матрица проводимостей ----------
function buildIsland(ids, s, lineIn) {
  const idx = {}; ids.forEach((id, i) => idx[id] = i);
  const n = ids.length;
  const G = Array.from({ length: n }, () => new Array(n).fill(0));
  const B = Array.from({ length: n }, () => new Array(n).fill(0));
  const rT = 1 + 0.00403 * (s.T + 15 - 20); // провод нагрет ≈ на 15 °C выше воздуха
  const lp = {};
  lineIn.filter(l => idx[l.f] !== undefined).forEach(l => {
    const c = COND[l.c];
    const R = c.r * rT * l.km / ZB, X = c.x * l.km / ZB, Bc = c.b * l.km * ZB;
    const d = R * R + X * X, g = R / d, b = -X / d, i = idx[l.f], j = idx[l.t];
    lp[l.id] = { g, b, Bc, i, j };
    G[i][i] += g; B[i][i] += b + Bc / 2; G[j][j] += g; B[j][j] += b + Bc / 2;
    G[i][j] -= g; B[i][j] -= b; G[j][i] -= g; B[j][i] -= b;
  });
  return { ids, idx, n, G, B, lp };
}

// ---------- Ньютон–Рафсон (полярные координаты) ----------
function calcPQ(isl, V, th) {
  const { n, G, B } = isl, P = new Array(n).fill(0), Q = new Array(n).fill(0);
  for (let i = 0; i < n; i++) for (let k = 0; k < n; k++) {
    if (G[i][k] === 0 && B[i][k] === 0) continue;
    const a = th[i] - th[k], co = Math.cos(a), si = Math.sin(a);
    P[i] += V[i] * V[k] * (G[i][k] * co + B[i][k] * si);
    Q[i] += V[i] * V[k] * (G[i][k] * si - B[i][k] * co);
  }
  return { P, Q };
}
function buildJ(isl, pvq, pq, V, th, P, Q) {
  const { G, B } = isl, m = pvq.length + pq.length;
  const J = Array.from({ length: m }, () => new Array(m).fill(0));
  const c = (i, k) => Math.cos(th[i] - th[k]), s = (i, k) => Math.sin(th[i] - th[k]);
  const dPdth = (i, k) => i === k ? -Q[i] - B[i][i] * V[i] * V[i] : V[i] * V[k] * (G[i][k] * s(i, k) - B[i][k] * c(i, k));
  const dPdV = (i, k) => i === k ? P[i] / V[i] + G[i][i] * V[i] : V[i] * (G[i][k] * c(i, k) + B[i][k] * s(i, k));
  const dQdth = (i, k) => i === k ? P[i] - G[i][i] * V[i] * V[i] : -V[i] * V[k] * (G[i][k] * c(i, k) + B[i][k] * s(i, k));
  const dQdV = (i, k) => i === k ? Q[i] / V[i] - B[i][i] * V[i] : V[i] * (G[i][k] * s(i, k) - B[i][k] * c(i, k));
  pvq.forEach((i, r) => { pvq.forEach((k, cc) => J[r][cc] = dPdth(i, k)); pq.forEach((k, cc) => J[r][pvq.length + cc] = dPdV(i, k)); });
  pq.forEach((i, r) => { pvq.forEach((k, cc) => J[pvq.length + r][cc] = dQdth(i, k)); pq.forEach((k, cc) => J[pvq.length + r][pvq.length + cc] = dQdV(i, k)); });
  return J;
}
// типы узлов и заданные мощности для острова при коэффициенте нагрузки lam
function islandSetup(isl, gens, Pd, Qd, lam) {
  const n = isl.n, type = new Array(n).fill('PQ'), Vs = new Array(n).fill(1), Pg = new Array(n).fill(0), Qf = new Array(n).fill(0);
  gens.forEach(g => {
    const k = isl.idx[g.bus];
    if (g.slack) { type[k] = 'SL'; Vs[k] = g.Vset; return; }
    Pg[k] += g.Pset / BASE;
    if (g.fixedQ === null) { if (type[k] !== 'SL') { type[k] = 'PV'; Vs[k] = g.Vset; } }
    else Qf[k] += g.fixedQ / BASE;
  });
  const pvq = [], pq = [];
  for (let i = 0; i < n; i++) { if (type[i] !== 'SL') pvq.push(i); if (type[i] === 'PQ') pq.push(i); }
  const Psp = Pg.map((p, i) => p - lam * Pd[i]), Qsp = Qf.map((q, i) => q - lam * Qd[i]);
  return { type, Vs, pvq, pq, Psp, Qsp };
}
function nr(isl, setup, V0, th0, damped, maxIt) {
  const { Vs, pvq, pq, Psp, Qsp, type } = setup;
  const V = V0.map((v, i) => type[i] === 'PQ' ? v : Vs[i]), th = th0.slice();
  for (let it = 0; it < maxIt; it++) {
    const { P, Q } = calcPQ(isl, V, th);
    const mis = [...pvq.map(i => Psp[i] - P[i]), ...pq.map(i => Qsp[i] - Q[i])];
    const mx = mis.reduce((a, b) => Math.max(a, Math.abs(b)), 0);
    if (mx < 1e-8) return { ok: true, V, th, P, Q, it };
    if (!isFinite(mx) || mx > 1e3) return { ok: false };
    const J = buildJ(isl, pvq, pq, V, th, P, Q);
    const dx = solveLinear(J, mis);
    if (!dx) return { ok: false };
    let a = 1;
    if (damped) { const mth = Math.max(0, ...pvq.map((_, r) => Math.abs(dx[r]))), mv = Math.max(0, ...pq.map((_, r) => Math.abs(dx[pvq.length + r]))); a = Math.min(1, 0.3 / Math.max(mth, 1e-9), 0.08 / Math.max(mv, 1e-9)); }
    pvq.forEach((i, r) => th[i] += a * dx[r]);
    pq.forEach((i, r) => V[i] += a * dx[pvq.length + r]);
    if (V.some(v => v < 0.2 || v > 2)) return { ok: false };
  }
  return { ok: false };
}
// решение с учётом ограничений генераторов по реактивной мощности
function solveQ(isl, gens, Pd, Qd, lam, V0, th0, damped) {
  let sol = null;
  for (let outer = 0; outer < 5; outer++) {
    const setup = islandSetup(isl, gens, Pd, Qd, lam);
    sol = nr(isl, setup, V0, th0, damped, damped ? 80 : 30);
    if (!sol.ok) return { ok: false };
    let changed = false;
    gens.forEach(g => {
      if (g.slack || g.fixedQ !== null) return;
      const k = isl.idx[g.bus];
      const Qg = (sol.Q[k] + lam * Qd[k]) * BASE;
      if (Qg > g.Qmax + 1e-6) { g.fixedQ = g.Qmax; changed = true; } else if (Qg < g.Qmin - 1e-6) { g.fixedQ = g.Qmin; changed = true; }
    });
    if (!changed) { sol.setup = setup; return sol; }
    V0 = sol.V; th0 = sol.th;
  }
  return { ok: false };
}
// наименьшее сингулярное число якобиана (обратные итерации) — признак «носа» P–V кривой
function sigmaMin(isl, sol) {
  const { pvq, pq } = sol.setup;
  const J = buildJ(isl, pvq, pq, sol.V, sol.th, sol.P, sol.Q);
  const m = J.length; if (!m) return 1;
  const JT = J[0].map((_, c) => J.map(r => r[c]));
  const JTJ = JT.map(r => J[0].map((_, c) => r.reduce((s, v, k) => s + v * J[k][c], 0)));
  let x = new Array(m).fill(1), lam = 0;
  for (let it = 0; it < 30; it++) {
    const y = solveLinear(JTJ, x); if (!y) return 0;
    const nrm = Math.hypot(...y); lam = nrm / Math.hypot(...x); x = y.map(v => v / nrm);
  }
  return 1 / Math.sqrt(lam);
}
// метод продолжения по нагрузке: отличает численную несходимость от физического предела
function continuation(isl, gens, Pd, Qd, lamTarget, keepCurve) {
  const n = isl.n, flatV = new Array(n).fill(1), flatT = new Array(n).fill(0);
  const reset = () => gens.forEach(g => g.fixedQ = null);
  reset();
  let lam = Math.min(0.25, lamTarget), sol = solveQ(isl, gens, Pd, Qd, lam, flatV, flatT, true);
  if (!sol.ok) return { status: 'numerical' };
  const curve = [{ lam, minV: Math.min(...sol.V) }];
  const sig0 = sigmaMin(isl, sol);
  let step = 0.2;
  while (lam < lamTarget - 1e-9) {
    const tryLam = Math.min(lamTarget, lam + step);
    const saved = gens.map(g => g.fixedQ);
    const s2 = solveQ(isl, gens, Pd, Qd, tryLam, sol.V, sol.th, true);
    if (s2.ok) { lam = tryLam; sol = s2; curve.push({ lam, minV: Math.min(...sol.V) }); if (keepCurve) step = Math.min(step * 1.5, 0.2); }
    else { gens.forEach((g, i) => g.fixedQ = saved[i]); step /= 2; if (step < 0.003) break; }
  }
  if (lam >= lamTarget - 1e-9) return { status: 'ok', sol, curve, lambdaMax: null };
  const sig = sigmaMin(isl, sol);
  return { status: 'pv_limit', sol, curve, lambdaMax: lam, sigmaRatio: sig / sig0, confirmed: sig / sig0 < 0.25 };
}

// ---------- расчёт режима всей сети ----------
function runPowerFlow(input) {
  const s = normScn(input), T = s.T, ctrl = s.ctrl;
  const lineIn = LINES.filter(l => !s.linesOut.includes(l.id));
  // компоненты связности (острова)
  const adj = {}; BUSES.forEach(b => adj[b.id] = []);
  lineIn.forEach(l => { adj[l.f].push(l.t); adj[l.t].push(l.f); });
  const comp = {}, comps = [];
  BUSES.forEach(b => { if (comp[b.id] !== undefined) return; const c = []; const q = [b.id]; comp[b.id] = comps.length; while (q.length) { const u = q.shift(); c.push(u); adj[u].forEach(v => { if (comp[v] === undefined) { comp[v] = comps.length; q.push(v); } }); } comps.push(c.sort((a, b) => a - b)); });

  const tan = 0.42 + 0.005 * Math.max(0, T - 25);
  const loadBus = {}; let shedMW = 0;
  BUSES.forEach(b => { const p0 = s.load * b.share, sh = ctrl.shed[b.id] || 0; shedMW += p0 * sh; loadBus[b.id] = { P: p0 * (1 - sh), P0: p0 }; });

  const res = { scn: s, loadBus, shedMW, unserved: 0, achr: 0, islands: [], V: {}, ang: {}, gen: {}, flows: {}, live: [], status: 'ok' };
  const reasons = [], warns = [];
  let score = 0, worstStatus = 'ok', lambdaMax = null, pvInfo = null;

  comps.forEach(ids => {
    const gens = GENS.filter(g => !s.gensOut.includes(g.id) && ids.includes(g.bus)).map(g => ({ ...g, Pset: g.slack ? 0 : Math.min(g.Pmax, ctrl.gP[g.id] ?? g.P), Vset: ctrl.vset[g.id] ?? g.V, fixedQ: null, slack: false }));
    const names = ids.map(id => BUSES[id - 1].name);
    const islLoad = ids.reduce((a, id) => a + loadBus[id].P, 0);
    if (!gens.length) {
      if (islLoad > 0.01) { res.unserved += islLoad; reasons.push(`Обесточены: ${names.join(', ')} — ${islLoad.toFixed(1)} МВт без питания.`); score += 50 * islLoad; }
      ids.forEach(id => { res.V[id] = 0; res.ang[id] = 0; });
      return;
    }
    // балансирующий источник острова: МАЭК, иначе самый мощный местный генератор
    const slack = gens.find(g => g.id === 'G1') || gens.slice().sort((a, b) => b.Pmax - a.Pmax)[0];
    slack.slack = true;
    const isMain = slack.id === 'G1';
    // АЧР: если мощности острова не хватает, часть нагрузки отключается автоматически
    let achrFrac = 0;
    const cap = gens.reduce((a, g) => a + g.Pmax, 0);
    if (!isMain && islLoad * 1.03 > cap) achrFrac = Math.min(1, 1 - cap * 0.95 / islLoad);
    const isl = buildIsland(ids, s, lineIn);
    const Pd = new Array(isl.n).fill(0), Qd = new Array(isl.n).fill(0);
    ids.forEach(id => { const k = isl.idx[id], p = loadBus[id].P * (1 - achrFrac); Pd[k] = p / BASE; Qd[k] = p * tan / BASE; });
    ids.forEach(id => { const k = isl.idx[id]; isl.B[k][k] += (ctrl.bsk[id] || 0) / BASE; });
    if (achrFrac > 0) { res.achr += islLoad * achrFrac; reasons.push(`Островной режим с дефицитом: ${names.join(', ')} питаются только от ${slack.name}; АЧР отключает ≈${(islLoad * achrFrac).toFixed(1)} МВт.`); score += 50 * islLoad * achrFrac; }
    else if (!isMain) warns.push(`Островной режим: ${names.join(', ')} ${names.length > 1 ? 'отделились' : 'отделилась'} от МАЭК и питаются от ${gens.map(g => g.name).join(' и ')}.`);
    res.islands.push({ ids, slack: slack.id, main: isMain, achrMW: islLoad * achrFrac });

    // 1) обычный Ньютон с плоского старта; 2) с демпфированием; 3) продолжение по нагрузке
    const flatV = new Array(isl.n).fill(1), flatT = new Array(isl.n).fill(0);
    let sol = solveQ(isl, gens, Pd, Qd, 1, flatV, flatT, false), how = 'nr';
    if (!sol.ok) { gens.forEach(g => g.fixedQ = null); sol = solveQ(isl, gens, Pd, Qd, 1, flatV, flatT, true); how = 'damped'; }
    if (!sol.ok) {
      const cr = continuation(isl, gens, Pd, Qd, 1, false);
      if (cr.status === 'ok') { sol = cr.sol; how = 'continuation'; }
      else if (cr.status === 'pv_limit') {
        worstStatus = 'pv_limit'; lambdaMax = lambdaMax === null ? cr.lambdaMax : Math.min(lambdaMax, cr.lambdaMax);
        pvInfo = cr;
        reasons.push(`Предел передаваемой мощности: режим существует только до ${(cr.lambdaMax * 100).toFixed(0)}% заданной нагрузки (метод продолжения${cr.confirmed ? '; якобиан у предела почти вырожден — это «нос» P–V кривой' : '; вырождение якобиана не подтверждено — нужна проверка'}).`);
        score += 1e4 * (1 - cr.lambdaMax) + 1e3;
        sol = cr.sol; how = 'pv';
      } else {
        if (worstStatus !== 'pv_limit') worstStatus = 'numerical';
        warns.push(`Расчёт по острову (${names.join(', ')}) не сошёлся ни одним из трёх методов — результат не определён, требуется проверка.`);
        score += 5e3;
        ids.forEach(id => { res.V[id] = null; });
        return;
      }
    }
    res.islands[res.islands.length - 1].method = how;
    const lamUsed = how === 'pv' ? pvInfo.lambdaMax : 1;
    ids.forEach(id => { const k = isl.idx[id]; res.V[id] = sol.V[k]; res.ang[id] = sol.th[k] * 180 / Math.PI; res.live.push(id); });
    gens.forEach(g => {
      const k = isl.idx[g.bus];
      const others = gens.filter(h => h.bus === g.bus && h !== g).reduce((a, h) => a + h.Pset / BASE, 0);
      const P = g.slack ? (sol.P[k] + lamUsed * Pd[k] - others) * BASE : g.Pset;
      const Q = g.fixedQ !== null ? g.fixedQ : (sol.Q[k] + lamUsed * Qd[k]) * BASE;
      res.gen[g.id] = { P, Q, Pmax: g.Pmax, Qmax: g.Qmax, Qmin: g.Qmin, atQlim: g.fixedQ !== null, slack: g.slack };
    });
    Object.entries(isl.lp).forEach(([id, { g, b, Bc, i, j }]) => {
      const flow = (a, c) => {
        const Va = sol.V[a], Vc = sol.V[c], d = sol.th[a] - sol.th[c];
        const P = Va * Va * g - Va * Vc * (g * Math.cos(d) + b * Math.sin(d));
        const Q = -Va * Va * (b + Bc / 2) - Va * Vc * (g * Math.sin(d) - b * Math.cos(d));
        return { P: P * BASE, Q: Q * BASE, I: Math.hypot(P, Q) * BASE / (Math.sqrt(3) * KV * Va) * 1000 };
      };
      const l = LINES.find(x => x.id === id);
      const Imax = (s.ratingA && s.ratingA[id]) || lineRatingA(l, T);
      const a = flow(i, j), z = flow(j, i), I = Math.max(a.I, z.I);
      res.flows[id] = { P: a.P, Q: a.Q, I, Imax, loading: I / Imax, loss: a.P + z.P };
    });
  });
  LINES.forEach(l => { if (!res.flows[l.id]) res.flows[l.id] = { out: s.linesOut.includes(l.id), dead: !s.linesOut.includes(l.id), Imax: (s.ratingA && s.ratingA[l.id]) || lineRatingA(l, T), loading: 0, P: 0, Q: 0, I: 0, loss: 0 }; });
  GENS.forEach(g => { if (!res.gen[g.id]) res.gen[g.id] = { P: 0, Q: 0, Pmax: g.Pmax, out: true }; });
  res.status = worstStatus;
  res.lambdaMax = lambdaMax; res.pv = pvInfo;
  res.converged = worstStatus === 'ok';
  res.collapse = worstStatus === 'pv_limit';
  res.unknown = worstStatus === 'numerical';
  res.losses = Object.values(res.flows).reduce((x, f) => x + (f.loss || 0), 0);

  if (worstStatus === 'ok') {
    BUSES.forEach(b => {
      const v = res.V[b.id]; if (!res.live.includes(b.id)) return;
      if (v < VMIN_CRIT) reasons.push(`Напряжение на ${b.name}: ${(v * KV).toFixed(1)} кВ (${(v * 100).toFixed(1)}% Uном) — ниже аварийного предела 90%.`);
      else if (v < VMIN_WARN) warns.push(`Пониженное напряжение на ${b.name}: ${(v * 100).toFixed(1)}% Uном.`);
      if (v > VMAX_CRIT) reasons.push(`Перенапряжение на ${b.name}: ${(v * 100).toFixed(1)}% Uном.`);
      else if (v > VMAX_WARN) warns.push(`Повышенное напряжение на ${b.name}: ${(v * 100).toFixed(1)}%.`);
      score += 400 * Math.max(0, VMIN_WARN - v) + 2000 * Math.max(0, VMIN_CRIT - v) + 400 * Math.max(0, v - VMAX_WARN);
    });
    LINES.forEach(l => {
      const f = res.flows[l.id]; if (f.out || f.dead) return;
      if (f.loading > 1) reasons.push(`Перегрузка ${l.id} (${l.name}): ${(f.loading * 100).toFixed(0)}% допустимого тока (${f.I.toFixed(0)} из ${f.Imax.toFixed(0)} А).`);
      else if (f.loading > 0.9) warns.push(`${l.id} (${l.name}) загружена на ${(f.loading * 100).toFixed(0)}%.`);
      score += 50 * Math.max(0, f.loading - 0.9) + 600 * Math.max(0, f.loading - 1);
    });
    Object.entries(res.gen).forEach(([id, g]) => {
      if (g.out) return;
      if (g.slack && g.P > g.Pmax + 0.5) { reasons.push(`${GENS.find(x => x.id === id).name}: требуется ${g.P.toFixed(0)} МВт при располагаемых ${g.Pmax} МВт.`); score += 10 * (g.P - g.Pmax); }
      if (g.atQlim && !g.slack) warns.push(`${id} вышел на предел реактивной мощности (${g.Q.toFixed(0)} Мвар).`);
    });
    res.minV = Math.min(...res.live.map(id => res.V[id]));
    res.maxLoading = Math.max(0, ...Object.values(res.flows).map(f => f.loading || 0));
  }
  res.totalLoad = res.live.reduce((x, id) => x + loadBus[id].P, 0);
  res.reasons = reasons; res.warns = warns; res.score = score;
  res.critical = reasons.length > 0;
  return res;
}

// P–V кривая для детального просмотра: минимальное напряжение в зависимости от доли нагрузки
function pvCurve(input, lamMax = 2.5) {
  const s = normScn(input);
  const curve = [];
  for (let lam = 0.3; lam <= lamMax + 1e-9; lam += 0.05) {
    const r = runPowerFlow({ ...s, load: s.load * lam });
    if (r.status !== 'ok') { const lm = r.lambdaMax; if (lm) curve.push({ lam: lam * lm, minV: r.pv && r.pv.sol ? Math.min(...r.pv.sol.V) : null, nose: true }); return { curve, lambdaMax: r.lambdaMax ? lam * r.lambdaMax : lam }; }
    curve.push({ lam, minV: r.minV, crit: r.critical });
  }
  return { curve, lambdaMax: null };
}

// ===== Оптимизация: жадный поиск управляющих воздействий =====
function objective(r) {
  if (r.status === 'pv_limit') return 1e6 * (2 - r.lambdaMax) + r.shedMW * 100;
  if (r.status === 'numerical') return 1.5e6 + r.shedMW * 100;
  return r.score * 100 + r.shedMW * 60 + r.losses + Object.values(r.scn.ctrl.bsk).reduce((a, b) => a + b, 0) * 0.15 + ((r.gen.G2 && !r.gen.G2.out ? r.gen.G2.P : 0) + (r.gen.G3 && !r.gen.G3.out ? r.gen.G3.P : 0)) * 0.3;
}
function candidateActions(scn, ctrl) {
  const acts = [];
  GENS.filter(g => !g.slack && !scn.gensOut.includes(g.id)).forEach(g => {
    if ((ctrl.gP[g.id] ?? g.P) < g.Pmax) acts.push({ kind: 'gen', id: g.id, d: 5 });
    if ((ctrl.gP[g.id] ?? g.P) > 0) acts.push({ kind: 'gen', id: g.id, d: -5 });
  });
  GENS.filter(g => !scn.gensOut.includes(g.id)).forEach(g => { if ((ctrl.vset[g.id] ?? g.V) < 1.07) acts.push({ kind: 'vset', id: g.id, d: 0.01 }); });
  [4, 5, 6, 7, 8, 3].forEach(b => { if ((ctrl.bsk[b] || 0) < 20) acts.push({ kind: 'bsk', bus: b, d: 5 }); });
  [2, 3, 4, 5, 6, 7, 8].forEach(b => { if ((ctrl.shed[b] || 0) < 0.5) acts.push({ kind: 'shed', bus: b, d: 0.05 }); });
  return acts;
}
function applyAction(ctrl, a) {
  const c = cloneCtrl(ctrl);
  if (a.kind === 'gen') c.gP[a.id] = Math.max(0, Math.min(GENS.find(g => g.id === a.id).Pmax, (c.gP[a.id] ?? 0) + a.d));
  if (a.kind === 'vset') c.vset[a.id] = +((c.vset[a.id] ?? 1) + a.d).toFixed(3);
  if (a.kind === 'bsk') c.bsk[a.bus] = (c.bsk[a.bus] || 0) + a.d;
  if (a.kind === 'shed') c.shed[a.bus] = +((c.shed[a.bus] || 0) + a.d).toFixed(2);
  return c;
}
function optimize(scnIn, maxSteps = 40) {
  const scn = normScn(scnIn);
  let ctrl = cloneCtrl(scn.ctrl), cur = runPowerFlow({ ...scn, ctrl }), best = objective(cur);
  const steps = [];
  for (let k = 0; k < maxSteps; k++) {
    let pick = null, pickRes = null, pickObj = best, pickCtrl = null;
    for (const a of candidateActions(scn, ctrl)) {
      const c2 = applyAction(ctrl, a), r = runPowerFlow({ ...scn, ctrl: c2 }), o = objective(r);
      if (o < pickObj - 1e-6) { pickObj = o; pick = a; pickRes = r; pickCtrl = c2; }
    }
    if (!pick) break;
    steps.push(pick); ctrl = pickCtrl; cur = pickRes; best = pickObj;
  }
  const base = scn.ctrl, summary = [];
  Object.keys(ctrl.gP).forEach(id => { const d = ctrl.gP[id] - (base.gP[id] ?? 0); if (Math.abs(d) > 0.01) summary.push(`${d > 0 ? 'Загрузить' : 'Разгрузить'} ${id} (${GENS.find(g => g.id === id).name}): ${base.gP[id]} → ${ctrl.gP[id]} МВт`); });
  Object.keys(ctrl.vset).forEach(id => { const d = ctrl.vset[id] - (base.vset[id] ?? 1); if (Math.abs(d) > 1e-4) summary.push(`Уставка напряжения ${id}: ${(base.vset[id] * 100).toFixed(0)}% → ${(ctrl.vset[id] * 100).toFixed(0)}% Uном`); });
  Object.keys(ctrl.bsk).forEach(b => { const d = ctrl.bsk[b] - (base.bsk[b] || 0); if (d > 0) summary.push(`Включить БСК ${d} Мвар на ${BUSES[b - 1].name}`); });
  Object.keys(ctrl.shed).forEach(b => { const d = ctrl.shed[b] - (base.shed[b] || 0); if (d > 0) summary.push(`Ограничить потребление ${BUSES[b - 1].name} на ${(d * 100).toFixed(0)}% (${(scn.load * BUSES[b - 1].share * d).toFixed(1)} МВт)`); });
  return { ctrl, result: cur, steps, summary };
}

if (typeof module !== 'undefined') module.exports = { BUSES, LINES, GENS, COND, runPowerFlow, optimize, pvCurve, buildIsland, calcPQ, buildJ, defaultLoadForTemp, ratingFactor, lineRatingA, defaultCtrl, KV, BASE, ZB };
