"""Независимая проверка расчёта режима MANGYSTAU DIGITAL TWIN.

Другая формулировка (прямоугольные координаты, комплексная арифметика numpy)
и другой численный метод (гибридный метод Пауэлла, scipy.optimize.root),
старт с плоских напряжений — без использования решения JS.
"""
import json, sys
import numpy as np
from scipy.optimize import root

D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'validation/cases.json'))
BASE, KV = 100.0, 110.0
ZB = KV * KV / BASE
BUS = {b['id']: b for b in D['BUSES']}
GEN = {g['id']: g for g in D['GENS']}

def solve_case(c):
    T = c['T']; tan = 0.42 + 0.005 * max(0, T - 25)
    rT = 1 + 0.00403 * (T + 15 - 20)
    lines = [l for l in D['LINES'] if l['id'] not in c['linesOut']]
    out = {'V': {}, 'ang': {}, 'flows': {}, 'gen': {}}
    for isl in c['islands']:
        ids = isl['ids']; n = len(ids); ix = {b: i for i, b in enumerate(ids)}
        Y = np.zeros((n, n), complex); lp = {}
        for l in lines:
            if l['f'] not in ix: continue
            cd = D['COND'][l['c']]
            z = complex(cd['r'] * rT * l['km'], cd['x'] * l['km']) / ZB
            y = 1 / z; bsh = cd['b'] * l['km'] * ZB / 2
            i, j = ix[l['f']], ix[l['t']]
            Y[i, i] += y + 1j * bsh; Y[j, j] += y + 1j * bsh; Y[i, j] -= y; Y[j, i] -= y
            lp[l['id']] = (i, j, y, bsh)
        for b in ids:
            Y[ix[b], ix[b]] += 1j * c['ctrl']['bsk'].get(str(b), 0) / BASE
        island_load = sum(c['loadBus'][str(b)]['P'] for b in ids)
        frac = isl['achrMW'] / island_load if island_load > 0 else 0
        S_load = np.array([c['loadBus'][str(b)]['P'] * (1 - frac) * (1 + 1j * tan) / BASE for b in ids])
        Pg = np.zeros(n); Qfix = np.zeros(n); kind = ['PQ'] * n; Vset = np.ones(n)
        gens = [g for g in D['GENS'] if g['id'] not in c['gensOut'] and g['bus'] in ix]
        for g in gens:
            k = ix[g['bus']]; js = c['gen'][g['id']]
            vs = c['ctrl']['vset'].get(g['id'], g['V'])
            if g['id'] == isl['slack']:
                kind[k] = 'SL'; Vset[k] = vs; continue
            Pg[k] += min(g['Pmax'], c['ctrl']['gP'].get(g['id'], g['P'])) / BASE
            if js['atQlim']: Qfix[k] += js['Q'] / BASE        # генератор на пределе по Q → PQ
            elif kind[k] != 'SL': kind[k] = 'PV'; Vset[k] = vs
        sl = kind.index('SL'); others = [i for i in range(n) if i != sl]

        def volt(x):
            V = np.zeros(n, complex); V[sl] = Vset[sl]
            for m, i in enumerate(others): V[i] = complex(x[2 * m], x[2 * m + 1])
            return V
        def F(x):
            V = volt(x); S = V * np.conj(Y @ V); r = []
            for i in others:
                P_sp = Pg[i] - S_load[i].real
                if kind[i] == 'PQ': r += [S[i].real - P_sp, S[i].imag - (Qfix[i] - S_load[i].imag)]
                else: r += [S[i].real - P_sp, abs(V[i]) ** 2 - Vset[i] ** 2]
            return r
        x0 = np.ravel([[Vset[i] if kind[i] == 'PV' else 1.0, 0.0] for i in others]) if others else np.array([])
        if others:
            sol = root(F, x0, method='hybr', tol=1e-13)
            if not sol.success or max(abs(np.array(F(sol.x)))) > 1e-8:
                sol = root(F, x0, method='lm', tol=1e-14)   # запасной метод Левенберга–Марквардта
                if max(abs(np.array(F(sol.x)))) > 1e-8: return None
            V = volt(sol.x)
        else: V = np.array([complex(Vset[sl], 0)])
        S = V * np.conj(Y @ V)
        for b in ids:
            out['V'][b] = abs(V[ix[b]]); out['ang'][b] = np.degrees(np.angle(V[ix[b]]))
        for lid, (i, j, y, bsh) in lp.items():
            Iij = (V[i] - V[j]) * y + V[i] * 1j * bsh
            Sij = V[i] * np.conj(Iij) * BASE
            Iji = (V[j] - V[i]) * y + V[j] * 1j * bsh
            Sji = V[j] * np.conj(Iji) * BASE
            Ia = abs(Iij) * BASE / (np.sqrt(3) * KV) * 1000; Ib = abs(Iji) * BASE / (np.sqrt(3) * KV) * 1000
            out['flows'][lid] = {'P': Sij.real, 'Q': Sij.imag, 'I': max(Ia, Ib), 'loss': (Sij + Sji).real}
        for g in gens:
            k = ix[g['bus']]
            Qg = (S[k].imag + S_load[k].imag) * BASE
            Pg_ = (S[k].real + S_load[k].real) * BASE if g['id'] == isl['slack'] else min(g['Pmax'], c['ctrl']['gP'].get(g['id'], g['P']))
            out['gen'][g['id']] = {'P': Pg_, 'Q': Qg}
    return out

stats = {'V': 0, 'ang': 0, 'P': 0, 'Q': 0, 'I': 0, 'genP': 0, 'genQ': 0, 'balance': 0}
n_ok = n_fail = qviol = 0; worst = None
for c in D['cases']:
    if c['status'] != 'ok': continue
    p = solve_case(c)
    if p is None: n_fail += 1; continue
    n_ok += 1
    for b, v in p['V'].items():
        d = abs(v - c['V'][str(b)]); stats['V'] = max(stats['V'], d)
        if d == stats['V'] and d > 0: worst = (c['T'], c['load'], c['linesOut'], c['gensOut'])
        stats['ang'] = max(stats['ang'], abs(p['ang'][b] - c['ang'][str(b)]))
    for lid, f in p['flows'].items():
        j = c['flows'][lid]
        stats['P'] = max(stats['P'], abs(f['P'] - j['P'])); stats['Q'] = max(stats['Q'], abs(f['Q'] - j['Q'])); stats['I'] = max(stats['I'], abs(f['I'] - j['I']))
    for gid, g in p['gen'].items():
        j = c['gen'][gid]
        stats['genP'] = max(stats['genP'], abs(g['P'] - j['P'])); stats['genQ'] = max(stats['genQ'], abs(g['Q'] - j['Q']))
        G = GEN[gid]
        if not j['atQlim'] and not j.get('slack') and (g['Q'] > G['Qmax'] + 1e-3 or g['Q'] < G['Qmin'] - 1e-3): qviol += 1
    # баланс мощности в JS-результате: генерация = нагрузка + потери
    gen = sum(v['P'] for v in c['gen'].values() if not v.get('out'))
    live = [b for isl in c['islands'] for b in isl['ids']]
    load = sum(c['loadBus'][str(b)]['P'] for b in live) - c['achr']
    loss = sum(f['loss'] for f in c['flows'].values())
    stats['balance'] = max(stats['balance'], abs(gen - load - loss))

print(f"Сценариев сравнено: {n_ok}, независимый решатель не сошёлся: {n_fail}")
print(f"Макс. расхождение |U|: {stats['V']:.2e} о.е. ({stats['V']*KV*1000:.4f} В)")
print(f"Макс. расхождение угла: {stats['ang']:.2e}°")
print(f"Макс. расхождение потоков: P {stats['P']:.2e} МВт, Q {stats['Q']:.2e} Мвар, I {stats['I']:.2e} А")
print(f"Макс. расхождение генерации: P {stats['genP']:.2e} МВт, Q {stats['genQ']:.2e} Мвар")
print(f"Небаланс мощности в JS (генерация − нагрузка − потери): {stats['balance']:.2e} МВт")
print(f"PV-генераторы вне пределов Q без перевода в PQ: {qviol}")
