"""Проверка случаев «предел передаваемой мощности»: решение существует при 0.97·λmax
и не находится при полной нагрузке ни одним методом, ни с одного из 20 случайных стартов.
Генераторы рассматриваются без ограничений по Q (это даёт независимой проверке больше
шансов найти решение, т.е. проверка строже)."""
import json, numpy as np, scipy.optimize as so
exec(open('validation/check.py').read().split('stats = {')[0].replace("D = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'validation/cases.json'))","D = json.load(open('validation/cases.json'))"))
rng = np.random.default_rng(1)
orig = so.root
def scaled(c, lam):
    c2 = dict(c); c2['loadBus'] = {k: {'P': v['P'] * lam, 'P0': v['P0']} for k, v in c['loadBus'].items()}
    c2['islands'] = [dict(i, achrMW=i['achrMW'] * lam) for i in c['islands']]; return c2
def try_solve(c, lam, starts):
    global root
    for k in range(starts):
        for m in ('hybr', 'lm'):
            def r(F, x0, method, tol, k=k, m=m):
                x = x0 if k == 0 else x0 * rng.uniform(0.7, 1.1, len(x0)) + rng.normal(0, 0.15, len(x0)) * (np.arange(len(x0)) % 2)
                return orig(F, x, method=m, tol=1e-13)
            root = r
            p = solve_case(c)
            if p is not None:
                bad = [g for g, v in p['gen'].items() if not (GEN[g]['Qmin'] - 0.5 <= v['Q'] <= GEN[g]['Qmax'] + 0.5)]
                return 'q' if bad else True
    return False
ok_below = fail_full = qonly = 0; pv = [c for c in D['cases'] if c['status'] == 'pv_limit']
for c in pv:
    lm = c['lambdaMax']
    if try_solve(scaled(c, lm * 0.97), lm, 3): ok_below += 1
    r = try_solve(scaled(c, 1.0), 1, 20)
    if r is False: fail_full += 1
    elif r == 'q': qonly += 1
    else: print('  РАСХОЖДЕНИЕ:', round(c['T'],1), round(c['load']), c['linesOut'], c['gensOut'])
print(f"Случаев «предел P–V»: {len(pv)}")
print(f"Решение найдено при 0.97·λmax: {ok_below}/{len(pv)}")
print(f"Решение НЕ найдено при 100% нагрузки (2 метода × 20 стартов): {fail_full}/{len(pv)}")
print(f"Решение при 100% существует только с нарушением пределов генераторов по Q: {qonly}/{len(pv)}")
