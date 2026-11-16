"""Модели прогноза нагрузки для эксперимента B — перенос из core_js/day.js (и web_prototype/index.html) на numpy.

  true_load  — генератор «истинной» нагрузки прототипа (trueLoad): промышленность + быт с профилем + коммерция +
               кондиционирование (растёт с жарой, в т. ч. квадратично выше 30 °C) + отопление; плюс шум AR(1);
  L0         — линейная модель прототипа (fitLinear): час суток (24 признака), выходной, T;
  L1         — «честная» линейная модель: признаки нейросети + нелинейные температурные члены (ДОПУЩЕНИЕ состава —
               то, что знал бы аналитик без знания генератора);
  MLP        — нейросеть прототипа (MLPReg + trainLoadModel): 8 признаков, 16 нейронов tanh, Adam, 6 эпох.
mulberry32 — тот же генератор псевдослучайных чисел, что в прототипе (инициализация и перемешивание совпадают).
"""
from __future__ import annotations

import math

import numpy as np

RES_PROFILE = np.array([0.55, 0.5, 0.47, 0.46, 0.47, 0.52, 0.62, 0.72, 0.78, 0.8, 0.8, 0.82, 0.84, 0.84, 0.83, 0.84,
                        0.88, 0.95, 1.0, 1.0, 0.97, 0.9, 0.78, 0.65])
LOAD_MU, LOAD_SD = 165.0, 30.0
M32 = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & M32


def mulberry32(a: int):
    """Побитовая копия mulberry32 из прототипа: возвращает функцию → число в [0, 1)."""
    state = [a & M32]

    def rnd() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & M32
        a = state[0]
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & M32) ^ t
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return rnd


def occ(h):
    return 0.55 + 0.45 * np.maximum(0, np.sin(np.pi * (np.asarray(h) - 8) / 16))


def true_load(h, T, Tlag, weekend):
    """trueLoad прототипа (без шума), векторно."""
    h, T, Tlag, weekend = (np.asarray(x) for x in (h, T, Tlag, weekend))
    ind = 88.0
    res = 48 * RES_PROFILE[h] * np.where(weekend, 1.06, 1.0)
    comm = np.where(weekend, 4.0, 12 * np.maximum(0, np.sin(np.pi * (h - 7) / 12)))
    cool = 3.1 * np.maximum(0, Tlag - 24) * occ(h) + 0.06 * np.maximum(0, Tlag - 30) ** 2 * occ(h)
    heat = 1.3 * np.maximum(0, 8 - Tlag) * (0.7 + 0.3 * RES_PROFILE[h])
    return ind * np.where(weekend, 0.97, 1.0) + res + comm + cool + heat


# ---------------- признаки ----------------
def nn_features(h, T, Tlag, Tmax, weekend) -> np.ndarray:
    """loadFeatures прототипа (8 признаков)."""
    h = np.asarray(h, dtype=float)
    return np.column_stack([np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
                            np.sin(4 * np.pi * h / 24), np.cos(4 * np.pi * h / 24),
                            np.asarray(weekend, dtype=float), (np.asarray(T) - 12) / 15,
                            (np.asarray(Tlag) - 12) / 15, (np.asarray(Tmax) - 15) / 15])


def _hour_dummies(h):
    D = np.zeros((len(h), 24))
    D[np.arange(len(h)), np.asarray(h, dtype=int)] = 1
    return D


def l0_features(h, T, Tlag, Tmax, weekend) -> np.ndarray:
    return np.column_stack([_hour_dummies(h), np.asarray(weekend, dtype=float), np.asarray(T, dtype=float)])


def l1_features(h, T, Tlag, Tmax, weekend) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    Tlag = np.asarray(Tlag, dtype=float)
    cdd = np.maximum(0, Tlag - 24)
    return np.column_stack([_hour_dummies(h), np.asarray(weekend, dtype=float), np.asarray(T, dtype=float), Tlag,
                            np.asarray(Tmax, dtype=float), cdd, cdd * np.sin(2 * np.pi * h / 24),
                            cdd * np.cos(2 * np.pi * h / 24), np.maximum(0, Tlag - 30) ** 2, np.maximum(0, 8 - Tlag)])


class Linear:
    """МНК с малой ридж-добавкой 1e-3 на диагональ (как fitLinear прототипа)."""

    def __init__(self, feats):
        self.feats = feats

    def fit(self, X: dict, y):
        A = self.feats(**X)
        self.w = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ np.asarray(y))
        return self

    def predict(self, X: dict):
        return self.feats(**X) @ self.w


class MLP:
    """MLPReg + trainLoadModel прототипа: tanh-слой, Adam, мини-пакеты по 32, lr = 0,01·0,6^эпоха."""

    def __init__(self, n_hidden: int = 16, epochs: int = 6, seed: int = 5, shuffle_seed: int = 9):
        self.nH, self.epochs, self.seed, self.shuffle_seed = n_hidden, epochs, seed, shuffle_seed

    def fit(self, X: dict, y):
        x = nn_features(**X)
        yt = (np.asarray(y, dtype=float) - LOAD_MU) / LOAD_SD
        n, nIn = x.shape
        rnd = mulberry32(self.seed)
        rn = lambda: rnd() * 2 - 1  # noqa: E731
        W1 = np.array([[rn() * math.sqrt(2 / nIn) for _ in range(nIn)] for _ in range(self.nH)])
        W2 = np.array([rn() * math.sqrt(1 / self.nH) for _ in range(self.nH)])
        b1, b2 = np.zeros(self.nH), 0.0
        params = {"W1": W1, "b1": b1, "W2": W2}
        m = {k: np.zeros_like(v) for k, v in params.items()}
        v = {k: np.zeros_like(p) for k, p in params.items()}
        mb2 = vb2 = 0.0
        B1, B2 = 0.9, 0.999
        t = 0
        order = np.arange(n)
        srnd = mulberry32(self.shuffle_seed)
        for e in range(self.epochs):
            for i in range(n - 1, 0, -1):
                j = int(math.floor(srnd() * (i + 1)))
                order[i], order[j] = order[j], order[i]
            lr = 0.01 * 0.6 ** e
            for k in range(0, n, 32):
                idx = order[k:k + 32]
                xb, yb = x[idx], yt[idx]
                hb = np.tanh(xb @ params["W1"].T + params["b1"])
                yh = hb @ params["W2"] + b2
                dy = yh - yb
                N = len(idx)
                dh = dy[:, None] * params["W2"][None, :] * (1 - hb * hb)
                grads = {"W1": dh.T @ xb / N, "b1": dh.sum(0) / N, "W2": hb.T @ dy / N}
                gb2 = dy.sum() / N
                t += 1
                for key, g in grads.items():
                    m[key] = B1 * m[key] + (1 - B1) * g
                    v[key] = B2 * v[key] + (1 - B2) * g * g
                    params[key] = params[key] - lr * (m[key] / (1 - B1 ** t)) / (np.sqrt(v[key] / (1 - B2 ** t)) + 1e-8)
                mb2 = B1 * mb2 + (1 - B1) * gb2
                vb2 = B2 * vb2 + (1 - B2) * gb2 * gb2
                b2 = b2 - lr * (mb2 / (1 - B1 ** t)) / (math.sqrt(vb2 / (1 - B2 ** t)) + 1e-8)
        self.params, self.b2 = params, b2
        return self

    def predict(self, X: dict):
        x = nn_features(**X)
        return (np.tanh(x @ self.params["W1"].T + self.params["b1"]) @ self.params["W2"] + self.b2) * LOAD_SD + LOAD_MU
