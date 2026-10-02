"""Причинные индикаторы: значение на индексе i зависит только от данных 0..i.

Всё векторно на numpy/pandas; окна — только назад (rolling без center).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False, min_periods=n).mean().to_numpy()


def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).mean().to_numpy()


def rolling_std(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).std(ddof=0).to_numpy()


def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """ATR Уайлдера."""
    prev = np.r_[np.nan, close[:-1]]
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev), np.abs(low - prev)]), axis=0)
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False, min_periods=n).mean().to_numpy()


def highest_prev(x: np.ndarray, n: int) -> np.ndarray:
    """Максимум предыдущих n значений (без текущего) — уровень пробоя."""
    return pd.Series(x).rolling(n).max().shift(1).to_numpy()


def lowest_prev(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).min().shift(1).to_numpy()


def efficiency_ratio(close: np.ndarray, n: int) -> np.ndarray:
    """Коэффициент эффективности Кауфмана: |Δ за n| / сумма |Δ|. 1 — чистый тренд, 0 — шум."""
    c = pd.Series(close)
    change = (c - c.shift(n)).abs()
    path = c.diff().abs().rolling(n).sum()
    return (change / path.replace(0, np.nan)).to_numpy()


def pct_rank(x: np.ndarray, n: int) -> np.ndarray:
    """Доля значений в окне последних n (включая текущее), не превышающих текущее."""
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sliding_window_view(x, n)
        out[n - 1:] = (w <= w[:, -1:]).mean(axis=1)
    out[np.isnan(x)] = np.nan
    return out


def zscore(x: np.ndarray, n: int) -> np.ndarray:
    m, s = sma(x, n), rolling_std(x, n)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x - m) / s


def rsi(close: np.ndarray, n: int) -> np.ndarray:
    """RSI Уайлдера."""
    d = np.diff(close, prepend=np.nan)
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = up / dn
    out = (100 - 100 / (1 + rs)).to_numpy(copy=True)
    out[np.isnan(d)] = np.nan
    return out


def rolling_median(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n).median().to_numpy()
