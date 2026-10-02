"""Стратегии только для тестов движка (не кандидаты исследования)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.data.bars import Bars
from bot.strategy.base import LONG, SHORT, Enter, Exit, PositionView, Strategy


class Scripted(Strategy):
    """Решения заданы заранее по номеру свечи."""
    name = "scripted"

    def __init__(self, script: dict, timeframe: str = "5m"):
        self.script, self.timeframe, self.warmup_bars = script, timeframe, 0

    def prepare(self, bars: Bars) -> None:
        self.bars = bars

    def on_bar(self, i, pos):
        d = self.script.get(i)
        return d(self.bars, i, pos) if callable(d) else d


class SmaCross(Strategy):
    """Пересечение SMA, стоп = k×ATR. Индикаторы — только прошлые свечи."""
    name = "sma_cross"

    def __init__(self, fast=10, slow=30, atr_n=14, atr_k=2.0, timeframe="15m"):
        self.fast, self.slow, self.atr_n, self.atr_k = fast, slow, atr_n, atr_k
        self.timeframe, self.warmup_bars = timeframe, slow + atr_n

    def params(self):
        return {"fast": self.fast, "slow": self.slow}

    def prepare(self, bars: Bars) -> None:
        c = pd.Series(bars.close)
        self.f = c.rolling(self.fast).mean().to_numpy()
        self.s = c.rolling(self.slow).mean().to_numpy()
        prev_c = c.shift(1)
        tr = pd.concat([pd.Series(bars.high) - pd.Series(bars.low),
                        (pd.Series(bars.high) - prev_c).abs(), (pd.Series(bars.low) - prev_c).abs()], axis=1).max(axis=1)
        self.atr = tr.rolling(self.atr_n).mean().to_numpy()
        self.close = bars.close

    def on_bar(self, i: int, pos: PositionView | None):
        if i < 1 or np.isnan(self.s[i - 1]) or np.isnan(self.atr[i]):
            return None
        up = self.f[i] > self.s[i] and self.f[i - 1] <= self.s[i - 1]
        dn = self.f[i] < self.s[i] and self.f[i - 1] >= self.s[i - 1]
        if pos is None:
            if up:
                return Enter(LONG, self.close[i] - self.atr_k * self.atr[i])
            if dn:
                return Enter(SHORT, self.close[i] + self.atr_k * self.atr[i])
        elif (pos.side == LONG and dn) or (pos.side == SHORT and up):
            return Exit("cross")
        return None


class Cheater(SmaCross):
    """Подглядывает в закрытие СЛЕДУЮЩЕЙ свечи — проверки причинности обязаны это поймать."""
    name = "cheater"

    def on_bar(self, i, pos):
        if i + 1 >= len(self.close) or np.isnan(self.atr[i]):
            return None
        if pos is None:
            if self.close[i + 1] > self.close[i]:
                return Enter(LONG, self.close[i] - self.atr_k * self.atr[i])
            return Enter(SHORT, self.close[i] + self.atr_k * self.atr[i])
        return Exit("peek")


class NormalizedLeak(SmaCross):
    """Тонкая утечка: нормировка по среднему ВСЕЙ выборки (будущее в прошлом)."""
    name = "normalized_leak"

    def prepare(self, bars):
        super().prepare(bars)
        self.mean_all = float(np.mean(bars.close))

    def on_bar(self, i, pos):
        if np.isnan(self.atr[i]):
            return None
        if pos is None and self.close[i] < self.mean_all:
            return Enter(LONG, self.close[i] - self.atr_k * self.atr[i])
        if pos is not None and self.close[i] > self.mean_all:
            return Exit("mean")
        return None
