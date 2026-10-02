"""Четыре кандидата разного типа + фильтр рыночного режима.

Фильтр режима — коэффициент эффективности Кауфмана (ER) за ER_N свечей:
трендовые стратегии торгуют при ER ≥ er_min (направленный рынок),
возврат к среднему — при ER ≤ er_max (рынок «пилит»). er_min=0 / er_max=1 —
фильтр выключен.

Все стратегии выдают только сигнал и стоп; размер позиции и плечо считает
риск-модуль. Все индикаторы причинные (см. indicators.py).
"""
from __future__ import annotations

import numpy as np

from bot.data.bars import Bars
from bot.strategy import indicators as ind
from bot.strategy.base import LONG, SHORT, Enter, Exit, MoveStop, PositionView, Strategy

ATR_N = 14
ER_N = 48


class _Base(Strategy):
    family = "base"
    grid: dict[str, list] = {}

    def __init__(self, timeframe: str = "1h", **params):
        unknown = set(params) - set(self.grid)
        if unknown:
            raise ValueError(f"Неизвестные параметры {unknown}")
        self.timeframe = timeframe
        self.p = {k: params.get(k, v[0]) for k, v in self.grid.items()}
        self.name = f"{self.family}"

    def params(self) -> dict:
        return dict(self.p)

    def _common(self, bars: Bars) -> None:
        self.h, self.l, self.c = bars.high, bars.low, bars.close
        self.atr = ind.atr(bars.high, bars.low, bars.close, ATR_N)
        self.er = ind.efficiency_ratio(bars.close, ER_N)

    def _chandelier(self, i: int, pos: PositionView, k: float):
        """Трейлинг-стоп: экстремум с момента входа ∓ k×ATR. Двигается только в сторону прибыли."""
        a = max(i - pos.bars_held + 1, 0)
        if pos.side == LONG:
            s = float(np.max(self.h[a:i + 1])) - k * self.atr[i]
            return MoveStop(s) if s > pos.stop else None
        s = float(np.min(self.l[a:i + 1])) + k * self.atr[i]
        return MoveStop(s) if s < pos.stop else None


class DonchianTrend(_Base):
    """Пробой канала Дончиана по закрытию; начальный стоп k_stop×ATR, трейлинг k_trail×ATR,
    выход при пробое противоположной границы."""
    family = "donchian"
    grid = {"n": [20, 40, 60, 100], "k_stop": [1.5, 2.5, 3.5], "k_trail": [2.0, 3.0, 4.0],
            "er_min": [0.0, 0.2, 0.3]}

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        n = self.p["n"]
        self.hh, self.ll = ind.highest_prev(bars.high, n), ind.lowest_prev(bars.low, n)
        self.warmup_bars = max(n, ER_N, ATR_N) + 1

    def on_bar(self, i: int, pos: PositionView | None):
        c, a = self.c[i], self.atr[i]
        if np.isnan(a) or np.isnan(self.hh[i]) or np.isnan(self.er[i]):
            return None
        if pos is None:
            if self.er[i] < self.p["er_min"]:
                return None
            if c > self.hh[i]:
                return Enter(LONG, c - self.p["k_stop"] * a)
            if c < self.ll[i]:
                return Enter(SHORT, c + self.p["k_stop"] * a)
            return None
        if (pos.side == LONG and c < self.ll[i]) or (pos.side == SHORT and c > self.hh[i]):
            return Exit("opposite_breakout")
        return self._chandelier(i, pos, self.p["k_trail"])


class Momentum(_Base):
    """Импульс: доходность за L свечей в единицах волатильности. Вход при пересечении
    порога z_in, выход при смене знака импульса или по стопу k_stop×ATR."""
    family = "momentum"
    grid = {"L": [12, 24, 48, 96], "z_in": [1.0, 1.5, 2.0], "k_stop": [2.0, 3.0], "er_min": [0.0, 0.25]}

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        L = self.p["L"]
        logc = np.log(bars.close)
        r1 = np.r_[np.nan, np.diff(logc)]
        vol = ind.rolling_std(r1, 100)
        rL = logc - np.r_[np.full(L, np.nan), logc[:-L]]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.mom = rL / (vol * np.sqrt(L))
        self.warmup_bars = max(L + 100, ER_N, ATR_N) + 1

    def on_bar(self, i: int, pos: PositionView | None):
        m, mp, a, c = self.mom[i], self.mom[i - 1] if i > 0 else np.nan, self.atr[i], self.c[i]
        if np.isnan(m) or np.isnan(mp) or np.isnan(a) or np.isnan(self.er[i]):
            return None
        z = self.p["z_in"]
        if pos is None:
            if self.er[i] < self.p["er_min"]:
                return None
            if m >= z > mp:
                return Enter(LONG, c - self.p["k_stop"] * a)
            if m <= -z < mp:
                return Enter(SHORT, c + self.p["k_stop"] * a)
            return None
        if (pos.side == LONG and m <= 0) or (pos.side == SHORT and m >= 0):
            return Exit("momentum_flip")
        return None


class MeanReversion(_Base):
    """Возврат к среднему: |z| цены относительно EMA(n) ≥ z_in → вход против движения,
    тейк-профит (лимитный, мейкер) на EMA в момент входа, стоп k_stop×ATR,
    выход по времени через n свечей."""
    family = "meanrev"
    grid = {"n": [20, 50, 100], "z_in": [2.0, 2.5, 3.0], "k_stop": [1.5, 2.5, 3.5], "er_max": [1.0, 0.3, 0.2]}

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        n = self.p["n"]
        self.mid = ind.ema(bars.close, n)
        sd = ind.rolling_std(bars.close, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.z = (bars.close - self.mid) / sd
        self.warmup_bars = max(n, ER_N, ATR_N) + 1

    def on_bar(self, i: int, pos: PositionView | None):
        z, a, c, mid = self.z[i], self.atr[i], self.c[i], self.mid[i]
        if np.isnan(z) or np.isnan(a) or np.isnan(mid) or np.isnan(self.er[i]):
            return None
        if pos is None:
            if self.er[i] > self.p["er_max"]:
                return None
            if z <= -self.p["z_in"]:
                return Enter(LONG, c - self.p["k_stop"] * a, take_profit=mid)
            if z >= self.p["z_in"]:
                return Enter(SHORT, c + self.p["k_stop"] * a, take_profit=mid)
            return None
        if pos.bars_held >= self.p["n"]:
            return Exit("time")
        return None


class SqueezeBreakout(_Base):
    """Пробой волатильности: после сжатия (ширина полос Боллинджера в нижних p
    за 250 свечей в одной из 5 последних) — пробой n-свечного канала по закрытию.
    Стоп k_stop×ATR, трейлинг k_trail×ATR."""
    family = "squeeze"
    grid = {"n": [20, 40], "p": [0.15, 0.3], "k_stop": [1.5, 2.5], "k_trail": [2.0, 3.0, 4.0]}
    RANK_N, RECENT = 250, 5

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        n = self.p["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            bw = 4 * ind.rolling_std(bars.close, n) / ind.sma(bars.close, n)
        rank = ind.pct_rank(bw, self.RANK_N)
        # минимум ранга за RECENT свечей ДО текущей
        import pandas as pd
        self.recent_min_rank = pd.Series(rank).rolling(self.RECENT).min().shift(1).to_numpy()
        self.hh, self.ll = ind.highest_prev(bars.high, n), ind.lowest_prev(bars.low, n)
        self.warmup_bars = self.RANK_N + n + self.RECENT + 1

    def on_bar(self, i: int, pos: PositionView | None):
        c, a = self.c[i], self.atr[i]
        if np.isnan(a) or np.isnan(self.hh[i]) or np.isnan(self.recent_min_rank[i]):
            return None
        if pos is None:
            if self.recent_min_rank[i] > self.p["p"]:
                return None
            if c > self.hh[i]:
                return Enter(LONG, c - self.p["k_stop"] * a)
            if c < self.ll[i]:
                return Enter(SHORT, c + self.p["k_stop"] * a)
            return None
        return self._chandelier(i, pos, self.p["k_trail"])


# ---------------------------------------------------------------- раунд 2
# Добавлены после раунда 1: частые трендовые конфигурации теряли до издержек
# (пробои на 15–30m HYPE систематически не продолжаются — хуже случайных входов),
# медленные не набирают 100 сделок за полгода.


class BreakoutFade(_Base):
    """Против пробоя: закрытие за n-свечным каналом → вход в обратную сторону,
    тейк tp_k×ATR (лимитный), стоп k_stop×ATR, выход по времени через hold свечей.
    Фильтр режима: только при ER ≤ er_max."""
    family = "fade"
    grid = {"n": [20, 40, 60], "k_stop": [1.0, 1.5, 2.5], "tp_k": [1.0, 1.5, 2.5], "hold": [8, 24],
            "er_max": [1.0, 0.3]}

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        n = self.p["n"]
        self.hh, self.ll = ind.highest_prev(bars.high, n), ind.lowest_prev(bars.low, n)
        self.warmup_bars = max(n, ER_N, ATR_N) + 1

    def on_bar(self, i: int, pos: PositionView | None):
        c, a = self.c[i], self.atr[i]
        if np.isnan(a) or np.isnan(self.hh[i]) or np.isnan(self.er[i]):
            return None
        if pos is None:
            if self.er[i] > self.p["er_max"]:
                return None
            if c > self.hh[i]:
                return Enter(SHORT, c + self.p["k_stop"] * a, take_profit=c - self.p["tp_k"] * a)
            if c < self.ll[i]:
                return Enter(LONG, c - self.p["k_stop"] * a, take_profit=c + self.p["tp_k"] * a)
            return None
        return Exit("time") if pos.bars_held >= self.p["hold"] else None


class TrendPullback(_Base):
    """Откат по тренду: тренд — цена над растущей EMA(trend_n) (лонг) / под падающей (шорт);
    вход, когда RSI(2) ≤ rsi_in (лонг) / ≥ 100 − rsi_in (шорт); выход при RSI(2) ≥ exit_rsi
    (лонг) / ≤ 100 − exit_rsi (шорт), стоп k_stop×ATR, не дольше 20 свечей."""
    family = "pullback"
    grid = {"trend_n": [100, 200], "rsi_in": [5.0, 10.0, 20.0], "k_stop": [1.5, 2.5, 3.5],
            "exit_rsi": [50.0, 70.0]}
    MAX_HOLD, SLOPE = 20, 10

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        self.ema = ind.ema(bars.close, self.p["trend_n"])
        self.rsi = ind.rsi(bars.close, 2)
        self.warmup_bars = self.p["trend_n"] + self.SLOPE + 2

    def on_bar(self, i: int, pos: PositionView | None):
        if i < self.SLOPE:
            return None
        c, a, e, e0, r = self.c[i], self.atr[i], self.ema[i], self.ema[i - self.SLOPE], self.rsi[i]
        if np.isnan(a) or np.isnan(e0) or np.isnan(r):
            return None
        if pos is None:
            if c > e > e0 and r <= self.p["rsi_in"]:
                return Enter(LONG, c - self.p["k_stop"] * a)
            if c < e < e0 and r >= 100 - self.p["rsi_in"]:
                return Enter(SHORT, c + self.p["k_stop"] * a)
            return None
        if (pos.side == LONG and r >= self.p["exit_rsi"]) or (pos.side == SHORT and r <= 100 - self.p["exit_rsi"]):
            return Exit("rsi")
        return Exit("time") if pos.bars_held >= self.MAX_HOLD else None


class ShockFade(_Base):
    """Отскок после шока: доходность свечи ≤ −k_sig×σ на объёме ≥ v_mult×медианы (каскад
    ликвидаций) → вход против движения; тейк — возврат r_tp доли тела свечи (лимитный),
    стоп k_stop×ATR, выход по времени через 12 свечей."""
    family = "shock"
    grid = {"k_sig": [2.5, 3.5, 4.5], "v_mult": [1.5, 3.0], "k_stop": [1.5, 2.5, 3.5], "r_tp": [0.5, 1.0]}
    WIN, MAX_HOLD = 200, 12

    def prepare(self, bars: Bars) -> None:
        self._common(bars)
        logc = np.log(bars.close)
        self.ret = np.r_[np.nan, np.diff(logc)]
        # σ и медиана объёма — по ПРЕДЫДУЩИМ свечам, чтобы шок не размывал собственный порог
        import pandas as pd
        self.sig = pd.Series(self.ret).rolling(self.WIN).std(ddof=0).shift(1).to_numpy()
        self.vmed = pd.Series(bars.volume).rolling(self.WIN).median().shift(1).to_numpy()
        self.vol, self.o = bars.volume, bars.open
        self.warmup_bars = self.WIN + 2

    def on_bar(self, i: int, pos: PositionView | None):
        a, s, vm = self.atr[i], self.sig[i], self.vmed[i]
        if np.isnan(a) or np.isnan(s) or np.isnan(vm) or s <= 0:
            return None
        if pos is None:
            if self.vol[i] < self.p["v_mult"] * vm:
                return None
            c, body = self.c[i], self.o[i] - self.c[i]
            if self.ret[i] <= -self.p["k_sig"] * s and body > 0:
                return Enter(LONG, c - self.p["k_stop"] * a, take_profit=c + self.p["r_tp"] * body)
            if self.ret[i] >= self.p["k_sig"] * s and body < 0:
                return Enter(SHORT, c + self.p["k_stop"] * a, take_profit=c + self.p["r_tp"] * body)
            return None
        return Exit("time") if pos.bars_held >= self.MAX_HOLD else None


ROUND1 = (DonchianTrend, Momentum, MeanReversion, SqueezeBreakout)
ROUND2 = (BreakoutFade, TrendPullback, ShockFade)
FAMILIES: dict[str, type[_Base]] = {cls.family: cls for cls in ROUND1 + ROUND2}
TIMEFRAMES = ["15m", "30m", "1h", "4h"]
# 4h не даёт 100 сделок за полгода даже у самых частых кандидатов раунда 1 — во 2-м раунде не берём.
TIMEFRAMES_R2 = ["15m", "30m", "1h"]


def grid_configs(family: str) -> list[dict]:
    import itertools
    g = FAMILIES[family].grid
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def make(family: str, timeframe: str, params: dict) -> _Base:
    return FAMILIES[family](timeframe=timeframe, **params)
