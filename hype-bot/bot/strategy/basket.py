"""Стратегии корзины монет (протокол: docs/BASKET_PROTOCOL.md).

Каждая стратегия видит закрытые свечи всех монет и вспомогательные ряды только в том
виде, в каком они были известны на закрытии свечи (bot/data/panel.py). Индикаторы
считаются векторно в prepare и причинны: значение на свече k зависит только от
данных, известных к её закрытию (проверяется тестом с подменой будущего).

Решения возвращаются списком (строка монеты, решение): сначала выходы и подтяжки
стопов, затем входы — от сильнейшего сигнала к слабейшему.
"""
from __future__ import annotations

import itertools
import warnings

import numpy as np

from bot.backtest.portfolio import PortfolioStrategy
from bot.data.bars import TF_MINUTES
from bot.data.panel import PanelBars
from bot.strategy import indicators as ind
from bot.strategy.base import LONG, SHORT, Enter, Exit, MoveStop, PositionView

ATR_N = 14
MAX_STOP_FRAC = 0.095     # шире — минимальный ордер при 10 USDT и риске 5 % не помещается
BTC_EMA_DAYS = 50
FG_HI, FG_LO = 75.0, 25.0
VOL_DAYS = 30
DAY_MS = 86_400_000


def bars_per_day(tf: str) -> int:
    return 1440 // TF_MINUTES[tf]


def window_sum(known_ts: np.ndarray, values: np.ndarray, t: np.ndarray, window_ms: int) -> tuple[np.ndarray, np.ndarray]:
    """Сумма и число значений с known_ts в (t − window, t] — по накопленным суммам."""
    v = np.nan_to_num(values, nan=0.0)
    cs = np.r_[0.0, np.cumsum(v)]
    cs2 = np.r_[0.0, np.cumsum(v * v)]
    hi = np.searchsorted(known_ts, t, "right")
    lo = np.searchsorted(known_ts, t - window_ms, "right")
    return cs[hi] - cs[lo], (hi - lo).astype(float), cs2[hi] - cs2[lo]


class BasketStrategy(PortfolioStrategy):
    family = "basket"
    grid: dict[str, list] = {}
    timeframes: tuple[str, ...] = ("4h", "1d")

    def __init__(self, timeframe: str = "1d", **params):
        unknown = set(params) - set(self.grid)
        if unknown:
            raise ValueError(f"Неизвестные параметры {unknown}")
        self.timeframe = timeframe
        self.p = {k: params.get(k, v[0]) for k, v in self.grid.items()}
        self.name = self.family

    def params(self) -> dict:
        return dict(self.p)

    # ---------------------------------------------------------- общие ряды
    def _common(self, bars: PanelBars) -> None:
        n = bars.n_trade
        self.n = n
        self.bpd = bars_per_day(self.timeframe)
        self.c, self.h, self.l = bars.close[:n], bars.high[:n], bars.low[:n]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.atr = np.vstack([ind.atr(self.h[r], self.l[r], self.c[r], ATR_N) for r in range(n)])
        valid = ~np.isnan(self.c)
        first = np.where(valid.any(axis=1), valid.argmax(axis=1), len(bars))
        self.age = np.arange(len(bars))[None, :] - first[:, None]   # свечей с первой свечи монеты
        regime = self.p.get("regime", "none")
        self.long_ok = np.ones(len(bars), bool)
        self.short_ok = np.ones(len(bars), bool)
        if regime == "btc":
            b = bars.symbols.index("BTCUSDT")
            bc = bars.close[b]
            e = ind.ema(bc, BTC_EMA_DAYS * self.bpd)
            with np.errstate(invalid="ignore"):
                self.long_ok, self.short_ok = bc > e, bc < e      # NaN → оба запрещены
        elif regime == "fg":
            fg = bars.fear_greed()
            with np.errstate(invalid="ignore"):
                self.long_ok, self.short_ok = fg < FG_HI, fg > FG_LO

    def _vol_adj_return(self, L: int) -> np.ndarray:
        """Доходность за L свечей в единицах волатильности (σ свечи за VOL_DAYS дней)."""
        logc = np.log(self.c)
        r1 = np.diff(logc, axis=1, prepend=np.nan)
        win = VOL_DAYS * self.bpd
        vol = np.vstack([ind.rolling_std(r1[r], win) for r in range(self.n)])
        rL = logc - np.concatenate([np.full((self.n, L), np.nan), logc[:, :-L]], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return rL / (vol * np.sqrt(L))

    def _enter(self, r: int, k: int, side: int, score: float, tag: str = ""):
        """Вход со стопом k_stop × ATR; None, если стоп шире MAX_STOP_FRAC или режим запрещает."""
        if not (self.long_ok[k] if side == LONG else self.short_ok[k]):
            return None
        c, a = self.c[r, k], self.atr[r, k]
        if np.isnan(c) or np.isnan(a) or a <= 0 or self.age[r, k] < self.warmup_bars:
            return None
        dist = self.p["k_stop"] * a
        if dist / c > MAX_STOP_FRAC:
            return None
        return (score, (r, Enter(side, c - side * dist, tag=tag or self.family)))

    @staticmethod
    def _ordered(exits: list, entries: list) -> list:
        entries.sort(key=lambda x: -x[0])
        return exits + [e[1] for e in entries]

    def _chandelier(self, r: int, k: int, pos: PositionView, k_trail: float):
        a0 = max(k - pos.bars_held + 1, 0)
        if pos.side == LONG:
            s = float(np.nanmax(self.h[r, a0:k + 1])) - k_trail * self.atr[r, k]
            return MoveStop(s) if s > pos.stop else None
        s = float(np.nanmin(self.l[r, a0:k + 1])) + k_trail * self.atr[r, k]
        return MoveStop(s) if s < pos.stop else None


# ------------------------------------------------------------------ F0
class TsMom(BasketStrategy):
    """Импульс по монете: z = доходность за L дней / (σ·√L) пересекает ±z_in → вход по
    направлению; выход при смене знака z или по стопу."""
    family = "tsmom"
    grid = {"L": [3, 7, 21], "z_in": [1.0, 2.0], "k_stop": [2.0, 3.0], "sides": ["both", "long"],
            "regime": ["none", "btc", "fg"]}

    def prepare(self, bars: PanelBars) -> None:
        self._common(bars)
        L = self.p["L"] * self.bpd
        self.mom = self._vol_adj_return(L)
        self.warmup_bars = L + VOL_DAYS * self.bpd + 1

    def on_bar(self, k: int, positions: dict):
        if k < 1:
            return []
        z, exits, entries = self.p["z_in"], [], []
        for r in range(self.n):
            m, mp = self.mom[r, k], self.mom[r, k - 1]
            pos = positions.get(r)
            if pos is not None:
                if (pos.side == LONG and m <= 0) or (pos.side == SHORT and m >= 0):
                    exits.append((r, Exit("momentum_flip")))
                continue
            if np.isnan(m) or np.isnan(mp):
                continue
            e = None
            if m >= z > mp:
                e = self._enter(r, k, LONG, m)
            elif m <= -z < mp and self.p["sides"] == "both":
                e = self._enter(r, k, SHORT, -m)
            if e:
                entries.append(e)
        return self._ordered(exits, entries)


# ------------------------------------------------------------------ F1
class Breakout(BasketStrategy):
    """Пробой канала Дончиана n свечей по закрытию; трейлинг k_trail × ATR; выход при
    пробое канала n/2 в обратную сторону. Приоритет — сила пробоя в ATR."""
    family = "breakout"
    grid = {"n": [20, 55], "k_stop": [2.0, 3.0], "k_trail": [3.0, 5.0], "sides": ["both", "long"],
            "regime": ["none", "btc", "fg"]}

    def prepare(self, bars: PanelBars) -> None:
        self._common(bars)
        n = self.p["n"]
        hp = lambda x, w: np.vstack([ind.highest_prev(x[r], w) for r in range(self.n)])  # noqa: E731
        lp = lambda x, w: np.vstack([ind.lowest_prev(x[r], w) for r in range(self.n)])  # noqa: E731
        self.hh, self.ll = hp(self.h, n), lp(self.l, n)
        self.hx, self.lx = hp(self.h, n // 2), lp(self.l, n // 2)
        self.warmup_bars = max(n, ATR_N) + 1

    def _signal_ok(self, r: int, k: int, side: int) -> bool:
        return True

    def on_bar(self, k: int, positions: dict):
        exits, entries = [], []
        k_trail = self.p.get("k_trail", 3.0)
        for r in range(self.n):
            c, a = self.c[r, k], self.atr[r, k]
            pos = positions.get(r)
            if pos is not None:
                if np.isnan(c):
                    continue
                if (pos.side == LONG and c < self.lx[r, k]) or (pos.side == SHORT and c > self.hx[r, k]):
                    exits.append((r, Exit("channel_exit")))
                elif not np.isnan(a):
                    ms = self._chandelier(r, k, pos, k_trail)
                    if ms is not None:
                        exits.append((r, ms))
                continue
            hh, ll = self.hh[r, k], self.ll[r, k]
            if np.isnan(c) or np.isnan(hh) or np.isnan(a) or a <= 0:
                continue
            e = None
            if c > hh and self._signal_ok(r, k, LONG):
                e = self._enter(r, k, LONG, (c - hh) / a)
            elif c < ll and self.p["sides"] == "both" and self._signal_ok(r, k, SHORT):
                e = self._enter(r, k, SHORT, (ll - c) / a)
            if e:
                entries.append(e)
        return self._ordered(exits, entries)


# ------------------------------------------------------------------ F4
class OiBreakout(Breakout):
    """Пробой (как breakout, трейлинг 3 × ATR) только при росте открытого интереса за те же
    n свечей не меньше чем на oi_min."""
    family = "oi_breakout"
    grid = {"n": [20, 55], "k_stop": [2.0, 3.0], "oi_min": [0.0, 0.05, 0.15], "sides": ["both", "long"]}
    timeframes = ("4h",)

    def prepare(self, bars: PanelBars) -> None:
        super().prepare(bars)
        oi = bars.aux("oi_1h")[:self.n]
        n = self.p["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            self.doi = oi / np.concatenate([np.full((self.n, n), np.nan), oi[:, :-n]], axis=1) - 1

    def _signal_ok(self, r: int, k: int, side: int) -> bool:
        d = self.doi[r, k]
        return bool(d >= self.p["oi_min"]) if not np.isnan(d) else False


# ------------------------------------------------------------ межмонетные
class CrossSection(BasketStrategy):
    """Общая механика межмонетных стратегий: score (n, n_bars) — чем выше, тем лучше для лонга.

    mode=ls: лонг 2 лучших, шорт 2 худших; long/short — до 4 позиций в одну сторону.
    Позиция закрывается, когда монета выпадает из зоны «топ k + BUFFER» своей стороны.
    """
    BUFFER = 2
    score: np.ndarray

    def _cs_prepare(self, bars: PanelBars, warm: int) -> None:
        self.warmup_bars = max(warm, ATR_N) + 1
        ok = (~np.isnan(self.score)) & (self.age >= self.warmup_bars) & (~np.isnan(self.c))
        self.ok = ok

    def on_bar(self, k: int, positions: dict):
        mode = self.p["mode"]
        n_long = {"ls": 2, "long": 4, "short": 0}[mode]
        n_short = {"ls": 2, "long": 0, "short": 4}[mode]
        avail = np.flatnonzero(self.ok[:, k])
        if len(avail) < max(n_long, n_short) * 2 + 1:
            return []
        order = avail[np.argsort(-self.score[avail, k], kind="stable")]   # лучшие для лонга — первыми
        rank_long = {r: i for i, r in enumerate(order)}
        rank_short = {r: i for i, r in enumerate(order[::-1])}
        exits, entries = [], []
        for r, pos in positions.items():
            if r not in rank_long:
                continue   # данных на этой свече нет — держим
            if pos.side == LONG and rank_long[r] >= n_long + self.BUFFER:
                exits.append((r, Exit("rank")))
            elif pos.side == SHORT and rank_short[r] >= n_short + self.BUFFER:
                exits.append((r, Exit("rank")))
        for i in range(max(n_long, n_short)):
            if i < n_long:
                r = int(order[i])
                if r not in positions:
                    e = self._enter(r, k, LONG, -i)
                    if e:
                        entries.append(e)
            if i < n_short:
                r = int(order[::-1][i])
                if r not in positions:
                    e = self._enter(r, k, SHORT, -i - 0.5)
                    if e:
                        entries.append(e)
        return self._ordered(exits, entries)


class XsMom(CrossSection):
    """Межмонетный импульс (dir=mom) или разворот (dir=rev) по доходности за L дней / (σ·√L)."""
    family = "xsmom"
    grid = {"L": [1, 3, 7, 14, 30], "dir": ["mom", "rev"], "k_stop": [2.0, 3.0], "mode": ["ls", "long"]}

    def prepare(self, bars: PanelBars) -> None:
        self._common(bars)
        L = self.p["L"] * self.bpd
        s = self._vol_adj_return(L)
        self.score = s if self.p["dir"] == "mom" else -s
        self._cs_prepare(bars, L + VOL_DAYS * self.bpd)


class Carry(CrossSection):
    """Против перегретых: средняя ставка финансирования (в сутки) или средний премиальный
    индекс за D дней; лонг самых «дешёвых», шорт самых «дорогих»."""
    family = "carry"
    grid = {"signal": ["funding", "premium"], "D": [1, 3], "k_stop": [2.0, 3.0], "mode": ["ls", "short"]}

    def prepare(self, bars: PanelBars) -> None:
        self._common(bars)
        D = self.p["D"]
        t = bars.close_ts
        carry = np.full((self.n, len(bars)), np.nan)
        for r, s in enumerate(bars.symbols[:self.n]):
            sd = bars.panel.sym[s]
            if self.p["signal"] == "funding":
                tot, cnt, _ = window_sum(sd.funding_ts, sd.funding_rate, t, D * DAY_MS)
                v = tot / D                                   # ставка за сутки
            else:
                ser = sd.aux.get("premium_1h")
                if ser is None:
                    continue
                tot, cnt, _ = window_sum(ser.known_ts, ser.value, t, D * DAY_MS)
                with np.errstate(invalid="ignore", divide="ignore"):
                    v = tot / cnt
            v = np.where(cnt > 0, v, np.nan)
            carry[r] = v
        carry[np.isnan(self.c)] = np.nan
        self.carry = carry
        self.score = -carry
        self._cs_prepare(bars, D * self.bpd)


class Crowd(CrossSection):
    """Против толпы: z доли лонгов по счетам относительно среднего за W дней (часовые значения);
    лонг там, где толпа необычно в шортах, шорт — где необычно в лонгах."""
    family = "crowd"
    grid = {"W": [3, 7, 30], "k_stop": [2.0, 3.0], "mode": ["ls", "long"]}

    def prepare(self, bars: PanelBars) -> None:
        self._common(bars)
        W = self.p["W"]
        t = bars.close_ts
        lsr = bars.aux("lsr_1h")[:self.n]
        z = np.full((self.n, len(bars)), np.nan)
        for r, s in enumerate(bars.symbols[:self.n]):
            ser = bars.panel.sym[s].aux.get("lsr_1h")
            if ser is None:
                continue
            tot, cnt, tot2 = window_sum(ser.known_ts, ser.value, t, W * DAY_MS)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = tot / cnt
                sd = np.sqrt(np.maximum(tot2 / cnt - mean ** 2, 0.0))
                zz = (lsr[r] - mean) / sd
            z[r] = np.where(cnt >= W * 24 * 0.8, zz, np.nan)
        z[np.isnan(self.c)] = np.nan
        self.z = z
        self.score = -z
        self._cs_prepare(bars, W * self.bpd)


FAMILIES: dict[str, type[BasketStrategy]] = {c.family: c for c in (TsMom, Breakout, XsMom, Carry, OiBreakout, Crowd)}
COMBOS: list[tuple[str, str]] = [(f, tf) for f, cls in FAMILIES.items() for tf in cls.timeframes]


def grid_configs(family: str) -> list[dict]:
    g = FAMILIES[family].grid
    keys = list(g)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(g[k] for k in keys))]


def make(family: str, timeframe: str, params: dict) -> BasketStrategy:
    return FAMILIES[family](timeframe=timeframe, **params)
