"""Портфельные правила риска — один код для бэктеста и боевого движка.

Правила (docs/RISK_PROTOCOL.md):
  - не больше max_positions позиций;
  - суммарный открытый риск (убыток до текущих стопов) плюс риск новой позиции
    ≤ max_open_risk × капитал; при downsize_to_fit новая позиция уменьшается до остатка
    лимита, иначе пропускается;
  - лимит риска с учётом корреляции (corr_cap): H = √(Σ hᵢ hⱼ dᵢ dⱼ ρᵢⱼ) ≤ corr_cap × капитал,
    ρ — корреляция доходностей свечей стратегии за corr_days дней;
  - снижение риска при высокой волатильности рынка (vol_scaling): риск × min(1, σ_ref / σ);
  - правило ликвидности: монета без новых входов, если за 30 дней до начала суток UTC
    шаг цены слишком крупный, оборот мал или торги прерывались.

Все ряды причинны: значение на свече k зависит только от данных, известных на её закрытии.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.data.bars import TF_MINUTES
from bot.data.panel import BASE_MS, PanelBars
from bot.market import Instrument
from bot.risk.sizing import Sizing, SizingParams, Skip, size_by_margin, size_position

DAY_MS = 86_400_000
EPS = 1e-9


@dataclass(frozen=True)
class LiquidityRule:
    days: int = 30
    max_tick_frac: float = 0.0004        # шаг цены / медианная цена
    min_turnover: float = 10_000_000.0   # медианный дневной оборот, USDT
    max_zero_volume_frac: float = 0.01   # доля 15m-свечей без сделок


@dataclass(frozen=True)
class PortfolioRules:
    max_positions: int = 4
    max_open_risk: float = 0.20
    downsize_to_fit: bool = False
    corr_cap: float | None = None
    corr_days: int = 30
    corr_min_bars: int = 120
    vol_scaling: bool = False
    vol_days: int = 30
    vol_ref_days: int = 365
    vol_ref_min_days: int = 180
    liquidity: LiquidityRule | None = None
    sizing: str = "risk"            # risk — от риска до стопа; margin — маржа = доля баланса, плечо максимальное
    margin_fraction: float = 0.25
    margin_stop: str = "none"       # для sizing=margin: none | near_liq | gap | roi (bot.risk.sizing.size_by_margin)
    liq_gap: float = 0.035
    roi_gap: float = 0.04           # для margin_stop=roi: стоп на столько долей маржи раньше ликвидации
    liq_formula: str = "model"      # для sizing=margin: model (осторожная) | bybit (как на бирже)


@dataclass(frozen=True)
class Exposure:
    """Позиция (или уже запланированный вход) для лимитов риска."""
    row: int
    side: int
    risk: float     # USDT до стопа, ≥ 0


def liquidity_mask(bars: PanelBars, rule: LiquidityRule) -> np.ndarray:
    """(n_trade, n_bars): можно ли открывать новые позиции по монете на закрытии свечи."""
    P = bars.panel
    n = bars.n_trade
    if len(P.ts) == 0 or len(bars) == 0:
        return np.zeros((n, len(bars)), bool)
    day = P.ts // DAY_MS
    d0 = int(day[0])
    di = (day - d0).astype(np.int64)
    nd = int(di[-1]) + 1
    per_day = DAY_MS // BASE_MS
    out = np.zeros((n, len(bars)), bool)
    bar_day = (bars.close_ts // DAY_MS - d0).astype(np.int64)      # сутки решения
    W = rule.days
    for r in range(n):
        ok = ~np.isnan(P.close[r])
        cnt = np.bincount(di[ok], minlength=nd).astype(float)
        turn = np.bincount(di[ok], weights=np.nan_to_num(P.turnover[r][ok]), minlength=nd)
        zero = np.bincount(di[ok], weights=(np.nan_to_num(P.volume[r][ok]) <= 0).astype(float), minlength=nd)
        # цена закрытия дня — последняя 15m-свеча суток
        close_d = np.full(nd, np.nan)
        idx = np.flatnonzero(ok)
        if len(idx):
            last = np.r_[np.flatnonzero(np.diff(di[idx]) != 0), len(idx) - 1]
            close_d[di[idx[last]]] = P.close[r][idx[last]]
        full = cnt >= per_day * 0.99
        s_turn = pd.Series(np.where(full, turn, np.nan))
        med_turn = s_turn.rolling(W, min_periods=W).median().to_numpy()
        med_px = pd.Series(close_d).rolling(W, min_periods=W).median().to_numpy()
        zf = pd.Series(zero).rolling(W, min_periods=W).sum().to_numpy() / \
            np.maximum(pd.Series(cnt).rolling(W, min_periods=W).sum().to_numpy(), 1)
        tick = float(P.sym[P.symbols[r]].inst.tick_size)
        with np.errstate(invalid="ignore", divide="ignore"):
            good = (tick / med_px <= rule.max_tick_frac + 1e-15) & (med_turn >= rule.min_turnover) & \
                   (zf <= rule.max_zero_volume_frac)
        good = np.where(np.isnan(med_turn) | np.isnan(med_px), False, good)
        # сутки D разрешены по окну D−W … D−1
        allowed = np.zeros(nd + 1, bool)
        allowed[1:] = good
        bd = np.clip(bar_day, 0, nd)
        out[r] = allowed[bd]
    return out


class PortfolioRisk:
    """Ряды и проверки портфельного риска для одной сетки свечей стратегии."""

    def __init__(self, bars: PanelBars, rules: PortfolioRules):
        self.bars, self.rules = bars, rules
        n = bars.n_trade
        self.n = n
        self.bpd = 1440 // TF_MINUTES[bars.tf]
        with np.errstate(invalid="ignore", divide="ignore"):
            logc = np.log(bars.close[:n])
        self.ret = np.diff(logc, axis=1, prepend=np.nan) if len(bars) else np.zeros((n, 0))
        self.vol_mult = np.ones(len(bars))
        if rules.vol_scaling and len(bars):
            w = rules.vol_days * self.bpd
            sd = np.vstack([pd.Series(self.ret[r]).rolling(w, min_periods=w).std(ddof=1).to_numpy()
                            for r in range(n)])
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sigma = np.nanmedian(sd, axis=0) if n else np.full(len(bars), np.nan)
            ref = pd.Series(sigma).rolling(rules.vol_ref_days * self.bpd,
                                           min_periods=rules.vol_ref_min_days * self.bpd).median().to_numpy()
            with np.errstate(invalid="ignore", divide="ignore"):
                m = np.minimum(1.0, ref / sigma)
            self.vol_mult = np.where(np.isfinite(m) & (m > 0), m, 1.0)
        self.liquid = liquidity_mask(bars, rules.liquidity) if rules.liquidity else None
        self._corr_cache: dict[int, np.ndarray] = {}

    # --------------------------------------------------------------- ряды
    def is_liquid(self, r: int, k: int) -> bool:
        return True if self.liquid is None else bool(self.liquid[r, k])

    def corr(self, k: int) -> np.ndarray:
        if k not in self._corr_cache:
            w = self.rules.corr_days * self.bpd
            win = self.ret[:, max(0, k - w + 1):k + 1]
            c = pd.DataFrame(win.T).corr(min_periods=self.rules.corr_min_bars).to_numpy()
            c = np.where(np.isnan(c), 1.0, c)
            np.fill_diagonal(c, 1.0)
            if len(self._corr_cache) > 64:
                self._corr_cache.clear()
            self._corr_cache[k] = c
        return self._corr_cache[k]

    def corr_heat(self, k: int, book: list[Exposure]) -> float:
        if not book:
            return 0.0
        c = self.corr(k)
        v = np.array([e.side * e.risk for e in book])
        rows = [e.row for e in book]
        return math.sqrt(max(0.0, float(v @ c[np.ix_(rows, rows)] @ v)))

    def corr_room(self, k: int, book: list[Exposure], row: int, side: int, cap_usdt: float) -> float:
        """Наибольший риск новой позиции (USDT), при котором H ≤ cap_usdt."""
        h0 = self.corr_heat(k, book)
        if h0 >= cap_usdt:
            return 0.0
        c = self.corr(k)
        lin = sum(e.risk * e.side * side * c[e.row, row] for e in book)
        return -lin + math.sqrt(lin * lin + cap_usdt * cap_usdt - h0 * h0)

    # ------------------------------------------------------------- решение
    def plan_entry(self, *, k: int, row: int, side: int, px: float, stop: float, equity: float,
                   available: float, inst: Instrument, sp: SizingParams, risk_mult: float,
                   book: list[Exposure]) -> Sizing | Skip:
        """Объём новой позиции по всем портфельным правилам (места и маржу проверяет вызывающий)."""
        R = self.rules
        if R.sizing == "margin":
            s = size_by_margin(side, px, stop, equity * R.margin_fraction * risk_mult * float(self.vol_mult[k]),
                               available, inst, sp, stop_rule=R.margin_stop, liq_gap=R.liq_gap,
                               roi_gap=R.roi_gap, liq_formula=R.liq_formula)
            if isinstance(s, Skip):
                return s
            if s.planned_loss > R.max_open_risk * equity * (1 + EPS) - sum(e.risk for e in book):
                return Skip("превышен суммарный риск открытых позиций")
            return s
        base = equity * sp.risk_per_trade * risk_mult * float(self.vol_mult[k])
        used = sum(e.risk for e in book)
        sum_room = R.max_open_risk * equity * (1 + EPS) - used
        if not R.downsize_to_fit:
            s = size_position(side, px, stop, equity, available, inst, sp, risk_budget=base)
            if isinstance(s, Skip):
                return s
            if s.planned_loss > sum_room:
                return Skip("превышен суммарный риск открытых позиций")
            if R.corr_cap is not None and self.corr_heat(k, book + [Exposure(row, side, s.planned_loss)]) > \
                    R.corr_cap * equity * (1 + EPS):
                return Skip("превышен лимит риска с учётом корреляции")
            return s
        budget, why = base, ""
        if sum_room < budget:
            budget, why = sum_room, "суммарный риск"
        if R.corr_cap is not None:
            cr = self.corr_room(k, book, row, side, R.corr_cap * equity * (1 + EPS))
            if cr < budget:
                budget, why = cr, "риск с учётом корреляции"
        if budget <= 0:
            return Skip(f"превышен лимит: {why}")
        s = size_position(side, px, stop, equity, available, inst, sp, risk_budget=budget)
        if isinstance(s, Skip) and why and "ниже минимума" in s.reason:
            return Skip(f"объём ниже минимума после уменьшения по лимиту ({why})")
        return s
