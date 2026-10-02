"""Планирование решений одной свечи стратегии — общий код бэктеста и боевого движка.

На входе — решения стратегии (on_bar) и состояние счёта на закрытии свечи; на выходе —
что исполнить по открытию следующей свечи: выходы, новые стопы, входы с объёмом и плечом,
а также причины пропуска сигналов. Исполнение (бэктест или биржа) — отдельно.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bot.market import Instrument
from bot.risk.guards import RiskGuard
from bot.risk.portfolio import Exposure, PortfolioRisk, PortfolioRules
from bot.risk.sizing import LONG, Sizing, SizingParams, Skip
from bot.strategy.base import Enter, Exit, MoveStop


@dataclass(frozen=True)
class Held:
    """Открытая позиция с точки зрения планирования."""
    side: int
    qty: float
    stop: float
    margin: float


@dataclass
class BarPlan:
    exits: dict[int, tuple[int, str]] = field(default_factory=dict)        # строка → (время решения, причина)
    stops: dict[int, float] = field(default_factory=dict)                  # строка → новый стоп
    entries: list[tuple[int, int, Enter, Sizing]] = field(default_factory=list)
    skips: list[tuple[int, str, str]] = field(default_factory=list)        # (время, монета, причина)
    decisions: list[tuple[int, str, str]] = field(default_factory=list)


def plan_stop(inst: Instrument, side: int, current: float, stop: float, px: float) -> float | None:
    """Новый стоп: округлён наружу, только ближе к цене (риск не растёт) и по правильную сторону цены."""
    stop = inst.round_price(stop, "down" if side == LONG else "up")
    tighter = stop > current if side == LONG else stop < current
    valid = stop < px if side == LONG else stop > px
    return stop if (tighter and valid) else None


class Planner:
    def __init__(self, symbols: list[str], inst: list[Instrument], sp: SizingParams, rules: PortfolioRules,
                 prisk: PortfolioRisk, guard: RiskGuard, trade_start_ms: int | None = None,
                 trade_end_ms: int | None = None):
        self.symbols, self.inst, self.sp, self.rules = symbols, inst, sp, rules
        self.prisk, self.guard = prisk, guard
        self.trade_start_ms, self.trade_end_ms = trade_start_ms, trade_end_ms

    def plan(self, ds: list, k: int, t_close: int, held: dict[int, Held], cash: float, last_px: np.ndarray,
             equity: float, entries_allowed: bool = True) -> BarPlan:
        """entries_allowed=False — только выходы и стопы (например, свеча обработана с опозданием)."""
        bp = BarPlan()
        for r, d in ds:
            if d is None:
                continue
            bp.decisions.append((t_close, self.symbols[r], repr(d)))
            if isinstance(d, Exit) and r in held:
                bp.exits[r] = (t_close, d.reason)
            elif isinstance(d, MoveStop) and r in held and r not in bp.exits:
                h = held[r]
                s = plan_stop(self.inst[r], h.side, h.stop, d.stop, float(last_px[r]))
                if s is not None:
                    bp.stops[r] = s
            elif isinstance(d, Enter) and r not in held:
                if not entries_allowed:
                    bp.skips.append((t_close, self.symbols[r], "новые входы на этой свече запрещены"))
                    continue
                e = self._plan_entry(r, k, t_close, d, equity, held, cash, last_px, bp)
                if e is not None:
                    bp.entries.append(e)
        return bp

    def _plan_entry(self, r: int, k: int, t_close: int, d: Enter, equity: float, held: dict[int, Held],
                    cash: float, last_px: np.ndarray, bp: BarPlan):
        sym = self.symbols[r]
        if (self.trade_start_ms is not None and t_close < self.trade_start_ms) or \
           (self.trade_end_ms is not None and t_close >= self.trade_end_ms):
            return None
        ok, why = self.guard.can_open(t_close)
        if not ok:
            bp.skips.append((t_close, sym, why))
            return None
        bars = self.prisk.bars
        px = float(bars.close[r, k])
        if np.isnan(px):
            return None
        if not self.prisk.is_liquid(r, k):
            bp.skips.append((t_close, sym, "монета не проходит правило ликвидности"))
            return None
        staying = [(q, h) for q, h in held.items() if q not in bp.exits]
        if len(staying) + len(bp.entries) >= self.rules.max_positions:
            bp.skips.append((t_close, sym, "нет свободных мест"))
            return None
        margin_used = sum(h.margin for _, h in staying) + sum(e[3].margin for e in bp.entries)
        available = cash - margin_used
        book = [Exposure(q, h.side, h.qty * max(0.0, h.side * (float(last_px[q]) - h.stop))) for q, h in staying]
        book += [Exposure(e[0], e[2].side, e[3].planned_loss) for e in bp.entries]
        s = self.prisk.plan_entry(k=k, row=r, side=d.side, px=px, stop=d.stop, equity=equity, available=available,
                                  inst=self.inst[r], sp=self.sp, risk_mult=self.guard.risk_multiplier(equity),
                                  book=book)
        if isinstance(s, Skip):
            bp.skips.append((t_close, sym, s.reason))
            return None
        return (r, t_close, d, s)
