"""Событийный бэктестер без заглядывания в будущее.

Время идёт свечами таймфрейма стратегии, внутри свечи — по минутам.

Порядок событий для свечи [T, T+tf):
  1. Финансирование в момент T — за позицию, открытую ДО T.
  2. Исполнение решения, принятого на закрытии предыдущей свечи (момент T):
     вход/выход по цене открытия минуты T ± проскальзывание, подтяжка стопа.
  3. Поминутная проверка выходов: стоп и тейк (по последней цене),
     ликвидация (по mark-цене), аварийный порог просадки. Финансирование
     внутри свечи — в своих минутах.
  4. Закрытие свечи T+tf: переоценка капитала, проверки риска, затем решение
     стратегии по ЗАКРЫТОЙ свече. Оно исполнится не раньше минуты T+tf.

Консервативные допущения:
  - стоп и тейк в одной минуте → сначала стоп (если цена не открылась за тейком);
  - стоп-маркет исполняется хуже стопа: base − stop_penetration × (base − экстремум
    минуты) − проскальзывание; при гэпе base = цена открытия;
  - тейк — лимитный reduce-only по цене тейка, только если цена прошла его
    минимум на тик; комиссия мейкера;
  - ликвидация (изолированная маржа) = потеря всей маржи позиции.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.config import CostsCfg, RiskCfg
from bot.data.bars import Bars, MinuteData, aggregate
from bot.market import Instrument
from bot.risk.guards import GuardEvent, RiskGuard, RiskState
from bot.risk.sizing import (LONG, SHORT, Sizing, SizingParams, Skip, liquidation_price,
                             max_leverage_for_stop, size_position)
from bot.strategy.base import Enter, Exit, MoveStop, PositionView, Strategy

MINUTE_MS = 60_000


@dataclass
class EngineConfig:
    risk: RiskCfg
    costs: CostsCfg
    initial_equity: float
    trade_start_ms: int | None = None   # входы только при решении в [start, end)
    trade_end_ms: int | None = None
    close_at_end: bool = True
    # Состояние ограничителей из предыдущего окна (пик капитала, серия, пауза, остановка) —
    # для склейки окон walk-forward в непрерывную историю.
    initial_risk_state: RiskState | None = None


@dataclass
class Position:
    side: int
    qty: float
    entry_price: float
    entry_ts: int
    decision_ts: int
    stop: float
    stop_initial: float
    take_profit: float | None
    leverage: float
    margin: float
    liq_price: float
    entry_fee: float
    planned_loss: float
    tag: str
    funding: float = 0.0       # уплачено финансирования (минус — получено)
    bars_held: int = 0


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    events: list[GuardEvent]
    skips: list[tuple[int, str]]
    decisions: list[tuple[int, str]]
    bars: Bars
    final_equity: float
    halted: bool
    params: dict = field(default_factory=dict)
    risk_state: RiskState | None = None


class Backtester:
    def __init__(self, md: MinuteData, funding_ts: np.ndarray, funding_rate: np.ndarray,
                 inst: Instrument, strategy: Strategy, cfg: EngineConfig):
        self.md, self.inst, self.strategy, self.cfg = md, inst, strategy, cfg
        self.f_ts, self.f_rate = funding_ts, funding_rate
        c = cfg.costs
        self.sp = SizingParams(risk_per_trade=cfg.risk.risk_per_trade, taker_fee=c.taker_fee,
                               slippage=c.slippage, liq_buffer=cfg.risk.liq_distance_vs_stop_min,
                               max_leverage=cfg.risk.max_leverage)
        self.tick = float(inst.tick_size)

    # ------------------------------------------------------------------ run
    def run(self) -> BacktestResult:
        md, strat, cfg = self.md, self.strategy, self.cfg
        bars = aggregate(md, strat.timeframe)
        strat.prepare(bars)
        self.cash = float(cfg.initial_equity)
        self.pos: Position | None = None
        self.trades: list[dict] = []
        self.events: list[GuardEvent] = []
        self.skips: list[tuple[int, str]] = []
        self.decisions: list[tuple[int, str]] = []
        eq_rows: list[tuple[int, float, float, int]] = []
        if len(bars) == 0:
            return self._result(bars, eq_rows)
        state = copy.deepcopy(cfg.initial_risk_state) if cfg.initial_risk_state else None
        self.guard = RiskGuard(cfg.risk, self.cash, int(bars.ts[0]), state)
        pending: tuple | None = None       # ("enter", decision_ts, Enter, Sizing) | ("exit", decision_ts, reason)
        pending_stop: float | None = None

        for k in range(len(bars)):
            t_open, t_close = int(bars.ts[k]), int(bars.close_ts[k])
            ms, me = int(bars.m_start[k]), int(bars.m_end[k])
            f_lo = int(np.searchsorted(self.f_ts, t_open, "left"))
            f_hi = int(np.searchsorted(self.f_ts, t_close, "left"))

            # 1. финансирование ровно в момент открытия свечи — за позицию, открытую до него
            fi = f_lo
            while fi < f_hi and self.f_ts[fi] == t_open:
                self._apply_funding(fi, ms)
                fi += 1
            # 2. исполнение решения прошлой свечи
            if pending is not None:
                if pending[0] == "enter" and self.pos is None:
                    self._execute_entry(ms, pending)
                elif pending[0] == "exit" and self.pos is not None:
                    self._close_market(ms, pending[2])
                pending = None
            if pending_stop is not None and self.pos is not None:
                self.pos.stop = pending_stop
            pending_stop = None
            # 3. поминутная проверка выходов, сегментами между выплатами финансирования
            seg_a = ms
            while True:
                seg_b = me
                if fi < f_hi:
                    seg_b = ms + int(np.searchsorted(md.ts[ms:me], self.f_ts[fi], "left"))
                if self.pos is not None and seg_b > seg_a:
                    self._scan_exits(seg_a, seg_b)
                if fi >= f_hi:
                    break
                if seg_b < me and md.ts[seg_b] == self.f_ts[fi]:
                    self._apply_funding(fi, seg_b)
                fi += 1
                seg_a = seg_b
            # 4. закрытие свечи
            px = float(bars.close[k])
            equity = self.cash + self._upl(px)
            self.events += self.guard.update_equity(t_close, equity)
            eq_rows.append((t_close, equity, self.cash, 0 if self.pos is None else self.pos.side))
            if self.pos is not None:
                self.pos.bars_held += 1
            if self.guard.state.halted:
                if self.pos is not None:
                    pending = ("exit", t_close, "max_drawdown")
                continue
            if k < strat.warmup_bars:
                continue
            view = None if self.pos is None else PositionView(
                self.pos.side, self.pos.entry_price, self.pos.qty, self.pos.stop,
                self.pos.take_profit, self.pos.entry_ts, self.pos.bars_held)
            d = strat.on_bar(k, view)
            if d is not None:
                self.decisions.append((t_close, repr(d)))
            if isinstance(d, Enter) and self.pos is None:
                pending = self._plan_entry(t_close, px, d)
            elif isinstance(d, Exit) and self.pos is not None:
                pending = ("exit", t_close, d.reason)
            elif isinstance(d, MoveStop) and self.pos is not None:
                pending_stop = self._plan_stop(d.stop, px)

        if self.pos is not None and cfg.close_at_end:
            self._close_market(len(md) - 1, "end", at_close=True)
            eq_rows[-1] = (eq_rows[-1][0], self.cash, self.cash, 0)
        return self._result(bars, eq_rows)

    # ------------------------------------------------------------ planning
    def _plan_entry(self, t_close: int, px: float, d: Enter):
        cfg = self.cfg
        if (cfg.trade_start_ms is not None and t_close < cfg.trade_start_ms) or \
           (cfg.trade_end_ms is not None and t_close >= cfg.trade_end_ms):
            return None
        ok, why = self.guard.can_open(t_close)
        if not ok:
            self.skips.append((t_close, why))
            return None
        s = size_position(d.side, px, d.stop, self.cash, self.cash, self.inst, self.sp)
        if isinstance(s, Skip):
            self.skips.append((t_close, s.reason))
            return None
        return ("enter", t_close, d, s)

    def _plan_stop(self, stop: float, px: float) -> float | None:
        p = self.pos
        stop = self.inst.round_price(stop, "down" if p.side == LONG else "up")
        tighter = stop > p.stop if p.side == LONG else stop < p.stop
        valid = stop < px if p.side == LONG else stop > px
        return stop if (tighter and valid) else None

    # ----------------------------------------------------------- execution
    def _execute_entry(self, m: int, pending) -> None:
        _, decision_ts, d, s = pending
        s: Sizing
        side, slip, fee = d.side, self.cfg.costs.slippage, self.cfg.costs.taker_fee
        fill = float(self.md.open[m]) * (1 + side * slip)
        if (side == LONG and fill <= s.stop) or (side == SHORT and fill >= s.stop):
            self.skips.append((int(self.md.ts[m]), "цена прошла стоп до исполнения входа"))
            return
        # Плечо пересчитывается от фактической цены входа: ликвидация дальше стопа с запасом.
        tier = self.inst.tier_for(s.qty * fill)
        dist = abs(fill - s.stop) / fill
        lev = self.inst.floor_leverage(min(s.leverage, tier.max_leverage, max_leverage_for_stop(
            dist, tier.mmr, fee, self.sp.liq_buffer)))
        qty = s.qty
        if lev < 1:
            self.skips.append((int(self.md.ts[m]), "ликвидация не помещается за стопом"))
            return
        if qty * fill * (1 / lev + fee) > self.cash:
            qty = self.inst.floor_qty(self.cash / (fill * (1 / lev + fee)))
        if qty < self.inst.min_qty or qty * fill < self.inst.min_notional:
            self.skips.append((int(self.md.ts[m]), "после пересчёта объём ниже минимума"))
            return
        tp = d.take_profit
        if tp is not None:
            tp = self.inst.round_price(tp)
            if (side == LONG and tp <= fill) or (side == SHORT and tp >= fill):
                tp = None   # тейк уже позади цены входа — биржа такой не примет
        entry_fee = qty * fill * fee
        self.cash -= entry_fee
        cost = fee + slip
        self.pos = Position(
            side=side, qty=qty, entry_price=fill, entry_ts=int(self.md.ts[m]), decision_ts=decision_ts,
            stop=s.stop, stop_initial=s.stop, take_profit=tp, leverage=lev,
            margin=qty * fill / lev, liq_price=liquidation_price(side, fill, lev, tier.mmr, fee),
            entry_fee=entry_fee, planned_loss=qty * (abs(fill - s.stop) + fill * cost + s.stop * cost),
            tag=d.tag,
        )

    def _apply_funding(self, fi: int, m: int) -> None:
        if self.pos is None:
            return
        payment = self.pos.side * self.pos.qty * float(self.md.mark_open[m]) * float(self.f_rate[fi])
        self.cash -= payment
        self.pos.funding += payment

    def _upl(self, px: float) -> float:
        p = self.pos
        return 0.0 if p is None else p.side * p.qty * (px - p.entry_price)

    def _kill_price(self) -> float | None:
        """Цена, при которой капитал опускается до порога максимальной просадки."""
        p = self.pos
        room = self.cash - self.guard.drawdown_floor()
        return p.entry_price - p.side * room / p.qty

    def _scan_exits(self, a: int, b: int) -> None:
        md, p, tick = self.md, self.pos, self.tick
        kill = self._kill_price()
        if p.side == LONG:
            stop_hit = md.low[a:b] <= p.stop
            kill_hit = md.low[a:b] <= kill
            liq_hit = md.mark_low[a:b] <= p.liq_price
            tp_hit = md.high[a:b] >= p.take_profit + tick if p.take_profit else None
        else:
            stop_hit = md.high[a:b] >= p.stop
            kill_hit = md.high[a:b] >= kill
            liq_hit = md.mark_high[a:b] >= p.liq_price
            tp_hit = md.low[a:b] <= p.take_profit - tick if p.take_profit else None
        hit = stop_hit | kill_hit | liq_hit
        if tp_hit is not None:
            hit = hit | tp_hit
        idx = np.flatnonzero(hit)
        if idx.size == 0:
            return
        i = int(idx[0])
        m = a + i
        o = float(md.open[m])
        if tp_hit is not None and tp_hit[i]:
            gapped_through_tp = o >= p.take_profit + tick if p.side == LONG else o <= p.take_profit - tick
            if gapped_through_tp or not (stop_hit[i] or kill_hit[i] or liq_hit[i]):
                self._close(m, p.take_profit, "take_profit", self.cfg.costs.maker_fee)
                return
        levels = [(p.stop, "stop")] if stop_hit[i] else []
        if kill_hit[i]:
            levels.append((kill, "max_drawdown"))
        if levels:
            # первым срабатывает уровень, ближайший к цене (выше для лонга, ниже для шорта)
            lvl, reason = max(levels) if p.side == LONG else min(levels)
            if p.side == LONG:
                base = min(o, lvl)
                fill = base - self.cfg.costs.stop_penetration * max(0.0, base - float(md.low[m]))
            else:
                base = max(o, lvl)
                fill = base + self.cfg.costs.stop_penetration * max(0.0, float(md.high[m]) - base)
            fill *= 1 - p.side * self.cfg.costs.slippage
            # Исполнение хуже цены ликвидации невозможно: позицию раньше ликвидирует биржа,
            # а убыток в изолированной марже ограничен маржой позиции.
            beyond_liq = fill <= p.liq_price if p.side == LONG else fill >= p.liq_price
            if not beyond_liq:
                self._close(m, fill, reason, self.cfg.costs.taker_fee)
                if reason == "max_drawdown" and not self.guard.state.halted:
                    self.events.append(self.guard.halt(int(md.ts[m]), "порог максимальной просадки внутри свечи",
                                                       kind="max_drawdown"))
                return
        self._liquidate(m)

    def _close_market(self, m: int, reason: str, at_close: bool = False) -> None:
        px = float(self.md.close[m] if at_close else self.md.open[m])
        self._close(m, px * (1 - self.pos.side * self.cfg.costs.slippage), reason, self.cfg.costs.taker_fee)

    def _liquidate(self, m: int) -> None:
        p = self.pos
        bankruptcy = p.entry_price * (1 - p.side / p.leverage)   # PnL по этой цене = −маржа
        self._close(m, bankruptcy, "liquidation", 0.0)

    def _close(self, m: int, price: float, reason: str, fee_rate: float) -> None:
        p = self.pos
        ts = int(self.md.ts[m])
        gross = p.side * p.qty * (price - p.entry_price)
        exit_fee = p.qty * price * fee_rate
        self.cash += gross - exit_fee
        net = gross - p.entry_fee - exit_fee - p.funding
        self.trades.append({
            "decision_ts": p.decision_ts, "entry_ts": p.entry_ts, "exit_ts": ts,
            "side": p.side, "qty": p.qty, "entry_price": p.entry_price, "exit_price": price,
            "stop_initial": p.stop_initial, "stop_final": p.stop, "take_profit": p.take_profit,
            "leverage": p.leverage, "margin": p.margin, "liq_price": p.liq_price,
            "entry_fee": p.entry_fee, "exit_fee": exit_fee, "funding": p.funding,
            "gross_pnl": gross, "net_pnl": net, "planned_loss": p.planned_loss,
            "r_multiple": net / p.planned_loss if p.planned_loss > 0 else np.nan,
            "exit_reason": reason, "bars_held": p.bars_held, "tag": p.tag,
            "equity_after": self.cash,
        })
        self.pos = None
        self.events += self.guard.on_trade_closed(ts, net)
        self.events += self.guard.update_equity(ts, self.cash)

    # -------------------------------------------------------------- result
    def _result(self, bars: Bars, eq_rows) -> BacktestResult:
        trades = pd.DataFrame(self.trades)
        equity = pd.DataFrame(eq_rows, columns=["ts", "equity", "cash", "position"])
        return BacktestResult(
            trades=trades, equity=equity, events=self.events, skips=self.skips,
            decisions=self.decisions, bars=bars, final_equity=self.cash,
            halted=bool(getattr(self, "guard", None) and self.guard.state.halted),
            risk_state=copy.deepcopy(self.guard.state) if getattr(self, "guard", None) else None,
            params={"strategy": self.strategy.name, "timeframe": self.strategy.timeframe,
                    **self.strategy.params(), "costs": self.cfg.costs.model_dump()},
        )
