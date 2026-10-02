"""Портфельный бэктестер: несколько монет на общем капитале.

Правила исполнения, издержки и ограничители — те же, что в движке одной монеты
(bot/backtest/engine.py); для одной монеты результаты совпадают сделка в сделку
(тест tests/test_portfolio.py). Отличия, которых нет у одной монеты:

  - капитал один на все позиции: размер считается от капитала с учётом нереализованного
    результата, маржа новой позиции — из свободного остатка (кошелёк минус маржа открытых);
  - не больше max_positions позиций одновременно;
  - суммарный риск открытых позиций (убыток до их текущих стопов) плюс риск новой не больше
    max_open_risk × капитал — иначе вход пропускается;
  - ограничители (дневной лимит, серия, просадка) — на уровне счёта;
  - аварийный порог просадки внутри свечи: для позиции считается цена, при которой
    капитал счёта упадёт до порога, если остальные позиции в той же 15m-свече окажутся
    в худшей точке (консервативно). Для одной позиции — ровно как в движке одной монеты.

Время идёт свечами таймфрейма стратегии; внутри свечи — по 15m-свечам панели.
Решение принимается на закрытии свечи и исполняется по открытию следующей 15m-свечи;
в одном цикле сначала исполняются выходы, затем входы — в порядке, заданном стратегией.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.config import CostsCfg, RiskCfg
from bot.data.panel import Panel, PanelBars, aggregate_panel
from bot.market import Instrument
from bot.risk.guards import GuardEvent, RiskGuard, RiskState
from bot.risk.planner import Held, Planner
from bot.risk.portfolio import PortfolioRisk, PortfolioRules
from bot.risk.sizing import (LONG, SHORT, Sizing, SizingParams, liquidation_price, max_leverage_for_gap,
                             max_leverage_for_stop, stop_near_liq, tighter_stop)
from bot.strategy.base import PositionView

EPS = 1e-9


@dataclass
class PortfolioConfig:
    risk: RiskCfg
    costs: CostsCfg
    initial_equity: float
    max_positions: int = 4
    max_open_risk: float = 0.20
    trade_start_ms: int | None = None
    trade_end_ms: int | None = None
    close_at_end: bool = True
    initial_risk_state: RiskState | None = None
    rules: PortfolioRules | None = None   # None — только max_positions и max_open_risk (как раньше)
    # Оптимистичная модель: стоп (по последней цене) срабатывает раньше ликвидации (по маркировочной), даже если
    # исполнился за ценой ликвидации; ликвидация — только когда свеча открылась за ней. Убыток не больше маржи.
    stop_beats_liq: bool = False

    def portfolio_rules(self) -> PortfolioRules:
        return self.rules or PortfolioRules(max_positions=self.max_positions, max_open_risk=self.max_open_risk)


class PortfolioStrategy:
    """Стратегия на всю корзину. on_bar возвращает список (строка монеты, решение) —
    входы исполняются в этом порядке (первые — приоритетнее, если не хватит мест/риска)."""
    name: str = "portfolio"
    timeframe: str = "1d"
    warmup_bars: int = 0

    def params(self) -> dict:
        return {}

    def prepare(self, bars: PanelBars) -> None:
        raise NotImplementedError

    def on_bar(self, k: int, positions: dict[int, PositionView]) -> list[tuple[int, object]]:
        raise NotImplementedError


@dataclass
class Pos:
    row: int
    symbol: str
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
    funding: float = 0.0
    bars_held: int = 0


@dataclass
class PortfolioResult:
    trades: pd.DataFrame
    equity: pd.DataFrame
    events: list[GuardEvent]
    skips: list[tuple[int, str, str]]
    decisions: list[tuple[int, str, str]]
    bars: PanelBars
    final_equity: float
    halted: bool
    params: dict = field(default_factory=dict)
    risk_state: RiskState | None = None
    open_positions: list[dict] = field(default_factory=list)   # при close_at_end=False


class PortfolioBacktester:
    def __init__(self, panel: Panel, strategy: PortfolioStrategy, cfg: PortfolioConfig):
        self.panel, self.strategy, self.cfg = panel, strategy, cfg
        c = cfg.costs
        self.sp = SizingParams(risk_per_trade=cfg.risk.risk_per_trade, taker_fee=c.taker_fee,
                               slippage=c.slippage, liq_buffer=cfg.risk.liq_distance_vs_stop_min,
                               max_leverage=cfg.risk.max_leverage)
        n_trade = len(panel.symbols)
        self.inst: list[Instrument] = [panel.sym[s].inst for s in panel.symbols]
        self.f_ts = [panel.sym[s].funding_ts for s in panel.symbols]
        self.f_rate = [panel.sym[s].funding_rate for s in panel.symbols]
        self.n_trade = n_trade

    # ------------------------------------------------------------------ run
    def run(self) -> PortfolioResult:
        P, strat, cfg = self.panel, self.strategy, self.cfg
        bars = aggregate_panel(P, strat.timeframe)
        strat.prepare(bars)
        self.bars = bars
        self.rules = cfg.portfolio_rules()
        self.prisk = PortfolioRisk(bars, self.rules)
        self.cash = float(cfg.initial_equity)
        self.pos: dict[int, Pos] = {}
        self.trades: list[dict] = []
        self.events: list[GuardEvent] = []
        self.skips: list[tuple[int, str, str]] = []
        self.decisions: list[tuple[int, str, str]] = []
        self.last_px = np.full(self.n_trade, np.nan)
        eq_rows: list[tuple[int, float, float, int]] = []
        if len(bars) == 0:
            return self._result(bars, eq_rows)
        state = copy.deepcopy(cfg.initial_risk_state) if cfg.initial_risk_state else None
        self.guard = RiskGuard(cfg.risk, self.cash, int(bars.ts[0]), state)
        self.planner = Planner(P.symbols, self.inst, self.sp, self.rules, self.prisk, self.guard,
                               cfg.trade_start_ms, cfg.trade_end_ms)
        exits: dict[int, tuple[int, str]] = {}        # строка → (время решения, причина)
        entries: list[tuple] = []                     # (строка, время решения, Enter, Sizing)
        stops: dict[int, float] = {}

        for k in range(len(bars)):
            t_open, t_close = int(bars.ts[k]), int(bars.close_ts[k])
            ms, me = int(bars.m_start[k]), int(bars.m_end[k])
            # 1. финансирование ровно в момент открытия свечи
            for r in list(self.pos):
                self._funding_at(r, t_open, ms)
            # 2. исполнение решений прошлой свечи: сначала выходы, потом входы
            for r, (_, reason) in exits.items():
                if r in self.pos:
                    self._close_market(r, ms, reason)
            for e in entries:
                if e[0] not in self.pos:
                    self._execute_entry(ms, e)
            for r, s in stops.items():
                if r in self.pos:
                    self.pos[r].stop = s
            exits, entries, stops = {}, [], {}
            # 3. проверка выходов по 15m-свечам, сегментами между выплатами финансирования
            if self.pos:
                self._scan_bar(t_open, t_close, ms, me)
            # 4. закрытие свечи
            cl = bars.close[:self.n_trade, k]
            ok = ~np.isnan(cl)
            self.last_px[ok] = cl[ok]
            equity = self.cash + self._upl_at(self.last_px)
            self.events += self.guard.update_equity(t_close, equity)
            eq_rows.append((t_close, equity, self.cash, len(self.pos)))
            for p in self.pos.values():
                p.bars_held += 1
            if self.guard.state.halted:
                exits = {r: (t_close, "max_drawdown") for r in self.pos}
                continue
            if k < strat.warmup_bars:
                continue
            views = {r: PositionView(p.side, p.entry_price, p.qty, p.stop, p.take_profit, p.entry_ts, p.bars_held)
                     for r, p in self.pos.items()}
            ds = strat.on_bar(k, views) or []
            held = {r: Held(p.side, p.qty, p.stop, p.margin) for r, p in self.pos.items()}
            plan = self.planner.plan(ds, k, t_close, held, self.cash, self.last_px, equity)
            exits, stops, entries = plan.exits, plan.stops, plan.entries
            self.skips += plan.skips
            self.decisions += plan.decisions

        if self.pos and cfg.close_at_end:
            last = len(P.ts) - 1
            for r in sorted(self.pos):
                self._close_market(r, last, "end", at_close=True)
            eq_rows[-1] = (eq_rows[-1][0], self.cash, self.cash, 0)
        return self._result(bars, eq_rows)

    # ----------------------------------------------------------- execution
    def _execute_entry(self, m: int, e) -> None:
        r, decision_ts, d, s = e
        s: Sizing
        P, inst = self.panel, self.inst[r]
        side, slip, fee = d.side, self.cfg.costs.slippage, self.cfg.costs.taker_fee
        o = float(P.open[r, m])
        ts = int(P.ts[m])
        if np.isnan(o):
            self.skips.append((ts, P.symbols[r], "нет цены для входа"))
            return
        fill = o * (1 + side * slip)
        if (side == LONG and fill <= s.stop) or (side == SHORT and fill >= s.stop):
            self.skips.append((ts, P.symbols[r], "цена прошла стоп до исполнения входа"))
            return
        tier = inst.tier_for(s.qty * fill)
        dist = abs(fill - s.stop) / fill
        R = self.rules
        stop = s.stop
        if R.sizing == "margin" and R.margin_stop == "gap":     # ликвидация на liq_gap цены дальше стопа
            lev = inst.floor_leverage(min(s.leverage, tier.max_leverage,
                                          max_leverage_for_gap(dist, R.liq_gap, tier.mmr, fee)))
        elif R.sizing == "margin":       # плечо — максимум биржи, правило «ликвидация вдвое дальше» не действует
            lev = inst.floor_leverage(min(s.leverage, tier.max_leverage))
            if R.margin_stop == "near_liq" and lev >= 1:        # стоп — от фактической цены входа
                stop = tighter_stop(side, d.stop, stop_near_liq(
                    side, fill, liquidation_price(side, fill, lev, tier.mmr, fee), R.liq_gap, inst))
                if (side == LONG and stop >= fill) or (side == SHORT and stop <= fill):
                    self.skips.append((ts, P.symbols[r], "стоп у ликвидации совпал с ценой входа"))
                    return
        else:
            lev = inst.floor_leverage(min(s.leverage, tier.max_leverage, max_leverage_for_stop(
                dist, tier.mmr, fee, self.sp.liq_buffer)))
        qty = s.qty
        if lev < 1:
            self.skips.append((ts, P.symbols[r], "ликвидация не помещается за стопом"))
            return
        available = self.cash - sum(p.margin for p in self.pos.values())
        if qty * fill * (1 / lev + fee) > available:
            qty = inst.floor_qty(available / (fill * (1 / lev + fee)))
        if qty < inst.min_qty or qty * fill < inst.min_notional:
            self.skips.append((ts, P.symbols[r], "после пересчёта объём ниже минимума"))
            return
        tp = d.take_profit
        if tp is not None:
            tp = inst.round_price(tp)
            if (side == LONG and tp <= fill) or (side == SHORT and tp >= fill):
                tp = None
        entry_fee = qty * fill * fee
        self.cash -= entry_fee
        cost = fee + slip
        self.pos[r] = Pos(
            row=r, symbol=P.symbols[r], side=side, qty=qty, entry_price=fill, entry_ts=ts,
            decision_ts=decision_ts, stop=stop, stop_initial=stop, take_profit=tp, leverage=lev,
            margin=qty * fill / lev, liq_price=liquidation_price(side, fill, lev, tier.mmr, fee),
            entry_fee=entry_fee,
            planned_loss=(qty * fill / lev + 2 * qty * fill * fee) if R.sizing == "margin" and R.margin_stop == "none"
            else qty * (abs(fill - stop) + fill * cost + stop * cost),
            tag=d.tag,
        )

    def _funding_at(self, r: int, t: int, m: int) -> None:
        f_ts = self.f_ts[r]
        i = int(np.searchsorted(f_ts, t, "left"))
        while i < len(f_ts) and f_ts[i] == t:
            p = self.pos.get(r)
            if p is None:
                return
            o = float(self.panel.open[r, m])
            if not np.isnan(o):
                pay = p.side * p.qty * o * float(self.f_rate[r][i])
                self.cash -= pay
                p.funding += pay
            i += 1

    def _upl_at(self, px: np.ndarray) -> float:
        tot = 0.0
        for r, p in self.pos.items():
            v = px[r]
            if not np.isnan(v):
                tot += p.side * p.qty * (float(v) - p.entry_price)
        return tot

    def _scan_bar(self, t_open: int, t_close: int, ms: int, me: int) -> None:
        # выплаты финансирования строго внутри свечи делят её на сегменты
        cuts = set()
        for r in self.pos:
            f = self.f_ts[r]
            lo, hi = np.searchsorted(f, t_open, "right"), np.searchsorted(f, t_close, "left")
            cuts.update(int(x) for x in f[lo:hi])
        a = ms
        for t in sorted(cuts):
            b = ms + int(np.searchsorted(self.panel.ts[ms:me], t, "left"))
            if b > a and self.pos:
                self._scan(a, b)
            if b < me and int(self.panel.ts[b]) == t:
                for r in list(self.pos):
                    self._funding_at(r, t, b)
            a = max(a, b)
        if a < me and self.pos:
            self._scan(a, me)

    def _scan(self, a: int, b: int) -> None:
        """Ищет первые срабатывания выходов в строках [a, b) по всем позициям."""
        P = self.panel
        while self.pos and a < b:
            floor = self.guard.drawdown_floor()
            adverse = {}
            for r, p in self.pos.items():
                ext = P.low[r, a:b] if p.side == LONG else P.high[r, a:b]
                adverse[r] = np.nan_to_num(p.side * p.qty * (ext - p.entry_price), nan=0.0)
            total_adv = sum(adverse.values()) if len(adverse) > 1 else None
            first = {}
            for r, p in self.pos.items():
                tick = float(self.inst[r].tick_size)
                room = self.cash - floor
                if total_adv is not None:
                    room = room + (total_adv - adverse[r])
                kill = p.entry_price - p.side * room / p.qty
                lo, hi = P.low[r, a:b], P.high[r, a:b]
                if p.side == LONG:
                    stop_hit, kill_hit, liq_hit = lo <= p.stop, lo <= kill, lo <= p.liq_price
                    tp_hit = hi >= p.take_profit + tick if p.take_profit else None
                else:
                    stop_hit, kill_hit, liq_hit = hi >= p.stop, hi >= kill, hi >= p.liq_price
                    tp_hit = lo <= p.take_profit - tick if p.take_profit else None
                hit = stop_hit | kill_hit | liq_hit
                if tp_hit is not None:
                    hit = hit | tp_hit
                idx = np.flatnonzero(hit)
                if idx.size:
                    i = int(idx[0])
                    first[r] = (i, bool(stop_hit[i]), bool(kill_hit[i]) if np.ndim(kill_hit) else bool(kill_hit),
                                bool(liq_hit[i]), bool(tp_hit[i]) if tp_hit is not None else False,
                                float(kill[i]) if np.ndim(kill) else float(kill))
            if not first:
                return
            i0 = min(v[0] for v in first.values())
            for r in sorted(first):
                if first[r][0] == i0 and r in self.pos:
                    self._exit_at(r, a + i0, *first[r][1:])
            a = a + i0

    def _exit_at(self, r: int, m: int, stop_hit: bool, kill_hit: bool, liq_hit: bool, tp_hit: bool,
                 kill: float) -> None:
        P, p, c = self.panel, self.pos[r], self.cfg.costs
        o = float(P.open[r, m])
        tick = float(self.inst[r].tick_size)
        if tp_hit:
            gapped = o >= p.take_profit + tick if p.side == LONG else o <= p.take_profit - tick
            if gapped or not (stop_hit or kill_hit or liq_hit):
                self._close(r, m, p.take_profit, "take_profit", c.maker_fee)
                return
        levels = [(p.stop, "stop")] if stop_hit else []
        if kill_hit:
            levels.append((kill, "max_drawdown"))
        if levels:
            lvl, reason = max(levels) if p.side == LONG else min(levels)
            if p.side == LONG:
                base = min(o, lvl)
                fill = base - c.stop_penetration * max(0.0, base - float(P.low[r, m]))
            else:
                base = max(o, lvl)
                fill = base + c.stop_penetration * max(0.0, float(P.high[r, m]) - base)
            fill *= 1 - p.side * c.slippage
            beyond_liq = fill <= p.liq_price if p.side == LONG else fill >= p.liq_price
            if beyond_liq and self.cfg.stop_beats_liq and (o > p.liq_price if p.side == LONG else o < p.liq_price):
                bankrupt = p.entry_price * (1 - p.side / p.leverage)
                fill = max(fill, bankrupt) if p.side == LONG else min(fill, bankrupt)
                beyond_liq = False
            if not beyond_liq:
                self._close(r, m, fill, reason, c.taker_fee)
                if reason == "max_drawdown" and not self.guard.state.halted:
                    self.events.append(self.guard.halt(int(P.ts[m]), "порог максимальной просадки внутри свечи",
                                                       kind="max_drawdown"))
                return
        bankruptcy = p.entry_price * (1 - p.side / p.leverage)
        self._close(r, m, bankruptcy, "liquidation", 0.0)

    def _close_market(self, r: int, m: int, reason: str, at_close: bool = False) -> None:
        P = self.panel
        px = float(P.close[r, m] if at_close else P.open[r, m])
        if np.isnan(px):   # нет свечи — закрываем по последней известной цене
            px = float(self.last_px[r])
        self._close(r, m, px * (1 - self.pos[r].side * self.cfg.costs.slippage), reason, self.cfg.costs.taker_fee)

    def _close(self, r: int, m: int, price: float, reason: str, fee_rate: float) -> None:
        P = self.panel
        p = self.pos.pop(r)
        ts = int(P.ts[m])
        gross = p.side * p.qty * (price - p.entry_price)
        exit_fee = p.qty * price * fee_rate
        self.cash += gross - exit_fee
        net = gross - p.entry_fee - exit_fee - p.funding
        self.trades.append({
            "symbol": p.symbol, "decision_ts": p.decision_ts, "entry_ts": p.entry_ts, "exit_ts": ts,
            "side": p.side, "qty": p.qty, "entry_price": p.entry_price, "exit_price": price,
            "stop_initial": p.stop_initial, "stop_final": p.stop, "take_profit": p.take_profit,
            "leverage": p.leverage, "margin": p.margin, "liq_price": p.liq_price,
            "entry_fee": p.entry_fee, "exit_fee": exit_fee, "funding": p.funding,
            "gross_pnl": gross, "net_pnl": net, "planned_loss": p.planned_loss,
            "r_multiple": net / p.planned_loss if p.planned_loss > 0 else np.nan,
            "exit_reason": reason, "bars_held": p.bars_held, "tag": p.tag,
            "equity_after": self.cash,
        })
        self.events += self.guard.on_trade_closed(ts, net)
        px_now = self.last_px.copy()
        cl = P.close[:self.n_trade, m]
        ok = ~np.isnan(cl)
        px_now[ok] = cl[ok]
        self.events += self.guard.update_equity(ts, self.cash + self._upl_at(px_now))

    # -------------------------------------------------------------- result
    def _result(self, bars: PanelBars, eq_rows) -> PortfolioResult:
        trades = pd.DataFrame(self.trades)
        equity = pd.DataFrame(eq_rows, columns=["ts", "equity", "cash", "position"])
        guard = getattr(self, "guard", None)
        return PortfolioResult(
            trades=trades, equity=equity, events=self.events, skips=self.skips, decisions=self.decisions,
            bars=bars, final_equity=self.cash, halted=bool(guard and guard.state.halted),
            risk_state=copy.deepcopy(guard.state) if guard else None,
            open_positions=[{"symbol": p.symbol, "side": p.side, "qty": p.qty, "entry_price": p.entry_price,
                             "decision_ts": p.decision_ts, "stop": p.stop} for p in getattr(self, "pos", {}).values()],
            params={"strategy": self.strategy.name, "timeframe": self.strategy.timeframe,
                    **self.strategy.params(), "costs": self.cfg.costs.model_dump()},
        )


class SingleSymbol(PortfolioStrategy):
    """Обёртка: стратегия одной монеты (bot/strategy) как портфельная — для проверки движка."""

    def __init__(self, strategy, row: int = 0):
        self.s, self.row = strategy, row
        self.name, self.timeframe = strategy.name, strategy.timeframe

    def params(self) -> dict:
        return self.s.params()

    def prepare(self, bars: PanelBars) -> None:
        self.s.prepare(bars.row_bars(self.row))
        self.warmup_bars = self.s.warmup_bars

    def on_bar(self, k: int, positions):
        return [(self.row, self.s.on_bar(k, positions.get(self.row)))]
