"""Боевой движок: стратегия корзины на бирже (demo или live).

Решения принимаются тем же кодом, что в бэктесте:
  данные → Panel (bot/data/panel.py) → стратегия (bot/runtime.py::make_strategy) →
  планирование свечи (bot/risk/planner.py: ограничители, места, лимиты риска, объём) →
  исполнение на бирже.

Надёжность:
  - каждый ордер сначала записывается в SQLite с уникальным orderLinkId (свеча + монета + цель),
    потом отправляется; повторная отправка того же orderLinkId биржей отклоняется — двойного
    входа не бывает; при обрыве связи ордер проверяется по orderLinkId;
  - вход и биржевой стоп — одним запросом; позиция без стопа на бирже сразу получает стоп,
    а если не получается — закрывается;
  - сверка с биржей при старте и регулярно: стоп-ауты и ликвидации записываются как сделки,
    расхождения (чужие позиции, другой объём) блокируют новые входы;
  - устаревшие данные (WebSocket молчит, свечи неполные, API недоступно) блокируют новые входы,
    но не выходы и не подтяжку стопов;
  - просадка от пика ≥ max_drawdown → все позиции закрываются, торговля остановлена до ручного
    перезапуска; файл STOP — аварийная остановка (закрыть всё, отменить ордера, выключиться).
"""
from __future__ import annotations

import json
import logging
import math
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bot.config import PROJECT_ROOT, Config
from bot.data.bars import TF_MINUTES
from bot.data.panel import aggregate_panel
from bot.engine.data import LiveData
from bot.engine.state import DbOrder, DbPosition, StateDB
from bot.exchange.client import (DUPLICATE_LINK_ID, REDUCE_ONLY_NOTHING, BybitClient, ExchangeError, NetworkError,
                                 OrderInfo)
from bot.market import Instrument
from bot.risk.guards import GuardEvent, RiskGuard, RiskState
from bot.risk.planner import Held, Planner
from bot.risk.portfolio import PortfolioRisk
from bot.risk.sizing import SizingParams, max_leverage_for_stop
from bot.runtime import make_strategy, warmup_days
from bot.strategy.base import PositionView

log = logging.getLogger(__name__)
DAY_MS = 86_400_000


class NullNotifier:
    def send(self, text: str, important: bool = False) -> None:
        log.info("[уведомление] %s", text)


class NullStream:
    """WebSocket не используется (тесты, ws_required=false)."""

    def start(self) -> None: ...

    def public_age_s(self, now_ms: int) -> float | None:
        return 0.0

    def pop_dirty(self) -> bool:
        return False

    def restart(self) -> None: ...

    def stop(self) -> None: ...


@dataclass
class StepResult:
    processed_bar: int | None = None
    killed: bool = False
    halted: bool = False


def sym_code(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


class Engine:
    def __init__(self, cfg: Config, client: BybitClient, db: StateDB, clock, data: LiveData, mode: str = "demo",
                 notifier=None, stream=None, base_dir: Path = PROJECT_ROOT):
        if mode not in ("demo", "live", "sim"):
            raise ValueError("mode: demo | live | sim")
        if mode == "live" and not cfg.live.enabled:
            raise PermissionError("Реальная торговля выключена: live.enabled = false в config/bot.yaml")
        self.cfg, self.client, self.db, self.clock, self.data = cfg, client, db, clock, data
        self.mode = mode
        self.notifier = notifier or NullNotifier()
        self.stream = stream or NullStream()
        self.base_dir = Path(base_dir)
        self.E = cfg.engine
        self.tf_ms = TF_MINUTES[cfg.strategy.timeframe] * 60_000
        self.rules = cfg.portfolio.rules()
        c = cfg.costs
        self.sp = SizingParams(risk_per_trade=cfg.risk.risk_per_trade, taker_fee=c.taker_fee, slippage=c.slippage,
                               liq_buffer=cfg.risk.liq_distance_vs_stop_min, max_leverage=cfg.risk.max_leverage)
        self.prefix = "hb" + {"demo": "d", "live": "l", "sim": "s"}[mode]
        self.tradable = list(cfg.strategy.tradable)
        self.guard: RiskGuard | None = None
        self.inst: dict[str, Instrument] = {}
        self.last_reconcile = -math.inf
        self.last_snapshot = -math.inf
        self.last_instruments = -math.inf
        self.started = False
        self.hooks: list = []          # вызываются в конце каждого шага (отчёты, теневые варианты)

    # =============================================================== служебное
    def now(self) -> int:
        return int(self.clock.now_ms())

    def event(self, level: str, kind: str, msg: str, notify: bool = False) -> None:
        self.db.event(self.now(), level, kind, msg)
        (log.warning if level in ("warn", "error") else log.info)("%s: %s", kind, msg)
        if notify:
            self.notifier.send(msg, important=level in ("warn", "error"))

    def _guard_events(self, evs: list[GuardEvent]) -> None:
        for e in evs:
            if e.kind == "day_start":
                continue
            level = "error" if e.kind in ("max_drawdown", "halt") else "warn"
            self.event(level, "guard", f"Ограничитель: {e.detail}", notify=e.kind not in ("blocker_off",))

    def _save_guard(self) -> None:
        if self.guard:
            self.db.set("risk_state", self.guard.state.to_dict())

    def stop_file(self) -> Path:
        return self.base_dir / self.E.stop_file

    def _link(self, t_ms: int, symbol: str, purpose: str) -> str:
        return f"{self.prefix}{t_ms // 1000:x}{sym_code(symbol)}{purpose}"[:36]

    def _load_instruments(self) -> None:
        for s in self.tradable:
            self.inst[s] = Instrument.from_files(self.data.dir, s)

    # ================================================================ капитал
    def offset(self) -> float:
        return float(self.db.get("equity_offset", 0.0))

    def equity(self) -> tuple[float, float]:
        """(капитал бота, баланс кошелька). Капитал бота = капитал счёта − смещение при первом запуске:
        на демо-счёте с большим балансом бот работает так, будто у него starting_equity_usdt."""
        w = self.client.wallet()
        return w.equity - self.offset(), w.equity

    # ================================================================== старт
    def start(self) -> bool:
        now = self.now()
        if self.db.get("halted"):
            h = self.db.get("halted")
            self.event("warn", "start", f"Бот остановлен ({h.get('reason')}), торговля не ведётся. "
                                        "Для запуска после разбора — windows\\resume.bat", notify=True)
            return False
        if self.stop_file().exists():
            self.kill("файл STOP при запуске")
            return False
        self.data.refresh_instruments(now)
        self.last_instruments = now
        self._load_instruments()
        if self.db.get("equity_offset") is None:
            w = self.client.wallet()
            off = max(0.0, w.equity - self.cfg.risk.starting_equity_usdt)
            self.db.set("equity_offset", off)
            self.db.set("equity_start", w.equity - off)
            self.db.set("started_ms", now)
            self.db.set("mode", self.mode)
            self.event("info", "start", f"Первый запуск ({self.mode}): баланс счёта {w.equity:.2f} USDT, капитал бота "
                                        f"{w.equity - off:.2f} USDT", notify=True)
        if self.db.get("anchor_ms") is None:
            self.db.set("anchor_ms", (now // DAY_MS) * DAY_MS - warmup_days(self.cfg) * DAY_MS)
        eq, _ = self.equity()
        st = self.db.get("risk_state")
        self.guard = RiskGuard(self.cfg.risk, eq, now, RiskState.from_dict(st) if st else None)
        for k in ("ws", "data", "mismatch", "api", "account"):
            self.guard.state.blockers.pop(k, None)
        try:
            changed = self.client.ensure_isolated_one_way()
            if changed:
                self.event("info", "account", "Настройки счёта: " + ", ".join(changed))
            mm = self.client.account_info().get("marginMode")
            if mm != "ISOLATED_MARGIN":
                self._blocker("account", f"режим маржи {mm}, нужен ISOLATED_MARGIN")
        except ExchangeError as e:
            self._blocker("account", f"не удалось включить изолированную маржу / одну позицию: {e}")
        if self.mode == "live" and self.db.get("exec_baseline") is None:
            self.db.set("exec_baseline", self._demo_execution_baseline())
        self.stream.start()
        self.reconcile()
        self.data.update(int(self.db.get("anchor_ms")), now)
        if self.db.get("last_bar_close") is None:
            self.db.set("last_bar_close", (now // self.tf_ms) * self.tf_ms)
        self._save_guard()
        self.started = True
        self.event("info", "start", f"Движок запущен ({self.mode}), капитал {eq:.2f} USDT, позиций "
                                    f"{len(self.db.positions())}", notify=True)
        return True

    def _op_bar(self, t_close: int, reason: str) -> None:
        """Свеча, на которой движок по операционной причине вёл себя не как бэктест (для сравнения)."""
        ops = self.db.get("op_bars", [])
        if not ops or ops[-1][0] != t_close:
            ops.append([int(t_close), reason])
            self.db.set("op_bars", ops[-2000:])

    def _blocker(self, key: str, reason: str | None) -> None:
        ev = self.guard.set_blocker(self.now(), key, reason) if self.guard else None
        if ev:
            self.event("warn" if reason else "info", "blocker",
                       f"{'Новые входы заблокированы' if reason else 'Блокировка снята'}: {key}"
                       f"{': ' + reason if reason else ''}", notify=True)

    # =================================================================== шаг
    def step(self) -> StepResult:
        res = StepResult()
        now = self.now()
        if self.stop_file().exists():
            self.kill("файл STOP")
            res.killed = True
            return res
        if self.db.get("halted"):
            res.halted = True
            if now - self.last_reconcile >= self.E.reconcile_every_s * 1000:
                self._ensure_flat("max_drawdown")
            return res
        # WebSocket: свежесть данных
        age = self.stream.public_age_s(now)
        if self.E.ws_required:
            if age is None or age > self.E.ws_stale_s:
                self._blocker("ws", f"нет живых данных WebSocket {age if age is None else round(age)} с")
                if age is not None and age > 3 * self.E.ws_stale_s:
                    self.stream.restart()
            else:
                self._blocker("ws", None)
        # сверка
        if self.stream.pop_dirty() or now - self.last_reconcile >= self.E.reconcile_every_s * 1000:
            self.reconcile()
        # капитал и ограничители
        try:
            eq, wallet = self.equity()
        except NetworkError as e:
            self._blocker("api", str(e))
            return res
        # Пик и дневной лимит обновляются на закрытии свечей и сделок (как в бэктесте); порог
        # остановки по просадке проверяется на каждом шаге (в бэктесте — внутри свечи).
        if eq <= self.guard.drawdown_floor() and not self.guard.state.halted:
            self._guard_events([self.guard.halt(now, f"порог просадки {self.cfg.risk.max_drawdown:.0%} от пика "
                                                     f"(капитал {eq:.2f}, пик {self.guard.state.peak_equity:.2f})",
                                                kind="max_drawdown")])
        if self.guard.state.halted:
            self._halt(self.guard.state.halt_reason)
            res.halted = True
            return res
        if now - self.last_snapshot >= self.E.snapshot_every_s * 1000:
            self.db.snapshot(now, eq, wallet, len(self.db.positions()))
            self.last_snapshot = now
        if now - self.last_instruments >= DAY_MS:
            try:
                self.data.refresh_instruments(now)
                self._load_instruments()
                self.last_instruments = now
            except NetworkError:
                pass
        # свеча стратегии
        bar = ((now - int(self.E.bar_delay_s * 1000)) // self.tf_ms) * self.tf_ms
        if bar > int(self.db.get("last_bar_close", 0)):
            done = self.process_bar(bar)
            if done:
                res.processed_bar = bar
        self._save_guard()
        for h in self.hooks:
            try:
                h(self)
            except Exception as e:  # отчёты не должны останавливать торговлю
                self.event("error", "hook", f"Ошибка отчёта: {e}\n{traceback.format_exc()[-800:]}")
        return res

    # ========================================================== свеча 4h
    def process_bar(self, t_close: int) -> bool:
        """True — свеча обработана (или пропущена окончательно); False — ждём данные."""
        now = self.now()
        last = int(self.db.get("last_bar_close", 0))
        if t_close - last > self.tf_ms and last > 0:
            self.event("warn", "missed", f"Пропущено свечей 4h: {(t_close - last) // self.tf_ms - 1} "
                                         "(бот был выключен или без связи); по ним входов нет", notify=True)
            for b in range(last + self.tf_ms, t_close, self.tf_ms):
                self._op_bar(b, "бот был выключен или без связи")
        lag = (now - t_close) / 1000
        try:
            self.data.update(self.db.get("anchor_ms"), t_close)
        except NetworkError as e:
            self._blocker("api", f"нет данных: {e}")
            if lag < self.E.data_wait_s:
                return False
        missing = self.data.missing(t_close)
        if missing and lag < self.E.data_wait_s:
            return False
        entries_allowed = lag <= self.E.max_bar_lag_s and not missing
        if missing:
            self._op_bar(t_close, "неполные данные")
            self.event("warn", "data", f"Неполные данные на {_utc(t_close)}: {', '.join(missing)} — новые входы "
                                       "по этой свече не открываются", notify=True)
        if lag > self.E.max_bar_lag_s:
            self._op_bar(t_close, "свеча обработана поздно")
            self.event("warn", "late", f"Свеча {_utc(t_close)} обрабатывается с опозданием {lag:.0f} с — "
                                       "только выходы и стопы")
        out = self.plan_bar(t_close, entries_allowed)
        if out is None:
            self.db.set("last_bar_close", t_close)
            return True
        plan, syms, insts, last_px = out
        for t, s, d in plan.decisions:
            self.db.event(now, "info", "decision", f"{_utc(t)} {s} {d}")
        for t, s, why in plan.skips:
            self.db.event(now, "info", "skip", f"{_utc(t)} {s}: {why}")
            if why.startswith("блокировка") or why.startswith("новые входы"):
                self._op_bar(t_close, why)
        # исполнение: выходы → стопы → входы (как в бэктесте)
        for r, (_, reason) in plan.exits.items():
            self._exit(syms[r], t_close, reason, float(last_px[r]))
        for r, stop in plan.stops.items():
            self._move_stop(syms[r], stop, t_close)
        for r, _, d, s in plan.entries:
            self._enter(syms[r], t_close, d, s, insts[r])
        self.db.set("last_bar_close", t_close)
        self.reconcile()
        if self.mode == "live" or self.db.get("exec_baseline") is not None:
            self.check_execution()
        return True

    # ===================================================== исполнение против демо
    def _demo_execution_baseline(self) -> dict:
        from bot.engine.control import paths
        from bot.report.execution import baseline
        db_path, data_dir = paths(self.cfg, "demo")
        trades = []
        if db_path.exists():
            demo = StateDB(db_path)
            trades = demo.trades()
            demo.close()
        b = baseline(trades, data_dir, self.cfg.costs)
        self.event("info", "execution", f"База исполнения ({b['source']}, сделок {b['n']}): проскальзывание "
                                        f"{b['slip'] * 100:.3f}%, стопы {b['stop'] * 100:+.3f}%")
        return b

    def check_execution(self) -> None:
        """Реальный счёт: исполнение заметно хуже демо → новые входы блокируются до ручного снятия."""
        from bot.report.execution import execution_alarm
        base = self.db.get("exec_baseline")
        if not base or "execution" in self.guard.state.blockers:
            return
        L = self.cfg.live
        why = execution_alarm(self.db.trades(), self.data.dir, self.cfg.costs, base, L.exec_window, L.exec_min_n,
                              L.max_slip_excess, L.max_stop_excess)
        if why:
            self._blocker("execution", f"исполнение хуже демо: {why}. Снять — windows\\resume.bat live")
            self._save_guard()

    def plan_bar(self, t_close: int, entries_allowed: bool = True, update_guard: bool = True):
        """Решения стратегии и план исполнения на закрытии свечи (без ордеров).
        None — свечи нет в данных, идёт разогрев или сработала остановка."""
        panel = self.data.panel(int(self.db.get("anchor_ms")), t_close)
        bars = aggregate_panel(panel, self.cfg.strategy.timeframe)
        ks = np.flatnonzero(bars.close_ts == t_close)
        if not len(ks):
            self.event("error", "data", f"Нет свечи {_utc(t_close)} в данных — свеча пропущена", notify=True)
            return None
        k = int(ks[0])
        strat = make_strategy(self.cfg)
        strat.prepare(bars)
        prisk = PortfolioRisk(bars, self.rules)
        n = bars.n_trade
        syms = bars.symbols[:n]
        insts = [panel.sym[s].inst for s in syms]
        closes = bars.close[:n, :k + 1]
        last_px = np.array([row[~np.isnan(row)][-1] if (~np.isnan(row)).any() else np.nan for row in closes])
        dbp = self.db.positions()
        row = {s: i for i, s in enumerate(syms)}
        held, views = {}, {}
        for s, p in dbp.items():
            if s not in row:
                continue
            r = row[s]
            held[r] = Held(p.side, p.qty, p.stop, p.margin)
            bars_held = int((t_close - p.decision_ts) // self.tf_ms)
            if bars_held >= 1:     # позиция, открытая по этой же свече (повтор после перезапуска), — не для стратегии
                views[r] = PositionView(p.side, p.entry_price, p.qty, p.stop, None, p.decision_ts, bars_held)
        eq, _ = self.equity()
        upl = sum(p.side * p.qty * (float(last_px[row[s]]) - p.entry_price) for s, p in dbp.items()
                  if s in row and not np.isnan(last_px[row[s]]))
        cash = eq - upl
        if update_guard:
            self._guard_events(self.guard.update_equity(t_close, eq))
            if self.guard.state.halted:
                self._halt(self.guard.state.halt_reason)
                return None
        if k < strat.warmup_bars:
            return None
        ds = strat.on_bar(k, views) or []
        planner = Planner(syms, insts, self.sp, self.rules, prisk, self.guard)
        plan = planner.plan(ds, k, t_close, held, cash, last_px, eq, entries_allowed=entries_allowed)
        return plan, syms, insts, last_px

    # ============================================================== ордера
    def _confirm(self, link_id: str, symbol: str | None = None, attempts: int = 5) -> OrderInfo | None:
        o = None
        for i in range(attempts):
            try:
                o = self.client.order_by_link_id(link_id, symbol)
            except NetworkError:
                o = None
            if o is not None and o.done:
                return o
            if i + 1 < attempts:
                self.clock.sleep(1.0)
        return o

    def _send(self, o: DbOrder, **kw) -> OrderInfo | None:
        """Отправка с записью намерения до запроса. None — ордер не исполнен (причина в БД)."""
        self.db.add_order(o)
        try:
            oid = self.client.place_market(o.symbol, o.side, kw.pop("qty_str"), o.link_id, **kw)
            self.db.update_order(o.link_id, status="sent", order_id=oid)
        except ExchangeError as e:
            if e.code == DUPLICATE_LINK_ID:
                self.db.update_order(o.link_id, status="sent", error="дубликат orderLinkId — ордер уже был")
            else:
                self.db.update_order(o.link_id, status="rejected", error=str(e))
                return None
        except NetworkError as e:
            self.db.update_order(o.link_id, status="unknown", error=str(e))
        ex = self._confirm(o.link_id, o.symbol)
        if ex is None:
            self.event("warn", "order", f"{o.symbol}: ордер {o.link_id} не подтверждён биржей — проверка при сверке")
            return None
        if ex.filled_qty <= 0:
            self.db.update_order(o.link_id, status="rejected", order_id=ex.order_id, error=ex.reject_reason or ex.status)
            return None
        self.db.update_order(o.link_id, status="filled", order_id=ex.order_id, filled_qty=ex.filled_qty,
                             avg_price=ex.avg_price, fee=ex.fee)
        return ex

    def _enter(self, symbol: str, t_close: int, d, s, inst: Instrument) -> None:
        link = self._link(t_close, symbol, "e")
        if self.db.order(link) is not None:
            return                                        # эта свеча уже исполнялась (перезапуск)
        try:
            self.client.set_leverage(symbol, s.leverage)
        except (ExchangeError, NetworkError) as e:
            self.event("warn", "skip", f"{symbol}: не удалось выставить плечо {s.leverage:g} ({e}) — вход пропущен")
            return
        o = DbOrder(link_id=link, symbol=symbol, purpose="entry", side=d.side, qty=s.qty, ref_price=s.entry_ref,
                    stop=s.stop, decision_ts=t_close, created_ms=self.now(), status="pending",
                    extra=json.dumps({"leverage": s.leverage, "planned_loss": s.planned_loss}))
        ex = self._send(o, qty_str=inst.fmt_qty(s.qty), stop_loss=inst.fmt_price(s.stop),
                        sl_trigger_by=self.cfg.risk.sl_trigger_by)
        if ex is None:
            dbo = self.db.order(link)
            if dbo and dbo.status == "rejected":
                self.event("info", "skip", f"{symbol}: вход отклонён биржей ({dbo.error})")
            return
        self._apply_entry_fill(self.db.order(link), ex)

    def _apply_entry_fill(self, o: DbOrder, ex: OrderInfo) -> None:
        inst = self.inst[o.symbol]
        extra = json.loads(o.extra or "{}")
        lev = float(extra.get("leverage", 1.0))
        fill, qty = ex.avg_price, ex.filled_qty
        fee_rate, liq_buf = self.cfg.costs.taker_fee, self.cfg.risk.liq_distance_vs_stop_min
        tier = inst.tier_for(qty * fill)
        need = inst.floor_leverage(min(lev, tier.max_leverage,
                                       max_leverage_for_stop(abs(fill - o.stop) / fill, tier.mmr, fee_rate, liq_buf)))
        if need < lev:
            try:
                self.client.set_leverage(o.symbol, need)
                lev = need
            except (ExchangeError, NetworkError) as e:
                self.event("warn", "leverage", f"{o.symbol}: не удалось снизить плечо до {need:g}: {e}")
        cost = self.cfg.costs.taker_fee + self.cfg.costs.slippage
        planned = qty * (abs(fill - o.stop) + fill * cost + o.stop * cost)
        self.db.put_position(DbPosition(
            symbol=o.symbol, side=o.side, qty=qty, entry_price=fill, entry_ts=self.now(), decision_ts=o.decision_ts,
            stop=o.stop, stop_initial=o.stop, leverage=lev, margin=qty * fill / lev, planned_loss=planned,
            ref_price=o.ref_price or fill, entry_fee=ex.fee, link_id=o.link_id))
        side = "ЛОНГ" if o.side > 0 else "ШОРТ"
        self.event("info", "trade_open", f"Открыта {side} {o.symbol}: {qty:g} по {fill:.6g}, стоп {o.stop:.6g}, "
                                         f"риск {planned:.2f} USDT, плечо {lev:g}", notify=True)
        self._verify_stop(o.symbol)

    def _verify_stop(self, symbol: str) -> None:
        """Позиция без биржевого стопа недопустима: выставить, а если нельзя — закрыть."""
        dp = self.db.positions().get(symbol)
        if dp is None:
            return
        try:
            ep = {p.symbol: p for p in self.client.positions()}.get(symbol)
        except NetworkError:
            return
        if ep is None:
            return
        tick = float(self.inst[symbol].tick_size)
        if ep.stop_loss is not None and abs(ep.stop_loss - dp.stop) <= tick / 2:
            return
        try:
            self.client.set_stop(symbol, self.inst[symbol].fmt_price(dp.stop), self.cfg.risk.sl_trigger_by)
            self.event("warn", "stop", f"{symbol}: стоп на бирже {'отсутствовал' if ep.stop_loss is None else 'другой'}"
                                       f" — выставлен {dp.stop:.6g}", notify=True)
        except (ExchangeError, NetworkError) as e:
            self.event("error", "stop", f"{symbol}: не удалось выставить стоп ({e}) — позиция закрывается", notify=True)
            self._close_now(symbol, "no_stop")

    def _move_stop(self, symbol: str, stop: float, t_close: int) -> None:
        dp = self.db.positions().get(symbol)
        if dp is None:
            return
        try:
            self.client.set_stop(symbol, self.inst[symbol].fmt_price(stop), self.cfg.risk.sl_trigger_by)
            self.db.update_position(symbol, stop=stop)
            self.db.event(self.now(), "info", "stop_move", f"{symbol}: стоп {dp.stop:.6g} → {stop:.6g}")
        except ExchangeError as e:
            if "StopLoss" in e.message or "base_price" in e.message:
                # цена уже за новым стопом — как в бэктесте: выход по рынку
                self.event("info", "stop_move", f"{symbol}: цена уже прошла новый стоп {stop:.6g} — выход по рынку")
                self._exit(symbol, t_close, "stop", None, purpose="s")
            else:
                self.event("warn", "stop_move", f"{symbol}: стоп не подтянут: {e}")
        except NetworkError as e:
            self.event("warn", "stop_move", f"{symbol}: стоп не подтверждён ({e}) — проверка при сверке")
            try:
                ep = {p.symbol: p for p in self.client.positions()}.get(symbol)
                if ep and ep.stop_loss and abs(ep.stop_loss - stop) <= float(self.inst[symbol].tick_size) / 2:
                    self.db.update_position(symbol, stop=stop)
            except NetworkError:
                pass

    def _exit(self, symbol: str, t_ms: int, reason: str, ref: float | None, purpose: str = "x") -> None:
        dp = self.db.positions().get(symbol)
        if dp is None:
            return
        link = self._link(t_ms, symbol, purpose)
        if self.db.order(link) is not None:
            o = self.db.order(link)
            if o.status != "filled":
                return
        else:
            o = DbOrder(link_id=link, symbol=symbol, purpose="exit", side=-dp.side, qty=dp.qty, ref_price=ref,
                        stop=None, decision_ts=t_ms, created_ms=self.now(), status="pending",
                        extra=json.dumps({"reason": reason}))
            ex = self._send(o, qty_str=self.inst[symbol].fmt_qty(dp.qty), reduce_only=True)
            if ex is None:
                dbo = self.db.order(link)
                if dbo and dbo.status == "rejected" and dbo.error and any(str(c) in dbo.error for c in REDUCE_ONLY_NOTHING):
                    self.event("info", "exit", f"{symbol}: позиции уже нет на бирже — запишу по сверке")
                return
            self._apply_exit_fill(self.db.order(link), ex)

    def _apply_exit_fill(self, o: DbOrder, ex: OrderInfo) -> None:
        dp = self.db.positions().get(o.symbol)
        if dp is None:
            return
        reason = json.loads(o.extra or "{}").get("reason", "signal")
        self._record_trade(dp, exit_price=ex.avg_price, exit_ts=self.now(), exit_fee=ex.fee, reason=reason,
                           exit_ref=o.ref_price, exit_link=o.link_id, qty=ex.filled_qty)

    def _record_trade(self, dp: DbPosition, exit_price: float, exit_ts: int, exit_fee: float, reason: str,
                      exit_ref: float | None, exit_link: str, qty: float | None = None) -> None:
        qty = dp.qty if qty is None else qty
        try:
            funding = float(self.client.funding_paid(dp.symbol, dp.entry_ts, exit_ts) or 0.0)
        except NetworkError:
            funding = 0.0
        gross = dp.side * qty * (exit_price - dp.entry_price)
        net = gross - dp.entry_fee - exit_fee - funding
        self.db.add_trade({
            "symbol": dp.symbol, "side": dp.side, "qty": qty, "decision_ts": dp.decision_ts, "entry_ts": dp.entry_ts,
            "entry_price": dp.entry_price, "ref_price": dp.ref_price, "exit_ts": exit_ts, "exit_price": exit_price,
            "exit_ref": exit_ref, "exit_reason": reason, "stop_initial": dp.stop_initial, "stop_final": dp.stop,
            "planned_loss": dp.planned_loss, "entry_fee": dp.entry_fee, "exit_fee": exit_fee, "funding": funding,
            "gross_pnl": gross, "net_pnl": net, "r_multiple": net / dp.planned_loss if dp.planned_loss > 0 else None,
            "leverage": dp.leverage, "entry_link": dp.link_id, "exit_link": exit_link})
        if qty < dp.qty - 1e-12:
            self.db.update_position(dp.symbol, qty=dp.qty - qty)
        else:
            self.db.remove_position(dp.symbol)
        self._guard_events(self.guard.on_trade_closed(exit_ts, net))
        try:
            self._guard_events(self.guard.update_equity(exit_ts, self.equity()[0]))
        except NetworkError:
            pass
        why = {"stop": "стоп", "channel_exit": "выход по каналу", "liquidation": "ЛИКВИДАЦИЯ",
               "max_drawdown": "остановка по просадке", "kill": "аварийная остановка", "external": "закрыта вне бота",
               "no_stop": "нет стопа"}.get(reason, reason)
        self.event("info", "trade_close", f"Закрыта {dp.symbol} ({why}): {exit_price:.6g}, результат {net:+.3f} USDT "
                                          f"({net / dp.planned_loss if dp.planned_loss else 0:+.2f} R)", notify=True)

    def _close_now(self, symbol: str, reason: str) -> None:
        """Немедленное закрытие по рынку (нет стопа, остановка, аварийный выход)."""
        self._exit(symbol, self.now(), reason, None, purpose="k")

    # ================================================================ сверка
    def reconcile(self) -> bool:
        now = self.now()
        self.last_reconcile = now
        try:
            self._resolve_orders()
            ex_pos = {p.symbol: p for p in self.client.positions()}
        except NetworkError as e:
            self._blocker("api", f"сверка невозможна: {e}")
            return False
        self._blocker("api", None)
        issues = []
        dbp = self.db.positions()
        for s, dp in dbp.items():
            ep = ex_pos.get(s)
            if ep is None:
                self._external_close(dp)
                continue
            if ep.side != dp.side or abs(ep.size - dp.qty) > 1e-9 * max(1.0, dp.qty):
                issues.append(f"{s}: на бирже {'лонг' if ep.side > 0 else 'шорт'} {ep.size:g}, у бота "
                              f"{'лонг' if dp.side > 0 else 'шорт'} {dp.qty:g}")
                continue
            tick = float(self.inst[s].tick_size) if s in self.inst else 0.0
            if ep.stop_loss is None or abs(ep.stop_loss - dp.stop) > tick / 2:
                self._verify_stop(s)
        for s, ep in ex_pos.items():
            if s not in dbp:
                issues.append(f"{s}: позиция на бирже ({'лонг' if ep.side > 0 else 'шорт'} {ep.size:g}), которой нет у бота"
                              f"{'' if ep.stop_loss else ', БЕЗ СТОПА'}")
        prev = self.db.get("mismatch_text")
        if issues:
            text = "; ".join(issues)
            self._blocker("mismatch", "расхождение с биржей")
            if text != prev:
                self.event("error", "mismatch", f"Расхождение с биржей: {text}", notify=True)
                self.db.set("mismatch_text", text)
        else:
            self._blocker("mismatch", None)
            if prev:
                self.db.delete("mismatch_text")
        return not issues

    def _resolve_orders(self) -> None:
        for o in self.db.orders(status=("pending", "sent", "unknown")):
            ex = self.client.order_by_link_id(o.link_id, o.symbol)
            if ex is None:
                if self.now() - o.created_ms > 120_000:
                    self.db.update_order(o.link_id, status="failed", error=(o.error or "") + "; на бирже не найден")
                    self.event("warn", "order", f"{o.symbol}: ордер {o.link_id} до биржи не дошёл")
                continue
            if not ex.done:
                continue
            if ex.filled_qty <= 0:
                self.db.update_order(o.link_id, status="rejected", order_id=ex.order_id, error=ex.status)
                continue
            self.db.update_order(o.link_id, status="filled", order_id=ex.order_id, filled_qty=ex.filled_qty,
                                 avg_price=ex.avg_price, fee=ex.fee)
            if o.purpose == "entry" and o.symbol not in self.db.positions():
                self.event("warn", "order", f"{o.symbol}: вход {o.link_id} исполнен, ответ был потерян — позиция "
                                            "восстановлена из истории ордеров", notify=True)
                self._apply_entry_fill(self.db.order(o.link_id), ex)
            elif o.purpose == "exit" and o.symbol in self.db.positions():
                self._apply_exit_fill(self.db.order(o.link_id), ex)

    def _external_close(self, dp: DbPosition) -> None:
        """Позиция есть у бота, но её нет на бирже: стоп, ликвидация или ручное закрытие."""
        now = self.now()
        try:
            exs = [e for e in self.client.executions(dp.entry_ts - 60_000, now, dp.symbol)
                   if e.side == -dp.side and e.ts >= dp.entry_ts and e.link_id != dp.link_id]
        except NetworkError:
            return
        if not exs:
            tries = int(self.db.get(f"close_wait_{dp.symbol}", 0)) + 1
            self.db.set(f"close_wait_{dp.symbol}", tries)
            if tries < 5:
                return
            px = self.client.ticker(dp.symbol).last
            self.event("error", "mismatch", f"{dp.symbol}: позиция исчезла, исполнений не найдено — записана по цене "
                                            f"{px:.6g}", notify=True)
            exs_price, exs_fee, ts, reason = px, 0.0, now, "external"
        else:
            q = sum(e.qty for e in exs)
            exs_price = sum(e.price * e.qty for e in exs) / q
            exs_fee = sum(e.fee for e in exs)
            ts = max(e.ts for e in exs)
            ours = [self.db.order(e.link_id) for e in exs if e.link_id]
            ours = [o for o in ours if o is not None]
            reason = ("liquidation" if any(e.exec_type in ("BustTrade", "AdlTrade") for e in exs) else
                      "stop" if any(e.stop_order_type == "StopLoss" for e in exs) else
                      json.loads(ours[0].extra or "{}").get("reason", "signal") if ours else "external")
        self.db.delete(f"close_wait_{dp.symbol}")
        self._record_trade(dp, exit_price=exs_price, exit_ts=ts, exit_fee=exs_fee, reason=reason,
                           exit_ref=dp.stop if reason == "stop" else None, exit_link="")

    # ============================================================ остановки
    def _ensure_flat(self, reason: str) -> None:
        try:
            pos = self.client.positions()
        except NetworkError:
            return
        for p in pos:
            if p.symbol in self.tradable:
                if p.symbol not in self.db.positions():
                    continue
                self._close_now(p.symbol, reason)
        self.reconcile()

    def _halt(self, reason: str) -> None:
        if not self.db.get("halted"):
            self.db.set("halted", {"reason": reason, "ts": self.now()})
            self.event("error", "halt", f"ОСТАНОВКА: {reason}. Все позиции закрываются, новые не открываются. "
                                        "Перезапуск — только вручную (windows\\resume.bat).", notify=True)
        self._ensure_flat("max_drawdown")
        self._save_guard()

    def kill(self, reason: str) -> None:
        """Аварийная остановка: отменить ордера, закрыть позиции бота, запретить торговлю."""
        self.event("error", "kill", f"АВАРИЙНАЯ ОСТАНОВКА: {reason}", notify=True)
        self.db.set("halted", {"reason": f"аварийная остановка: {reason}", "ts": self.now()})
        try:
            self.client.cancel_all()
        except (ExchangeError, NetworkError) as e:
            self.event("warn", "kill", f"Отмена ордеров: {e}")
        if not self.inst:
            try:
                self._load_instruments()
            except FileNotFoundError:
                pass
        try:
            ex_pos = self.client.positions()
        except NetworkError as e:
            self.event("error", "kill", f"Нет связи с биржей, позиции не закрыты: {e}", notify=True)
            return
        if self.guard is None:
            eq = self.cfg.risk.starting_equity_usdt
            st = self.db.get("risk_state")
            self.guard = RiskGuard(self.cfg.risk, eq, self.now(), RiskState.from_dict(st) if st else None)
        for p in ex_pos:
            if p.symbol not in self.tradable:
                continue
            if p.symbol not in self.db.positions():
                self.db.put_position(DbPosition(p.symbol, p.side, p.size, p.avg_price, self.now(), self.now(),
                                                p.stop_loss or 0.0, p.stop_loss or 0.0, p.leverage, p.position_im,
                                                0.0, p.avg_price, 0.0, "external"))
            if p.symbol not in self.inst:
                self.inst[p.symbol] = Instrument.from_api(self.client.instrument(p.symbol),
                                                          self.client.risk_limits(p.symbol))
            self._close_now(p.symbol, "kill")
        self.reconcile()
        left = [p.symbol for p in self.client.positions() if p.symbol in self.tradable]
        self._save_guard()
        self.event("error" if left else "info", "kill",
                   "Позиции закрыты, ордера отменены, бот выключен." if not left else
                   f"НЕ закрыты: {', '.join(left)} — закройте вручную в приложении Bybit", notify=True)


def _utc(ms: int) -> str:
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
