"""Имитатор биржи Bybit V5 (linear) для тестов движка без доступа к API.

`SimBybit` повторяет методы и формат ответов `pybit.unified_trading.HTTP`, которые использует
`BybitClient`: retCode/retMsg/result, строки вместо чисел, ошибки — исключения pybit
(InvalidRequestError с кодом Bybit, FailedRequestError при «обрыве связи»).

Рынок — 15-минутные свечи (из каталога данных в формате bot/data/store.py или из Panel).
Время задаёт `SimClock`; при каждом запросе имитатор «доигрывает» завершившиеся свечи:
  - финансирование в момент выплаты (до действий движка в эту минуту);
  - стоп-лосс: исполнение хуже цены стопа на stop_penetration пути до экстремума свечи и на
    проскальзывание — ровно как в бэктесте (bot/backtest/portfolio.py::_exit_at);
  - ликвидация в изолированной марже по цене банкротства.
Рыночный ордер исполняется по открытию текущей 15m-свечи с проскальзыванием (как вход в бэктесте).

Сбои для тестов: fail_next[метод] = "before" (запрос не дошёл) или "after" (исполнен, ответ потерян),
reject_next[метод] = (код, сообщение).
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from bot.data import store
from bot.market import Instrument
from bot.risk.sizing import liquidation_price

M15 = 15 * 60_000


class SimClock:
    def __init__(self, now_ms: int):
        self.now = int(now_ms)

    def now_ms(self) -> int:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += int(round(seconds * 1000))

    def advance_to(self, t_ms: int) -> None:
        self.now = max(self.now, int(t_ms))


@dataclass
class SymMarket:
    ts: np.ndarray
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    turnover: np.ndarray
    f_ts: np.ndarray
    f_rate: np.ndarray
    instrument: dict
    tiers: list[dict]
    inst: Instrument = field(init=False)

    def __post_init__(self):
        self.inst = Instrument.from_api(self.instrument, self.tiers)

    def idx(self, t: int) -> int:
        """Индекс свечи, внутри которой момент t (или −1)."""
        i = int(np.searchsorted(self.ts, t, "right")) - 1
        return i if i >= 0 and t < self.ts[i] + M15 else -1


@dataclass
class SimPos:
    symbol: str
    side: int
    size: float
    entry: float
    leverage: float
    stop: float | None = None
    created: int = 0
    realised: float = 0.0


def _err(code: int, msg: str):
    from pybit.exceptions import InvalidRequestError
    return InvalidRequestError(request="sim", message=msg, status_code=code, time="", resp_headers=None)


def _net():
    from pybit.exceptions import FailedRequestError
    return FailedRequestError(request="sim", message="имитация обрыва связи", status_code=0, time="",
                              resp_headers=None)


def _s(x: float) -> str:
    return repr(float(x))


class SimBybit:
    def __init__(self, market: dict[str, SymMarket], clock: SimClock, equity: float = 25.0,
                 taker_fee: float = 0.00055, maker_fee: float = 0.0002, slippage: float = 0.0002,
                 stop_penetration: float = 0.25, default_leverage: float = 10.0):
        self.m, self.clock = market, clock
        self.wallet_balance = float(equity)
        self.taker, self.maker, self.slip, self.pen = taker_fee, maker_fee, slippage, stop_penetration
        self.default_lev = default_leverage
        self.lev: dict[str, float] = {}
        self.pos: dict[str, SimPos] = {}
        self.orders: list[dict] = []          # история ордеров (все исполненные/отклонённые)
        self.links: set[str] = set()
        self.execs: list[dict] = []
        self.txlog: list[dict] = []
        self.margin_mode = "REGULAR_MARGIN"
        self.position_mode_set = False
        self.extra_open_orders: list[dict] = []
        self.fail_next: dict[str, str] = {}
        self.reject_next: dict[str, tuple[int, str]] = {}
        self.calls: list[str] = []
        self._ids = itertools.count(1)
        t0 = min(int(s.ts[0]) for s in market.values())
        self.cursor = (clock.now_ms() // M15) * M15 if clock.now_ms() > t0 else t0
        self.funded: set[int] = set()

    # ------------------------------------------------------------ построение
    @classmethod
    def from_dir(cls, data_dir: Path, symbols: list[str], clock: SimClock, **kw) -> "SimBybit":
        mk = {}
        for s in symbols:
            d = Path(data_dir) / s
            k = pd.read_parquet(d / "kline_last_15m.parquet").sort_values("ts")
            f = pd.read_parquet(d / "funding.parquet") if (d / "funding.parquet").exists() else \
                pd.DataFrame({"ts": [], "rate": []})
            inst = json.loads((d / "instrument.json").read_text(encoding="utf-8"))["instrument"]
            tiers = json.loads((d / "risk_limit.json").read_text(encoding="utf-8"))["tiers"]
            mk[s] = SymMarket(k["ts"].to_numpy("int64"), *(k[c].to_numpy(float) for c in
                                                            ("open", "high", "low", "close", "volume", "turnover")),
                              f["ts"].to_numpy("int64"), f["rate"].to_numpy(float), inst, tiers)
        return cls(mk, clock, **kw)

    @classmethod
    def from_panel(cls, panel, clock: SimClock, instrument: dict, tiers: list[dict], **kw) -> "SimBybit":
        mk = {}
        for r, s in enumerate(panel.all_symbols):
            ok = ~np.isnan(panel.close[r])
            sd = panel.sym[s]
            mk[s] = SymMarket(panel.ts[ok], panel.open[r][ok], panel.high[r][ok], panel.low[r][ok],
                              panel.close[r][ok], panel.volume[r][ok], panel.turnover[r][ok],
                              sd.funding_ts, sd.funding_rate, {**instrument, "symbol": s}, tiers)
        return cls(mk, clock, **kw)

    # ------------------------------------------------------------ служебное
    def _ok(self, result) -> dict:
        return {"retCode": 0, "retMsg": "OK", "result": result, "retExtInfo": {}, "time": self.clock.now_ms()}

    def _enter(self, name: str) -> str | None:
        self.calls.append(name)
        self._sync()
        if name in self.reject_next:
            code, msg = self.reject_next.pop(name)
            raise _err(code, msg)
        mode = self.fail_next.pop(name, None)
        if mode == "before":
            raise _net()
        return mode

    def price(self, symbol: str) -> float:
        """Текущая цена сделки: открытие текущей 15m-свечи (или последнее закрытие)."""
        sm = self.m[symbol]
        i = sm.idx(self.clock.now_ms())
        if i >= 0:
            return float(sm.o[i])
        j = int(np.searchsorted(sm.ts, self.clock.now_ms(), "right")) - 1
        return float(sm.c[max(j, 0)])

    def mark(self, symbol: str) -> float:
        """Последнее закрытие завершённой 15m-свечи — им оценивается позиция (как в бэктесте)."""
        sm = self.m[symbol]
        j = int(np.searchsorted(sm.ts, self.clock.now_ms() - M15, "right")) - 1
        return float(sm.c[j]) if j >= 0 else float(sm.o[0])

    def _tier(self, symbol: str, value: float):
        return self.m[symbol].inst.tier_for(value)

    def _liq(self, p: SimPos) -> float:
        return liquidation_price(p.side, p.entry, p.leverage, self._tier(p.symbol, p.size * p.entry).mmr, self.taker)

    def _margin(self, p: SimPos) -> float:
        return p.size * p.entry / p.leverage

    def available(self) -> float:
        return self.wallet_balance - sum(self._margin(p) for p in self.pos.values())

    def equity(self) -> float:
        return self.wallet_balance + sum(p.side * p.size * (self.mark(s) - p.entry) for s, p in self.pos.items())

    # ------------------------------------------------- доигрывание свечей
    def _sync(self) -> None:
        now = self.clock.now_ms()
        while self.cursor + M15 <= now:
            self._funding(self.cursor)
            self._candle(self.cursor)
            self.cursor += M15
        if self.cursor <= now:
            self._funding(self.cursor)

    def _funding(self, t: int) -> None:
        if t in self.funded:
            return
        self.funded.add(t)
        for s, p in list(self.pos.items()):
            sm = self.m[s]
            j = np.flatnonzero(sm.f_ts == t)
            i = sm.idx(t)
            if not len(j) or i < 0:
                continue
            pay = p.side * p.size * float(sm.o[i]) * float(sm.f_rate[j[0]])
            self.wallet_balance -= pay
            p.realised -= pay
            self.txlog.append({"symbol": s, "category": "linear", "type": "SETTLEMENT", "currency": "USDT",
                               "change": _s(-pay), "funding": _s(pay), "transactionTime": str(t),
                               "id": f"tx{next(self._ids)}"})

    def _candle(self, t: int) -> None:
        for s, p in list(self.pos.items()):
            sm = self.m[s]
            i = sm.idx(t)
            if i < 0 or sm.ts[i] != t:
                continue
            o, lo, hi = float(sm.o[i]), float(sm.l[i]), float(sm.h[i])
            liq = self._liq(p)
            stop_hit = p.stop is not None and (lo <= p.stop if p.side > 0 else hi >= p.stop)
            liq_hit = lo <= liq if p.side > 0 else hi >= liq
            if stop_hit:
                if p.side > 0:
                    base = min(o, p.stop)
                    fill = base - self.pen * max(0.0, base - lo)
                else:
                    base = max(o, p.stop)
                    fill = base + self.pen * max(0.0, hi - base)
                fill *= 1 - p.side * self.slip
                beyond = fill <= liq if p.side > 0 else fill >= liq
                if not beyond:
                    self._fill_close(p, fill, self.taker, t, exec_type="Trade", stop_order_type="StopLoss")
                    continue
            if stop_hit or liq_hit:
                self._fill_close(p, p.entry * (1 - p.side / p.leverage), 0.0, t, exec_type="BustTrade",
                                 stop_order_type="")

    def _fill_close(self, p: SimPos, price: float, fee_rate: float, t: int, exec_type: str, stop_order_type: str,
                    link_id: str = "", order_id: str | None = None) -> None:
        gross = p.side * p.size * (price - p.entry)
        fee = p.size * price * fee_rate
        self.wallet_balance += gross - fee
        oid = order_id or f"o{next(self._ids)}"
        self.execs.append({"execId": f"e{next(self._ids)}", "orderId": oid, "orderLinkId": link_id,
                           "symbol": p.symbol, "side": "Sell" if p.side > 0 else "Buy", "execPrice": _s(price),
                           "execQty": _s(p.size), "execFee": _s(fee), "execType": exec_type,
                           "stopOrderType": stop_order_type, "execTime": str(t), "closedSize": _s(p.size),
                           "feeRate": _s(fee_rate)})
        if not link_id:
            self.orders.append({"orderId": oid, "orderLinkId": "", "symbol": p.symbol,
                                "side": "Sell" if p.side > 0 else "Buy", "orderType": "Market", "qty": _s(p.size),
                                "cumExecQty": _s(p.size), "avgPrice": _s(price), "orderStatus": "Filled",
                                "reduceOnly": True, "stopOrderType": stop_order_type, "cumExecFee": _s(fee),
                                "createdTime": str(t), "updatedTime": str(t)})
        del self.pos[p.symbol]

    # ============================================================ API pybit
    def get_server_time(self, **_):
        self._enter("get_server_time")
        now = self.clock.now_ms()
        return self._ok({"timeSecond": str(now // 1000), "timeNano": str(now * 1_000_000)})

    def get_instruments_info(self, **p):
        self._enter("get_instruments_info")
        s = p.get("symbol")
        lst = [self.m[s].instrument] if s in self.m else ([] if s else [x.instrument for x in self.m.values()])
        return self._ok({"category": "linear", "list": lst, "nextPageCursor": ""})

    def get_risk_limit(self, **p):
        self._enter("get_risk_limit")
        return self._ok({"category": "linear", "list": self.m[p["symbol"]].tiers, "nextPageCursor": ""})

    def get_kline(self, **p):
        self._enter("get_kline")
        assert int(p["limit"]) <= 1000 and p.get("interval") == "15"
        sm = self.m[p["symbol"]]
        now = self.clock.now_ms()
        cur = (now // M15) * M15
        sel = (sm.ts >= int(p["start"])) & (sm.ts <= int(p["end"])) & (sm.ts <= cur)
        rows = []
        for i in np.flatnonzero(sel)[-int(p["limit"]):]:
            if sm.ts[i] == cur:      # формирующаяся свеча: на бирже она отдаётся неполной
                rows.append([str(sm.ts[i]), _s(sm.o[i]), _s(sm.o[i]), _s(sm.o[i]), _s(sm.o[i]), "0", "0"])
            else:
                rows.append([str(sm.ts[i]), _s(sm.o[i]), _s(sm.h[i]), _s(sm.l[i]), _s(sm.c[i]), _s(sm.v[i]),
                             _s(sm.turnover[i])])
        return self._ok({"symbol": p["symbol"], "category": "linear", "list": list(reversed(rows))})

    def get_funding_rate_history(self, **p):
        self._enter("get_funding_rate_history")
        sm = self.m[p["symbol"]]
        a, b = int(p["startTime"]), min(int(p["endTime"]), self.clock.now_ms())
        idx = np.flatnonzero((sm.f_ts >= a) & (sm.f_ts <= b))[::-1][: int(p["limit"])]
        return self._ok({"category": "linear", "list": [
            {"symbol": p["symbol"], "fundingRate": _s(sm.f_rate[i]), "fundingRateTimestamp": str(sm.f_ts[i])}
            for i in idx]})

    def get_tickers(self, **p):
        self._enter("get_tickers")
        s = p["symbol"]
        px = self.price(s)
        tick = float(self.m[s].inst.tick_size)
        return self._ok({"category": "linear", "list": [{
            "symbol": s, "lastPrice": _s(px), "bid1Price": _s(px - tick), "ask1Price": _s(px + tick),
            "markPrice": _s(self.mark(s))}]})

    def get_account_info(self, **_):
        self._enter("get_account_info")
        return self._ok({"marginMode": self.margin_mode, "unifiedMarginStatus": 5})

    def set_margin_mode(self, **p):
        self._enter("set_margin_mode")
        if self.pos:
            raise _err(3400045, "есть открытые позиции")
        self.margin_mode = p["setMarginMode"]
        return self._ok({"reasons": []})

    def switch_position_mode(self, **p):
        self._enter("switch_position_mode")
        if self.position_mode_set:
            raise _err(110025, "Position mode is not modified")
        self.position_mode_set = True
        return self._ok({})

    def get_wallet_balance(self, **_):
        self._enter("get_wallet_balance")
        upl = self.equity() - self.wallet_balance
        im = sum(self._margin(p) for p in self.pos.values())
        coin = {"coin": "USDT", "walletBalance": _s(self.wallet_balance), "equity": _s(self.equity()),
                "unrealisedPnl": _s(upl), "totalPositionIM": _s(im), "totalOrderIM": "0",
                "availableToWithdraw": _s(self.wallet_balance - im)}
        return self._ok({"list": [{"accountType": "UNIFIED", "totalEquity": _s(self.equity()), "coin": [coin]}]})

    def get_fee_rates(self, **p):
        self._enter("get_fee_rates")
        return self._ok({"list": [{"symbol": p.get("symbol", ""), "takerFeeRate": _s(self.taker),
                                   "makerFeeRate": _s(self.maker)}]})

    def get_positions(self, **_):
        self._enter("get_positions")
        out = []
        for s, p in self.pos.items():
            out.append({"symbol": s, "side": "Buy" if p.side > 0 else "Sell", "size": _s(p.size),
                        "avgPrice": _s(p.entry), "leverage": _s(p.leverage), "liqPrice": _s(self._liq(p)),
                        "stopLoss": _s(p.stop) if p.stop is not None else "", "takeProfit": "",
                        "markPrice": _s(self.mark(s)), "unrealisedPnl": _s(p.side * p.size * (self.mark(s) - p.entry)),
                        "positionIM": _s(self._margin(p)), "positionIdx": 0, "tpslMode": "Full",
                        "positionStatus": "Normal", "updatedTime": str(self.clock.now_ms())})
        return self._ok({"category": "linear", "list": out, "nextPageCursor": ""})

    def _sl_orders(self) -> list[dict]:
        out = []
        for s, p in self.pos.items():
            if p.stop is not None:
                out.append({"orderId": f"sl-{s}", "orderLinkId": "", "symbol": s,
                            "side": "Sell" if p.side > 0 else "Buy", "orderType": "Market", "qty": "0",
                            "cumExecQty": "0", "avgPrice": "", "orderStatus": "Untriggered", "reduceOnly": True,
                            "stopOrderType": "StopLoss", "triggerPrice": _s(p.stop), "cumExecFee": "0",
                            "createdTime": str(p.created), "updatedTime": str(p.created)})
        return out

    def get_open_orders(self, **p):
        self._enter("get_open_orders")
        lst = self._sl_orders() + self.extra_open_orders
        if p.get("orderLinkId"):
            lst = [o for o in lst if o.get("orderLinkId") == p["orderLinkId"]]
        return self._ok({"category": "linear", "list": lst, "nextPageCursor": ""})

    def get_order_history(self, **p):
        self._enter("get_order_history")
        lst = list(reversed(self.orders))
        if p.get("orderLinkId"):
            lst = [o for o in lst if o.get("orderLinkId") == p["orderLinkId"]]
        return self._ok({"category": "linear", "list": lst[: int(p.get("limit", 50))], "nextPageCursor": ""})

    def get_executions(self, **p):
        self._enter("get_executions")
        a, b = int(p.get("startTime", 0)), int(p.get("endTime", 2**62))
        lst = [e for e in self.execs if a <= int(e["execTime"]) <= b and
               (not p.get("symbol") or e["symbol"] == p["symbol"])]
        return self._ok({"category": "linear", "list": list(reversed(lst)), "nextPageCursor": ""})

    def get_transaction_log(self, **p):
        self._enter("get_transaction_log")
        a, b = int(p.get("startTime", 0)), int(p.get("endTime", 2**62))
        lst = [x for x in self.txlog if a <= int(x["transactionTime"]) <= b]
        return self._ok({"list": list(reversed(lst)), "nextPageCursor": ""})

    def set_leverage(self, **p):
        self._enter("set_leverage")
        s, lev = p["symbol"], float(p["buyLeverage"])
        inst = self.m[s].inst
        if lev < 1 or lev > inst.max_leverage:
            raise _err(10001, "leverage invalid")
        if self.lev.get(s, self.default_lev) == lev and (s not in self.pos or self.pos[s].leverage == lev):
            raise _err(110043, "Set leverage not modified")
        if s in self.pos:
            pos = self.pos[s]
            extra = pos.size * pos.entry / lev - self._margin(pos)
            if extra > self.available():
                raise _err(110007, "ab not enough for new leverage")
            pos.leverage = lev
        self.lev[s] = lev
        return self._ok({})

    def place_order(self, **p):
        mode = self._enter("place_order")
        s, link = p["symbol"], p.get("orderLinkId", "")
        if link and link in self.links:
            raise _err(110072, "OrderLinkedID is duplicate")
        sm = self.m[s]
        inst = sm.inst
        side = 1 if p["side"] == "Buy" else -1
        qty = float(p["qty"])
        px = self.price(s)
        fill = px * (1 + side * self.slip)
        if abs(qty / float(inst.qty_step) - round(qty / float(inst.qty_step))) > 1e-6:
            raise _err(10001, "Qty invalid")
        pos = self.pos.get(s)
        reduce = bool(p.get("reduceOnly"))
        if reduce and (pos is None or pos.side == side):
            raise _err(110017, "current position is zero, cannot fix reduce-only order qty")
        if not reduce and pos is not None and pos.side != side:
            raise _err(10001, "в тестах встречный ордер без reduceOnly не используется")
        if not reduce and (qty < inst.min_qty or qty * px < inst.min_notional):
            raise _err(110094, "Order does not meet minimum order value")
        sl = p.get("stopLoss")
        if sl is not None and not reduce:
            slv = float(sl)
            if (side > 0 and slv >= px) or (side < 0 and slv <= px):
                raise _err(10001, f"StopLoss:{sl} set for {'Buy' if side > 0 else 'Sell'} position should be "
                                  f"{'lower' if side > 0 else 'higher'} than base_price:{px}")
        lev = self.lev.get(s, self.default_lev)
        if not reduce and qty * fill * (1 / lev + self.taker) > self.available():
            raise _err(110007, "ab not enough for new order")
        oid = f"o{next(self._ids)}"
        now = self.clock.now_ms()
        if link:
            self.links.add(link)
        fee = qty * fill * self.taker
        if reduce:
            q = min(qty, pos.size)
            part = SimPos(s, pos.side, q, pos.entry, pos.leverage)
            self.pos[s] = part
            self._fill_close(part, fill, self.taker, now, exec_type="Trade", stop_order_type="", link_id=link,
                             order_id=oid)
            if q < pos.size:
                pos.size -= q
                self.pos[s] = pos
            fee = q * fill * self.taker
        else:
            self.wallet_balance -= fee
            if pos is None:
                pos = SimPos(s, side, qty, fill, lev, created=now)
                self.pos[s] = pos
            else:
                pos.entry = (pos.entry * pos.size + fill * qty) / (pos.size + qty)
                pos.size += qty
            if sl is not None:
                pos.stop = float(sl)
            self.execs.append({"execId": f"e{next(self._ids)}", "orderId": oid, "orderLinkId": link, "symbol": s,
                               "side": p["side"], "execPrice": _s(fill), "execQty": _s(qty), "execFee": _s(fee),
                               "execType": "Trade", "stopOrderType": "", "execTime": str(now), "closedSize": "0",
                               "feeRate": _s(self.taker)})
        self.orders.append({"orderId": oid, "orderLinkId": link, "symbol": s, "side": p["side"],
                            "orderType": "Market", "qty": p["qty"], "cumExecQty": p["qty"], "avgPrice": _s(fill),
                            "orderStatus": "Filled", "reduceOnly": reduce, "stopOrderType": "",
                            "cumExecFee": _s(fee), "createdTime": str(now), "updatedTime": str(now)})
        if mode == "after":
            raise _net()
        return self._ok({"orderId": oid, "orderLinkId": link})

    def set_trading_stop(self, **p):
        mode = self._enter("set_trading_stop")
        s = p["symbol"]
        pos = self.pos.get(s)
        if pos is None:
            raise _err(10001, "can not set tp/sl/ts for zero position")
        v = float(p["stopLoss"])
        px = self.price(s)
        if (pos.side > 0 and v >= px) or (pos.side < 0 and v <= px):
            raise _err(10001, f"StopLoss:{v} set for {'Buy' if pos.side > 0 else 'Sell'} position should be "
                              f"{'lower' if pos.side > 0 else 'higher'} than base_price:{px}")
        if pos.stop == v:
            raise _err(34040, "not modified")
        pos.stop = v
        if mode == "after":
            raise _net()
        return self._ok({})

    def cancel_all_orders(self, **_):
        self._enter("cancel_all_orders")
        self.extra_open_orders = []
        return self._ok({"list": [], "success": "1"})

    # ------------------------------------------------- вспомогательное для тестов
    def drop_stop(self, symbol: str) -> None:
        """Имитация: стоп пропал с биржи (ручная отмена и т. п.)."""
        self.pos[symbol].stop = None

    def close_externally(self, symbol: str) -> None:
        """Имитация: позицию закрыли вручную в приложении биржи."""
        self._sync()
        p = self.pos[symbol]
        self._fill_close(p, self.price(symbol) * (1 - p.side * self.slip), self.taker, self.clock.now_ms(),
                         exec_type="Trade", stop_order_type="")


def utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def write_market_dir(panel, data_dir: Path, instrument: dict, tiers: list[dict]) -> None:
    """Сохраняет Panel в формате хранилища (для SimBybit.from_dir и тестов загрузчика)."""
    for r, s in enumerate(panel.all_symbols):
        ok = ~np.isnan(panel.close[r])
        d = store.symbol_dir(data_dir, s)
        pd.DataFrame({"ts": panel.ts[ok], "open": panel.open[r][ok], "high": panel.high[r][ok],
                      "low": panel.low[r][ok], "close": panel.close[r][ok], "volume": panel.volume[r][ok],
                      "turnover": panel.turnover[r][ok]}).to_parquet(d / "kline_last_15m.parquet", index=False)
        sd = panel.sym[s]
        pd.DataFrame({"ts": sd.funding_ts, "rate": sd.funding_rate}).to_parquet(d / "funding.parquet", index=False)
        store.write_json(d / "instrument.json", {"instrument": {**instrument, "symbol": s}})
        store.write_json(d / "risk_limit.json", {"tiers": tiers})
