"""Клиент Bybit V5 (category=linear) поверх сессии pybit.

Один и тот же код работает с настоящей сессией `pybit.unified_trading.HTTP` (demo или live)
и с имитатором биржи `bot.exchange.sim.SimBybit`, у которого те же методы и тот же формат
ответов. Ответы приводятся к простым dataclass.

Ошибки:
  - ExchangeError — биржа ответила отказом (retCode ≠ 0): запрос точно не выполнен;
  - NetworkError — нет ответа (сеть, тайм-аут, HTTP ≠ 200): состояние НЕИЗВЕСТНО. Для ордеров
    вызывающий обязан проверить их по orderLinkId (order_by_link_id), а не слать повторно вслепую.

Чтения (GET) повторяются при сетевых ошибках; запросы, меняющие состояние, — нет.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable

log = logging.getLogger(__name__)

# Коды ответов Bybit V5, которые означают «уже так и есть» — не ошибка.
NOT_MODIFIED = {110025, 110026, 110043, 34040, 34036}
DUPLICATE_LINK_ID = 110072
REDUCE_ONLY_NOTHING = {110017}          # нечего сокращать: позиции уже нет
ORDER_NOT_FOUND = {110001, 110008, 170213}


# Что означают частые коды отказа Bybit — простыми словами (для самопроверки и мастера ключей).
HINTS = {
    10002: "часы компьютера расходятся с биржей: Параметры → Время и язык → «Синхронизировать сейчас»",
    10003: "биржа не знает этот ключ: ключ демо-счёта создаётся в режиме «Демо-торговля», ключ реального "
           "счёта — в обычном режиме; или ключ удалён",
    10004: "секрет не подходит к ключу (ошибка подписи): секрет скопирован не полностью или с лишними "
           "символами. Bybit показывает секрет только при создании ключа — удалите ключ, создайте новый и "
           "вставьте секрет правой кнопкой мыши",
    10005: "у ключа нет нужных прав: «Контракты: Ордера, Позиции» и «Единый торговый аккаунт: Торговля»",
    10010: "запрос пришёл с IP, которого нет в списке разрешённых у ключа",
    33004: "срок действия ключа истёк — создайте новый",
}


def explain(code: int) -> str:
    return HINTS.get(int(code), "")


def _clean(message: str) -> str:
    """Bybit вставляет в текст ошибки подписи строку с API-ключом — в журнал и отчёты она не попадает."""
    return re.sub(r"origin_string\[[^\]]*\]", "origin_string[скрыто]", str(message))


class ExchangeError(RuntimeError):
    def __init__(self, code: int, message: str):
        self.code, self.message = int(code), _clean(message)
        super().__init__(f"Bybit отказал: {self.message} (код {code})")


class NetworkError(RuntimeError):
    pass


def _f(x, default: float | None = 0.0) -> float | None:
    try:
        if x is None or x == "":
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class PositionInfo:
    symbol: str
    side: int               # +1 лонг, −1 шорт
    size: float
    avg_price: float
    stop_loss: float | None
    liq_price: float | None
    leverage: float
    mark_price: float
    unrealised_pnl: float
    position_im: float
    status: str
    updated_ms: int


@dataclass(frozen=True)
class OrderInfo:
    order_id: str
    link_id: str
    symbol: str
    side: int
    order_type: str
    qty: float
    filled_qty: float
    avg_price: float
    status: str
    reduce_only: bool
    stop_order_type: str
    fee: float
    created_ms: int
    updated_ms: int
    reject_reason: str = ""

    @property
    def done(self) -> bool:
        return self.status in ("Filled", "Cancelled", "Rejected", "PartiallyFilledCanceled", "Deactivated")


@dataclass(frozen=True)
class Execution:
    exec_id: str
    order_id: str
    link_id: str
    symbol: str
    side: int
    price: float
    qty: float
    fee: float
    exec_type: str
    stop_order_type: str
    ts: int
    closed_size: float


@dataclass(frozen=True)
class Wallet:
    equity: float           # USDT: баланс + нереализованный результат
    wallet_balance: float
    available: float        # свободно для новой маржи
    unrealised_pnl: float


@dataclass(frozen=True)
class Ticker:
    symbol: str
    last: float
    bid: float
    ask: float
    mark: float


def side_str(side: int) -> str:
    return "Buy" if side > 0 else "Sell"


def side_int(s: str) -> int:
    return 1 if s == "Buy" else -1 if s == "Sell" else 0


class BybitClient:
    def __init__(self, session, category: str = "linear", settle_coin: str = "USDT",
                 get_attempts: int = 4, sleep: Callable[[float], None] = time.sleep):
        self.s, self.category, self.coin = session, category, settle_coin
        self.get_attempts, self.sleep = get_attempts, sleep
        self.requests = 0

    # ------------------------------------------------------------ транспорт
    def _call(self, name: str, retry: bool, **params: Any) -> dict:
        from pybit.exceptions import FailedRequestError, InvalidRequestError
        import requests

        fn = getattr(self.s, name)
        attempts = self.get_attempts if retry else 1
        delay = 1.0
        for i in range(1, attempts + 1):
            try:
                self.requests += 1
                resp = fn(**params)
                if isinstance(resp, tuple):          # pybit с record_request_time
                    resp = resp[0]
                code = resp.get("retCode", 0)
                if code:
                    raise ExchangeError(code, resp.get("retMsg", ""))
                return resp.get("result") or {}
            except InvalidRequestError as e:
                raise ExchangeError(getattr(e, "status_code", -1), getattr(e, "message", str(e))) from e
            except (FailedRequestError, requests.exceptions.RequestException, ConnectionError, TimeoutError) as e:
                err = e
            if i < attempts:
                log.warning("Сеть: %s — повтор %s через %.0f с", str(err).splitlines()[0][:200], name, delay)
                self.sleep(delay)
                delay = min(delay * 2, 10)
        raise NetworkError(f"{name}: нет ответа биржи ({str(err).splitlines()[0][:200]})") from err

    def _get(self, name: str, **p) -> dict:
        return self._call(name, True, **p)

    def _post(self, name: str, **p) -> dict:
        return self._call(name, False, **p)

    def _pages(self, name: str, **p) -> list[dict]:
        out, cursor = [], None
        for _ in range(50):
            q = dict(p)
            if cursor:
                q["cursor"] = cursor
            res = self._get(name, **q)
            out += res.get("list") or []
            cursor = res.get("nextPageCursor")
            if not cursor:
                break
        return out

    # --------------------------------------------------------------- рынок
    def server_time_ms(self) -> int:
        res = self._get("get_server_time")
        if "timeNano" in res:
            return int(int(res["timeNano"]) // 1_000_000)
        return int(res.get("timeSecond", 0)) * 1000

    def instrument(self, symbol: str) -> dict:
        lst = self._get("get_instruments_info", category=self.category, symbol=symbol).get("list") or []
        if not lst:
            raise ExchangeError(-1, f"инструмент {symbol} не найден")
        return lst[0]

    def risk_limits(self, symbol: str) -> list[dict]:
        return self._pages("get_risk_limit", category=self.category, symbol=symbol)

    def ticker(self, symbol: str) -> Ticker:
        lst = self._get("get_tickers", category=self.category, symbol=symbol).get("list") or []
        if not lst:
            raise ExchangeError(-1, f"нет тикера {symbol}")
        t = lst[0]
        return Ticker(symbol, _f(t.get("lastPrice")), _f(t.get("bid1Price")), _f(t.get("ask1Price")),
                      _f(t.get("markPrice")))

    # ------------------------------------------------------------- аккаунт
    def account_info(self) -> dict:
        return self._get("get_account_info")

    def ensure_isolated_one_way(self) -> list[str]:
        """Изолированная маржа на уровне аккаунта и режим одной позиции. Возвращает что изменено."""
        done = []
        info = self.account_info()
        if info.get("marginMode") != "ISOLATED_MARGIN":
            try:
                self._post("set_margin_mode", setMarginMode="ISOLATED_MARGIN")
                done.append("маржа: изолированная")
            except ExchangeError as e:
                if e.code not in NOT_MODIFIED:
                    raise
        try:
            self._post("switch_position_mode", category=self.category, coin=self.coin, mode=0)
            done.append("режим позиций: одна позиция на монету")
        except ExchangeError as e:
            # на части счетов запрос не поддерживается; режим всё равно проверяется каждым ордером
            # (positionIdx=0 в режиме хеджа биржа отклоняет)
            if e.code not in NOT_MODIFIED:
                done.append(f"режим одной позиции не подтверждён запросом (код {e.code}: {e.message[:80]})")
        return done

    def wallet(self) -> Wallet:
        res = self._get("get_wallet_balance", accountType="UNIFIED", coin=self.coin)
        acc = (res.get("list") or [{}])[0]
        coins = [c for c in acc.get("coin") or [] if c.get("coin") == self.coin]
        c = coins[0] if coins else {}
        wb = _f(c.get("walletBalance"))
        upl = _f(c.get("unrealisedPnl"))
        eq = _f(c.get("equity"), None)
        if eq is None:
            eq = wb + upl
        avail = _f(c.get("availableToWithdraw"), None)
        if avail is None:
            avail = wb - _f(c.get("totalPositionIM")) - _f(c.get("totalOrderIM"))
        return Wallet(equity=eq, wallet_balance=wb, available=avail, unrealised_pnl=upl)

    def fee_rate(self, symbol: str) -> tuple[float, float]:
        lst = self._get("get_fee_rates", category=self.category, symbol=symbol).get("list") or []
        if not lst:
            raise ExchangeError(-1, "нет ставок комиссии")
        return _f(lst[0].get("makerFeeRate")), _f(lst[0].get("takerFeeRate"))

    # ------------------------------------------------------------- позиции
    def positions(self) -> list[PositionInfo]:
        out = []
        for p in self._pages("get_positions", category=self.category, settleCoin=self.coin, limit=200):
            size = _f(p.get("size"))
            if not size:
                continue
            sl = _f(p.get("stopLoss"), None)
            liq = _f(p.get("liqPrice"), None)
            out.append(PositionInfo(
                symbol=p["symbol"], side=side_int(p.get("side", "")), size=size, avg_price=_f(p.get("avgPrice")),
                stop_loss=sl if sl else None, liq_price=liq if liq else None, leverage=_f(p.get("leverage"), 1.0),
                mark_price=_f(p.get("markPrice")), unrealised_pnl=_f(p.get("unrealisedPnl")),
                position_im=_f(p.get("positionIM")), status=p.get("positionStatus", "Normal"),
                updated_ms=int(_f(p.get("updatedTime"), 0))))
        return out

    # --------------------------------------------------------------- ордера
    @staticmethod
    def _order(o: dict) -> OrderInfo:
        return OrderInfo(
            order_id=o.get("orderId", ""), link_id=o.get("orderLinkId", ""), symbol=o.get("symbol", ""),
            side=side_int(o.get("side", "")), order_type=o.get("orderType", ""), qty=_f(o.get("qty")),
            filled_qty=_f(o.get("cumExecQty")), avg_price=_f(o.get("avgPrice")), status=o.get("orderStatus", ""),
            reduce_only=bool(o.get("reduceOnly")), stop_order_type=o.get("stopOrderType", "") or "",
            fee=_f(o.get("cumExecFee")), created_ms=int(_f(o.get("createdTime"), 0)),
            updated_ms=int(_f(o.get("updatedTime"), 0)), reject_reason=o.get("rejectReason", "") or "")

    def open_orders(self) -> list[OrderInfo]:
        return [self._order(o) for o in self._pages("get_open_orders", category=self.category,
                                                     settleCoin=self.coin, limit=50)]

    def order_by_link_id(self, link_id: str, symbol: str | None = None) -> OrderInfo | None:
        scope = {"symbol": symbol} if symbol else {"settleCoin": self.coin}
        errors = []
        for name in ("get_open_orders", "get_order_history"):
            try:
                lst = self._get(name, category=self.category, orderLinkId=link_id, limit=1, **scope).get("list") or []
            except ExchangeError as e:       # метод может быть недоступен (например, на демо) — пробуем другой
                errors.append(e)
                continue
            if lst:
                return self._order(lst[0])
        if len(errors) == 2:
            raise errors[-1]
        return None

    def order_history(self, symbol: str, since_ms: int, max_pages: int = 3) -> list[OrderInfo]:
        """Последние ордера по монете (новые первыми), обновлённые не раньше since_ms."""
        out, cursor = [], None
        for _ in range(max_pages):
            q: dict[str, Any] = dict(category=self.category, symbol=symbol, limit=50)
            if cursor:
                q["cursor"] = cursor
            res = self._get("get_order_history", **q)
            page = [self._order(o) for o in res.get("list") or []]
            out += [o for o in page if o.updated_ms >= since_ms]
            cursor = res.get("nextPageCursor")
            if not cursor or (page and min(o.updated_ms for o in page) < since_ms):
                break
        return out

    def set_leverage(self, symbol: str, leverage: float) -> None:
        v = f"{leverage:.2f}".rstrip("0").rstrip(".")
        try:
            self._post("set_leverage", category=self.category, symbol=symbol, buyLeverage=v, sellLeverage=v)
        except ExchangeError as e:
            if e.code not in NOT_MODIFIED:
                raise

    def place_market(self, symbol: str, side: int, qty: str, link_id: str, stop_loss: str | None = None,
                     reduce_only: bool = False, sl_trigger_by: str = "LastPrice") -> str:
        """Рыночный ордер; со stop_loss — вход и биржевой стоп одним запросом. Возвращает orderId."""
        p: dict[str, Any] = dict(category=self.category, symbol=symbol, side=side_str(side), orderType="Market",
                                 qty=qty, orderLinkId=link_id, positionIdx=0)
        if reduce_only:
            p["reduceOnly"] = True
        if stop_loss is not None:
            p.update(stopLoss=stop_loss, slTriggerBy=sl_trigger_by, tpslMode="Full", slOrderType="Market")
        return self._post("place_order", **p).get("orderId", "")

    def set_stop(self, symbol: str, stop: str, sl_trigger_by: str = "LastPrice") -> None:
        try:
            self._post("set_trading_stop", category=self.category, symbol=symbol, stopLoss=stop,
                       slTriggerBy=sl_trigger_by, tpslMode="Full", slOrderType="Market", positionIdx=0)
        except ExchangeError as e:
            if e.code not in NOT_MODIFIED:
                raise

    def cancel_all(self) -> None:
        self._post("cancel_all_orders", category=self.category, settleCoin=self.coin)

    # ------------------------------------------------------------- история
    def executions(self, start_ms: int, end_ms: int, symbol: str | None = None) -> list[Execution]:
        out = []
        week = 7 * 86_400_000 - 1
        a = start_ms
        while a <= end_ms:
            b = min(a + week, end_ms)
            q: dict[str, Any] = dict(category=self.category, startTime=a, endTime=b, limit=100)
            if symbol:
                q["symbol"] = symbol
            for e in self._pages("get_executions", **q):
                out.append(Execution(
                    exec_id=e.get("execId", ""), order_id=e.get("orderId", ""), link_id=e.get("orderLinkId", ""),
                    symbol=e.get("symbol", ""), side=side_int(e.get("side", "")), price=_f(e.get("execPrice")),
                    qty=_f(e.get("execQty")), fee=_f(e.get("execFee")), exec_type=e.get("execType", ""),
                    stop_order_type=e.get("stopOrderType", "") or "", ts=int(_f(e.get("execTime"), 0)),
                    closed_size=_f(e.get("closedSize"))))
            a = b + 1
        uniq = {e.exec_id: e for e in out}
        return sorted(uniq.values(), key=lambda e: (e.ts, e.exec_id))

    def closed_pnl(self, symbol: str, start_ms: int, end_ms: int) -> list[dict]:
        """Закрытый результат по монете (/v5/position/closed-pnl), окнами по 7 дней."""
        out, week, a = [], 7 * 86_400_000 - 1, start_ms
        while a <= end_ms:
            b = min(a + week, end_ms)
            out += self._pages("get_closed_pnl", category=self.category, symbol=symbol, startTime=a, endTime=b,
                               limit=100)
            a = b + 1
        return out

    def funding_paid(self, symbol: str, start_ms: int, end_ms: int) -> float | None:
        """Финансирование по монете за период (USDT, + — уплачено). None — журнал недоступен."""
        total, week, a = 0.0, 7 * 86_400_000 - 1, start_ms
        try:
            while a <= end_ms:
                b = min(a + week, end_ms)
                for x in self._pages("get_transaction_log", accountType="UNIFIED", category=self.category,
                                     currency=self.coin, type="SETTLEMENT", startTime=a, endTime=b, limit=50):
                    if x.get("symbol") == symbol:
                        total -= _f(x.get("change"))      # change < 0 — списано
                a = b + 1
        except ExchangeError:
            return None
        return total
