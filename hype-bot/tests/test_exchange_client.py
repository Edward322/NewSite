"""Клиент Bybit V5: разбор ответов и обработка ошибок (как у настоящей сессии pybit)."""
import pytest
import requests
from pybit.exceptions import FailedRequestError, InvalidRequestError

from bot.exchange.client import BybitClient, ExchangeError, NetworkError


class Session:
    """Сессия, возвращающая заданные ответы/исключения по очереди."""

    def __init__(self, **script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def __getattr__(self, name):
        def fn(**p):
            self.calls.append((name, p))
            item = self.script[name].pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return fn


def ok(result):
    return {"retCode": 0, "retMsg": "OK", "result": result}


def inv(code, msg="x"):
    return InvalidRequestError(request="r", message=msg, status_code=code, time="", resp_headers=None)


def net():
    return FailedRequestError(request="r", message="HTTP status code is not 200.", status_code=502, time="",
                              resp_headers=None)


def client(**script):
    sleeps = []
    c = BybitClient(Session(**script), sleep=sleeps.append)
    return c, sleeps


def test_business_error_is_exchange_error_with_code():
    c, _ = client(place_order=[inv(110007, "ab not enough")])
    with pytest.raises(ExchangeError) as e:
        c.place_market("XRPUSDT", 1, "10", "hbd1")
    assert e.value.code == 110007


def test_get_retried_on_network_errors_then_network_error():
    c, sleeps = client(get_positions=[net(), requests.exceptions.ConnectionError("x"), net(), net()])
    with pytest.raises(NetworkError):
        c.positions()
    assert len(c.s.calls) == 4 and len(sleeps) == 3


def test_get_recovers_after_transient_error():
    c, _ = client(get_server_time=[requests.exceptions.Timeout("t"), ok({"timeSecond": "1700000000",
                                                                         "timeNano": "1700000000123000000"})])
    assert c.server_time_ms() == 1700000000123


def test_post_never_retried_blindly():
    c, sleeps = client(place_order=[requests.exceptions.ConnectionError("x"), ok({"orderId": "1"})])
    with pytest.raises(NetworkError):
        c.place_market("XRPUSDT", 1, "10", "hbd1", stop_loss="1.0")
    assert len(c.s.calls) == 1 and not sleeps
    name, p = c.s.calls[0]
    assert p["stopLoss"] == "1.0" and p["tpslMode"] == "Full" and p["slOrderType"] == "Market"
    assert p["positionIdx"] == 0 and p["orderLinkId"] == "hbd1" and p["orderType"] == "Market"


def test_not_modified_codes_are_ok():
    c, _ = client(set_leverage=[inv(110043, "Set leverage not modified")],
                  set_trading_stop=[inv(34040, "not modified")])
    c.set_leverage("XRPUSDT", 7.5)
    c.set_stop("XRPUSDT", "1.0")
    assert c.s.calls[0][1]["buyLeverage"] == "7.5" == c.s.calls[0][1]["sellLeverage"]


def test_positions_parsing_and_pagination():
    page1 = ok({"list": [{"symbol": "XRPUSDT", "side": "Buy", "size": "10", "avgPrice": "1.5", "stopLoss": "",
                          "liqPrice": "1.2", "leverage": "5", "markPrice": "1.51", "unrealisedPnl": "0.1",
                          "positionIM": "3", "positionStatus": "Normal", "updatedTime": "1"},
                         {"symbol": "ADAUSDT", "side": "", "size": "0", "avgPrice": "0"}],
                "nextPageCursor": "c2"})
    page2 = ok({"list": [{"symbol": "DOGEUSDT", "side": "Sell", "size": "100", "avgPrice": "0.1",
                          "stopLoss": "0.11", "liqPrice": "", "leverage": "10", "markPrice": "0.1"}],
                "nextPageCursor": ""})
    c, _ = client(get_positions=[page1, page2])
    ps = {p.symbol: p for p in c.positions()}
    assert set(ps) == {"XRPUSDT", "DOGEUSDT"}
    assert ps["XRPUSDT"].side == 1 and ps["XRPUSDT"].stop_loss is None and ps["XRPUSDT"].liq_price == 1.2
    assert ps["DOGEUSDT"].side == -1 and ps["DOGEUSDT"].stop_loss == 0.11 and ps["DOGEUSDT"].liq_price is None
    assert c.s.calls[1][1]["cursor"] == "c2"


def test_wallet_falls_back_when_fields_missing():
    c, _ = client(get_wallet_balance=[ok({"list": [{"coin": [{"coin": "USDT", "walletBalance": "25",
                                                              "unrealisedPnl": "-1", "totalPositionIM": "4",
                                                              "totalOrderIM": "0"}]}]})])
    w = c.wallet()
    assert w.equity == 24 and w.available == 21


def test_order_lookup_checks_open_then_history():
    order = {"orderId": "9", "orderLinkId": "hbd1", "symbol": "XRPUSDT", "side": "Sell", "orderType": "Market",
             "qty": "10", "cumExecQty": "10", "avgPrice": "1.49", "orderStatus": "Filled", "cumExecFee": "0.008"}
    c, _ = client(get_open_orders=[ok({"list": []})], get_order_history=[ok({"list": [order]})])
    o = c.order_by_link_id("hbd1")
    assert o.done and o.filled_qty == 10 and o.side == -1 and o.fee == 0.008
    c2, _ = client(get_open_orders=[ok({"list": []})], get_order_history=[ok({"list": []})])
    assert c2.order_by_link_id("nope") is None


def test_executions_split_into_7_day_windows():
    week = 7 * 86_400_000
    c, _ = client(get_executions=[ok({"list": [], "nextPageCursor": ""})] * 3)
    c.executions(0, 2 * week + 5, "XRPUSDT")
    spans = [(p["startTime"], p["endTime"]) for _, p in c.s.calls]
    assert len(spans) == 3 and all(b - a < week for a, b in spans)
    assert spans[0][0] == 0 and spans[-1][1] == 2 * week + 5


def test_isolated_margin_switch_once():
    c, _ = client(get_account_info=[ok({"marginMode": "REGULAR_MARGIN"})], set_margin_mode=[ok({})],
                  switch_position_mode=[inv(110025, "Position mode is not modified")])
    done = c.ensure_isolated_one_way()
    assert done == ["маржа: изолированная"]
    assert c.s.calls[1] == ("set_margin_mode", {"setMarginMode": "ISOLATED_MARGIN"})
