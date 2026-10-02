"""Поддельная сессия pybit, имитирующая контракт ответов Bybit V5 по документации."""
from __future__ import annotations

import math

MIN = 60_000


class FakeBybit:
    def __init__(self, launch_ms: int, now_ms: int, missing: set[int] | None = None,
                 funding_every_h: int = 8):
        self.launch_ms = launch_ms
        self.now_ms = now_ms
        self.missing = missing or set()
        self.funding_every = funding_every_h * 3_600_000
        self.calls: list[tuple[str, dict]] = []

    # --- helpers ---
    def price(self, ts: int) -> float:
        return 20 + 5 * math.sin(ts / (MIN * 600))

    def _candles(self, start, end, limit, mark=False, interval="1"):
        step = int(interval) * MIN
        first = max(start, self.launch_ms)
        first = -(-first // step) * step
        current_open = (self.now_ms // step) * step  # формирующаяся свеча тоже отдаётся, как на бирже
        out = []
        t = first
        while t <= min(end, current_open):
            if t not in self.missing:
                p = self.price(t)
                row = [str(t), str(p), str(p * 1.001), str(p * 0.999), str(p * 1.0005)]
                if not mark:
                    row += ["100", str(100 * p)]
                out.append(row)
            t += step
        out = out[-limit:] if len(out) > limit else out
        return list(reversed(out))  # Bybit: сортировка по убыванию startTime

    def _ok(self, result):
        return {"retCode": 0, "retMsg": "OK", "result": result, "time": self.now_ms}

    # --- pybit API surface ---
    def get_server_time(self):
        self.calls.append(("time", {}))
        return self._ok({"timeSecond": str(self.now_ms // 1000)})

    def get_kline(self, **p):
        self.calls.append(("kline", p))
        assert p["limit"] <= 1000
        return self._ok({"symbol": p["symbol"], "category": p["category"],
                         "list": self._candles(p["start"], p["end"], p["limit"], interval=p["interval"])})

    def get_mark_price_kline(self, **p):
        self.calls.append(("mark", p))
        assert p["limit"] <= 1000
        return self._ok({"symbol": p["symbol"], "category": p["category"],
                         "list": self._candles(p["start"], p["end"], p["limit"], mark=True,
                                               interval=p["interval"])})

    def get_funding_rate_history(self, **p):
        self.calls.append(("funding", p))
        assert "endTime" in p, "только startTime — ошибка по документации"
        assert p["limit"] <= 200
        first = -(-max(p["startTime"], self.launch_ms) // self.funding_every) * self.funding_every
        ts = list(range(first, min(p["endTime"], self.now_ms) + 1, self.funding_every))
        page = list(reversed(ts))[: p["limit"]]
        return self._ok({"category": "linear", "list": [
            {"symbol": p["symbol"], "fundingRate": f"{0.0001 * ((t // self.funding_every) % 3 - 1):.6f}",
             "fundingRateTimestamp": str(t)} for t in page]})

    def get_instruments_info(self, **p):
        self.calls.append(("instrument", p))
        return self._ok({"category": "linear", "list": [{
            "symbol": p["symbol"], "status": "Trading", "launchTime": str(self.launch_ms),
            "priceFilter": {"tickSize": "0.001"},
            "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.01", "minNotionalValue": "5"},
            "leverageFilter": {"maxLeverage": "50"}, "fundingInterval": 480,
            "lowerFundingRate": "-0.02", "upperFundingRate": "0.02"}]})

    def get_risk_limit(self, **p):
        self.calls.append(("risk", p))
        if p.get("cursor") is None:
            return self._ok({"category": "linear", "nextPageCursor": "page2",
                             "list": [{"id": 1, "symbol": p["symbol"], "maintenanceMargin": "0.01"}]})
        return self._ok({"category": "linear", "nextPageCursor": "",
                         "list": [{"id": 2, "symbol": p["symbol"], "maintenanceMargin": "0.02"}]})


H = 3_600_000


class FakeBybitBasket(FakeBybit):
    """+ эндпоинты для корзины: список инструментов, тикеры, OI, лонг/шорт, премиальный индекс."""

    def __init__(self, launch_ms, now_ms, instruments=None, tickers=None, fail=()):
        super().__init__(launch_ms, now_ms)
        self.instruments, self.tickers, self.fail = instruments or [], tickers or [], set(fail)

    def get_instruments_info(self, **p):
        if "symbol" in p:
            return super().get_instruments_info(**p)
        self.calls.append(("instruments_all", p))
        if p.get("cursor") is None:
            return self._ok({"list": self.instruments[:2], "nextPageCursor": "p2" if len(self.instruments) > 2 else ""})
        return self._ok({"list": self.instruments[2:], "nextPageCursor": ""})

    def get_tickers(self, **p):
        return self._ok({"list": self.tickers})

    def _hourly(self, name, p, start_key, end_key, make):
        if name in self.fail:
            return {"retCode": 10001, "retMsg": "params error", "result": {}}
        self.calls.append((name, p))
        first = -(-max(p[start_key], self.launch_ms) // H) * H
        ts = list(range(first, min(p[end_key], self.now_ms) + 1, H))
        ts = list(reversed(ts))[: p["limit"]]
        return self._ok({"list": [make(t) for t in ts], "nextPageCursor": ""})

    def get_open_interest(self, **p):
        return self._hourly("oi", p, "startTime", "endTime",
                            lambda t: {"openInterest": str(1000 + t // H % 50), "timestamp": str(t)})

    def get_long_short_ratio(self, **p):
        return self._hourly("lsr", p, "startTime", "endTime",
                            lambda t: {"symbol": p["symbol"], "buyRatio": "0.55", "sellRatio": "0.45", "timestamp": str(t)})

    def get_premium_index_price_kline(self, **p):
        self.calls.append(("premium", p))
        first = -(-max(p["start"], self.launch_ms) // (60 * MIN)) * (60 * MIN)
        cur_open = self.now_ms // (60 * MIN) * (60 * MIN)
        ts = [t for t in range(first, min(p["end"], cur_open) + 1, 60 * MIN)][-p["limit"]:]
        return self._ok({"list": [[str(t), "0.0001", "0.0002", "-0.0001", "0.00005"] for t in reversed(ts)]})
