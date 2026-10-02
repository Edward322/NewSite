"""Параметры инструмента Bybit и точное округление цены/объёма.

Источник — ответы /v5/market/instruments-info и /v5/market/risk-limit
(сохраняются загрузчиком в instrument.json / risk_limit.json).
Округление идёт через Decimal: шаги вроде 0.01 не представимы точно в float.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from pathlib import Path

from bot.data import store


@dataclass(frozen=True)
class RiskTier:
    limit_value: float   # максимальная стоимость позиции для уровня, USDT
    mmr: float           # ставка поддерживающей маржи
    imr: float           # ставка начальной маржи
    max_leverage: float
    mm_deduction: float


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: Decimal
    qty_step: Decimal
    min_qty: float
    min_notional: float
    max_leverage: float
    leverage_step: Decimal
    max_mkt_qty: float
    tiers: tuple[RiskTier, ...]

    # --- построение ---
    @classmethod
    def from_api(cls, inst: dict, tiers: list[dict]) -> "Instrument":
        lot, pf, lev = inst["lotSizeFilter"], inst["priceFilter"], inst["leverageFilter"]
        parsed = sorted(
            (RiskTier(limit_value=float(t["riskLimitValue"]),
                      mmr=float(t["maintenanceMargin"]),
                      imr=float(t["initialMargin"]),
                      max_leverage=float(t["maxLeverage"]),
                      mm_deduction=float(t.get("mmDeduction") or 0))
             for t in tiers),
            key=lambda t: t.limit_value,
        )
        if not parsed:
            raise ValueError("Нет уровней лимита риска")
        return cls(
            symbol=inst["symbol"],
            tick_size=Decimal(pf["tickSize"]),
            qty_step=Decimal(lot["qtyStep"]),
            min_qty=float(lot["minOrderQty"]),
            min_notional=float(lot.get("minNotionalValue") or 0),
            max_leverage=float(lev["maxLeverage"]),
            leverage_step=Decimal(lev.get("leverageStep") or "0.01"),
            max_mkt_qty=float(lot.get("maxMktOrderQty") or lot["maxOrderQty"]),
            tiers=tuple(parsed),
        )

    @classmethod
    def from_files(cls, data_dir: Path, symbol: str) -> "Instrument":
        inst = store.read_json(store.json_path(data_dir, symbol, "instrument"))
        tiers = store.read_json(store.json_path(data_dir, symbol, "risk_limit"))
        if inst is None or tiers is None:
            raise FileNotFoundError(f"Нет параметров инструмента {symbol} в {data_dir}")
        return cls.from_api(inst["instrument"], tiers["tiers"])

    # --- уровни риска ---
    def tier_for(self, position_value: float) -> RiskTier:
        for t in self.tiers:
            if position_value <= t.limit_value:
                return t
        return self.tiers[-1]

    # --- округление ---
    @staticmethod
    def _to_step(x: float, step: Decimal, rounding: str) -> Decimal:
        units = (Decimal(repr(float(x))) / step).to_integral_value(rounding=rounding)
        return units * step

    def round_price(self, price: float, mode: str = "nearest") -> float:
        rounding = {"nearest": ROUND_HALF_UP, "down": ROUND_FLOOR, "up": ROUND_CEILING}[mode]
        return float(self._to_step(price, self.tick_size, rounding))

    def floor_qty(self, qty: float) -> float:
        if qty <= 0 or not math.isfinite(qty):
            return 0.0
        return float(self._to_step(qty, self.qty_step, ROUND_FLOOR))

    def floor_leverage(self, lev: float) -> float:
        return float(self._to_step(lev, self.leverage_step, ROUND_FLOOR))

    def fmt_price(self, price: float) -> str:
        return str(self._to_step(price, self.tick_size, ROUND_HALF_UP))

    def fmt_qty(self, qty: float) -> str:
        return str(self._to_step(qty, self.qty_step, ROUND_FLOOR))
