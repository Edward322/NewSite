"""Параметры HYPEUSDT, как их вернул API Bybit 2026-10-02 (instrument.json / risk_limit.json)."""
from bot.market import Instrument

HYPE_INSTRUMENT = {
    "symbol": "HYPEUSDT",
    "priceFilter": {"tickSize": "0.010"},
    "lotSizeFilter": {"qtyStep": "0.01", "minOrderQty": "0.01", "minNotionalValue": "5",
                      "maxOrderQty": "73500.00", "maxMktOrderQty": "14700.00"},
    "leverageFilter": {"maxLeverage": "75.00", "leverageStep": "0.01"},
}
HYPE_TIERS = [
    {"riskLimitValue": "5000", "maintenanceMargin": "0.0067", "initialMargin": "0.013", "maxLeverage": "75.00", "mmDeduction": ""},
    {"riskLimitValue": "25000", "maintenanceMargin": "0.01", "initialMargin": "0.02", "maxLeverage": "50.00", "mmDeduction": "16.5"},
    {"riskLimitValue": "50000", "maintenanceMargin": "0.0111", "initialMargin": "0.022", "maxLeverage": "45.00", "mmDeduction": "44"},
]


def hype() -> Instrument:
    return Instrument.from_api(HYPE_INSTRUMENT, HYPE_TIERS)
