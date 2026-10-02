"""Интерфейс стратегии — общий для backtest, demo и live.

Стратегия видит только закрытые свечи своего таймфрейма и отвечает решением:
войти (со стопом), выйти, подтянуть стоп или ничего. Размер позиции, плечо и
все ограничения риска считает НЕ стратегия, а риск-модуль.

prepare(bars) вызывается на всей доступной истории и может считать индикаторы
векторно, но обязан быть причинным: значение на свече i зависит только от
свечей 0..i. Это проверяется автоматически (bot/backtest/causality.py).
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.data.bars import Bars

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class Enter:
    side: int
    stop: float
    take_profit: float | None = None
    tag: str = ""


@dataclass(frozen=True)
class Exit:
    reason: str = "signal"


@dataclass(frozen=True)
class MoveStop:
    stop: float     # применяется, только если уменьшает риск (стоп нельзя отодвинуть)


Decision = Enter | Exit | MoveStop | None


@dataclass(frozen=True)
class PositionView:
    side: int
    entry_price: float
    qty: float
    stop: float
    take_profit: float | None
    entry_ts: int
    bars_held: int


class Strategy:
    name: str = "base"
    timeframe: str = "1h"
    warmup_bars: int = 0

    def params(self) -> dict:
        return {}

    def prepare(self, bars: Bars) -> None:
        raise NotImplementedError

    def on_bar(self, i: int, pos: PositionView | None) -> Decision:
        raise NotImplementedError
