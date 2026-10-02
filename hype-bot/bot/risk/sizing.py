"""Расчёт размера позиции, плеча и цены ликвидации (изолированная маржа).

Правила (из конфига риска):
- размер считается от риска на сделку и расстояния до стопа; в «риск» входят
  комиссии входа и выхода и ожидаемое проскальзывание — потеря на стопе
  не превышает risk_per_trade × капитал;
- объём округляется ВНИЗ до шага, сделки ниже минимума биржи пропускаются
  (риск ради них не увеличивается);
- плечо — максимальное, при котором ликвидация дальше стопа минимум в
  liq_buffer раз (по расстоянию от входа), и не выше максимума биржи для
  уровня риска и собственного лимита (если задан). Большее плечо = меньше
  маржи в позиции = меньше потеря, если цена пролетит сквозь стоп;
- если маржи не хватает, объём уменьшается (риск становится меньше, не больше).

Цена ликвидации — стандартная формула изолированной маржи с запасом на
комиссию закрытия. В бою используется фактическая liqPrice с биржи.
"""
from __future__ import annotations

from dataclasses import dataclass

from bot.market import Instrument

LONG = 1
SHORT = -1


@dataclass(frozen=True)
class SizingParams:
    risk_per_trade: float
    taker_fee: float
    slippage: float            # ожидаемое проскальзывание одной рыночной сделки, доля цены
    liq_buffer: float          # ликвидация дальше стопа минимум во столько раз
    max_leverage: float | None = None


@dataclass(frozen=True)
class Sizing:
    side: int
    qty: float
    leverage: float
    entry_ref: float
    stop: float
    notional: float
    margin: float
    liq_price: float
    planned_loss: float        # потеря на стопе с комиссиями и проскальзыванием, USDT
    risk_budget: float         # risk_per_trade × капитал, USDT
    reduced_by_margin: bool


@dataclass(frozen=True)
class Skip:
    reason: str


def liquidation_price(side: int, entry: float, leverage: float, mmr: float, close_fee: float) -> float:
    """Изолированная маржа без довнесения: позиция теряет IM − MM − комиссия закрытия."""
    if side == LONG:
        return entry * (1 - 1 / leverage + mmr + close_fee)
    return entry * (1 + 1 / leverage - mmr - close_fee)


def max_leverage_for_stop(stop_dist: float, mmr: float, close_fee: float, liq_buffer: float) -> float:
    """Наибольшее плечо, при котором расстояние до ликвидации ≥ liq_buffer × расстояние до стопа."""
    return 1.0 / (liq_buffer * stop_dist + mmr + close_fee)


def size_position(
    side: int,
    entry_ref: float,
    stop: float,
    equity: float,
    available: float,
    inst: Instrument,
    p: SizingParams,
) -> Sizing | Skip:
    if side not in (LONG, SHORT):
        raise ValueError("side должен быть LONG или SHORT")
    if equity <= 0 or available <= 0:
        return Skip("нет капитала")
    # Стоп округляется наружу (дальше от входа): объём считается от реального расстояния.
    stop = inst.round_price(stop, "down" if side == LONG else "up")
    if (side == LONG and stop >= entry_ref) or (side == SHORT and stop <= entry_ref) or stop <= 0:
        return Skip("стоп с неправильной стороны от входа")

    cost = p.taker_fee + p.slippage
    per_unit_loss = abs(entry_ref - stop) + entry_ref * cost + stop * cost
    risk_budget = equity * p.risk_per_trade
    qty = inst.floor_qty(risk_budget / per_unit_loss)

    stop_dist = abs(entry_ref - stop) / entry_ref
    tier = inst.tier_for(qty * entry_ref)
    cap = min(tier.max_leverage, inst.max_leverage)
    if p.max_leverage is not None:
        cap = min(cap, p.max_leverage)
    lev = inst.floor_leverage(min(cap, max_leverage_for_stop(stop_dist, tier.mmr, p.taker_fee, p.liq_buffer)))
    if lev < 1:
        return Skip("стоп слишком далеко: ликвидация не может быть дальше стопа с запасом")

    # Маржа + комиссия входа должны поместиться в доступный баланс.
    reduced = False
    qty_affordable = inst.floor_qty(available / (entry_ref * (1 / lev + p.taker_fee)))
    if qty > qty_affordable:
        qty, reduced = qty_affordable, True

    if qty < inst.min_qty or qty * entry_ref < inst.min_notional:
        return Skip(f"объём ниже минимума биржи (нужно ≥ {inst.min_notional} USDT)")

    notional = qty * entry_ref
    liq = liquidation_price(side, entry_ref, lev, tier.mmr, p.taker_fee)
    # Самопроверка правила «ликвидация дальше стопа с запасом».
    assert abs(entry_ref - liq) >= p.liq_buffer * abs(entry_ref - stop) * (1 - 1e-9), "liq слишком близко"
    return Sizing(
        side=side, qty=qty, leverage=lev, entry_ref=entry_ref, stop=stop,
        notional=notional, margin=notional / lev, liq_price=liq,
        planned_loss=qty * per_unit_loss, risk_budget=risk_budget, reduced_by_margin=reduced,
    )
