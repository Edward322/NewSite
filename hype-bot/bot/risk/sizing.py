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
    risk_budget: float | None = None,
) -> Sizing | Skip:
    """risk_budget — допустимая потеря на стопе в USDT; по умолчанию risk_per_trade × капитал."""
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
    if risk_budget is None:
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


def size_by_margin(side: int, entry_ref: float, stop: float, margin_budget: float, available: float,
                   inst: Instrument, p: SizingParams, stop_rule: str = "none", liq_gap: float = 0.035) -> Sizing | Skip:
    """Объём от доли маржи (правило «маржа = доля баланса»), стоп по stop_rule.

    stop_rule:
      none     — плечо максимальное, стоп стратегии; ликвидация может быть ближе стопа, потеря сделки —
                 вся маржа: planned_loss = маржа + комиссии входа и выхода;
      near_liq — плечо максимальное, стоп переносится к ликвидации: за liq_gap расстояния до ликвидации
                 до неё (если стоп стратегии ближе — остаётся он);
      gap      — стоп стратегии, плечо снижается так, чтобы ликвидация была на liq_gap цены дальше стопа.
    Для near_liq и gap planned_loss — убыток до стопа с издержками, как в size_position.
    """
    if margin_budget <= 0 or available <= 0:
        return Skip("нет капитала")
    if stop_rule not in ("none", "near_liq", "gap"):
        raise ValueError(f"неизвестное правило стопа: {stop_rule}")
    stop = inst.round_price(stop, "down" if side == LONG else "up")
    if (side == LONG and stop >= entry_ref) or (side == SHORT and stop <= entry_ref) or stop <= 0:
        return Skip("стоп с неправильной стороны от входа")
    dist = abs(entry_ref - stop) / entry_ref
    margin = min(margin_budget, available / (1 + 2 * p.taker_fee * inst.max_leverage))
    lev = inst.max_leverage if p.max_leverage is None else min(inst.max_leverage, p.max_leverage)
    for _ in range(8):                      # уровень риска зависит от объёма позиции
        lev = inst.floor_leverage(lev)
        if lev < 1:
            return Skip("ликвидация не помещается за стопом")
        qty = inst.floor_qty(margin * lev / entry_ref)
        tier = inst.tier_for(qty * entry_ref)
        cap = tier.max_leverage
        if stop_rule == "gap":
            cap = min(cap, max_leverage_for_gap(dist, liq_gap, tier.mmr, p.taker_fee))
        if cap >= lev:
            break
        lev = cap
    else:
        return Skip("плечо не согласуется с уровнем риска биржи")
    if qty < inst.min_qty or qty * entry_ref < inst.min_notional:
        return Skip(f"объём ниже минимума биржи (нужно ≥ {inst.min_notional} USDT)")
    notional = qty * entry_ref
    liq = liquidation_price(side, entry_ref, lev, tier.mmr, p.taker_fee)
    if stop_rule == "near_liq":
        stop = tighter_stop(side, stop, stop_near_liq(side, entry_ref, liq, liq_gap, inst))
        if (side == LONG and stop >= entry_ref) or (side == SHORT and stop <= entry_ref):
            return Skip("стоп у ликвидации совпал с ценой входа")
    if stop_rule == "none":
        loss = notional / lev + 2 * notional * p.taker_fee
    else:
        cost = p.taker_fee + p.slippage
        loss = qty * (abs(entry_ref - stop) + entry_ref * cost + stop * cost)
    return Sizing(side=side, qty=qty, leverage=lev, entry_ref=entry_ref, stop=stop, notional=notional,
                  margin=notional / lev, liq_price=liq, planned_loss=loss,
                  risk_budget=margin_budget, reduced_by_margin=margin < margin_budget)


def max_leverage_for_gap(stop_dist: float, gap: float, mmr: float, close_fee: float) -> float:
    """Наибольшее плечо, при котором ликвидация на gap (доля цены) дальше стопа."""
    return 1.0 / (stop_dist + gap + mmr + close_fee)


def stop_near_liq(side: int, entry: float, liq: float, gap: float, inst: Instrument) -> float:
    """Стоп за gap расстояния до ликвидации до неё; округление — к цене входа (стоп не дальше)."""
    raw = entry - side * (1 - gap) * abs(entry - liq)
    return inst.round_price(raw, "up" if side == LONG else "down")


def tighter_stop(side: int, a: float, b: float) -> float:
    return max(a, b) if side == LONG else min(a, b)
