import random

import pytest

from bot.risk.sizing import LONG, SHORT, Sizing, SizingParams, Skip, liquidation_price, size_position
from tests.instruments import hype

INST = hype()
P = SizingParams(risk_per_trade=0.05, taker_fee=0.00055, slippage=0.0002, liq_buffer=2.0)


def test_basic_long_matches_hand_calculation():
    s = size_position(LONG, 40.0, 39.2, equity=10, available=10, inst=INST, p=P)
    assert isinstance(s, Sizing)
    # потеря на единицу: 0.8 + 40*0.00075 + 39.2*0.00075 = 0.8594 → 0.5/0.8594 = 0.5818 → вниз до 0.58
    assert s.qty == 0.58
    assert s.planned_loss == pytest.approx(0.58 * 0.8594)
    assert s.planned_loss <= s.risk_budget
    # плечо: 1/(2*0.02 + 0.0067 + 0.00055) = 21.164 → вниз до шага 0.01
    assert s.leverage == 21.16
    assert abs(s.entry_ref - s.liq_price) >= 2 * abs(s.entry_ref - s.stop)


def test_short_is_symmetric():
    s = size_position(SHORT, 40.0, 40.8, equity=10, available=10, inst=INST, p=P)
    assert isinstance(s, Sizing) and s.side == SHORT
    assert s.liq_price > s.stop > s.entry_ref
    assert s.liq_price - s.entry_ref >= 2 * (s.stop - s.entry_ref)


def test_stop_is_rounded_outward_to_tick():
    s_long = size_position(LONG, 40.0, 39.237, 10, 10, INST, P)
    s_short = size_position(SHORT, 40.0, 40.761, 10, 10, INST, P)
    assert s_long.stop == 39.23 and s_short.stop == 40.77


@pytest.mark.parametrize("side,stop", [(LONG, 40.5), (LONG, 40.0), (SHORT, 39.5), (LONG, -1.0)])
def test_wrong_side_stop_is_rejected(side, stop):
    assert isinstance(size_position(side, 40.0, stop, 10, 10, INST, P), Skip)


def test_below_exchange_minimum_is_skipped_not_upsized():
    # капитал 2 USDT, стоп 5% → риск 0.1 USDT → позиция ~1.9 USDT < 5 USDT
    r = size_position(LONG, 40.0, 38.0, equity=2, available=2, inst=INST, p=P)
    assert isinstance(r, Skip) and "минимума" in r.reason


def test_leverage_capped_by_exchange_and_by_config():
    tight = size_position(LONG, 40.0, 39.96, 10, 10, INST, P)       # стоп 0.1%
    assert tight.leverage == 75.0
    capped = size_position(LONG, 40.0, 39.96, 10, 10, INST,
                           SizingParams(0.05, 0.00055, 0.0002, 2.0, max_leverage=20))
    assert capped.leverage == 20.0


def test_margin_shortage_reduces_size_never_increases_risk():
    p = SizingParams(0.05, 0.00055, 0.0002, 2.0, max_leverage=10)
    s = size_position(LONG, 40.0, 39.96, equity=10, available=10, inst=INST, p=p)
    assert s.reduced_by_margin
    assert s.margin + s.notional * p.taker_fee <= 10 + 1e-9
    assert s.planned_loss < s.risk_budget


def test_huge_stop_cannot_keep_liquidation_beyond_it():
    r = size_position(LONG, 40.0, 10.0, 10, 10, INST, P)   # стоп 75%
    assert isinstance(r, Skip) and "ликвидация" in r.reason


def test_higher_tier_uses_its_mmr_and_max_leverage():
    s = size_position(LONG, 40.0, 39.96, equity=2_000, available=2_000, inst=INST, p=P)
    assert s.notional > 5_000
    assert s.leverage <= 50.0


def test_liquidation_formula():
    assert liquidation_price(LONG, 100, 10, 0.0067, 0.00055) == pytest.approx(100 * (1 - 0.1 + 0.00725))
    assert liquidation_price(SHORT, 100, 10, 0.0067, 0.00055) == pytest.approx(100 * (1 + 0.1 - 0.00725))


def test_invariants_on_random_inputs():
    rng = random.Random(7)
    checked = 0
    for _ in range(3000):
        side = rng.choice([LONG, SHORT])
        entry = rng.uniform(5, 200)
        dist = rng.uniform(0.0005, 0.3)
        stop = entry * (1 - dist) if side == LONG else entry * (1 + dist)
        equity = rng.uniform(1, 500)
        avail = equity * rng.uniform(0.2, 1.0)
        r = size_position(side, entry, stop, equity, avail, INST, P)
        if isinstance(r, Skip):
            continue
        checked += 1
        assert r.planned_loss <= r.risk_budget * (1 + 1e-9)            # риск не превышен
        assert r.notional >= INST.min_notional and r.qty >= INST.min_qty
        assert abs(r.entry_ref - r.liq_price) >= 2 * abs(r.entry_ref - r.stop) * (1 - 1e-9)
        assert 1 <= r.leverage <= 75
        assert r.margin + r.notional * P.taker_fee <= avail * (1 + 1e-9)
        assert round(r.qty / 0.01, 6).is_integer()
    assert checked > 1000
