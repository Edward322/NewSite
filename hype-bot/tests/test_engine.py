import numpy as np
import pytest

from bot.config import RiskCfg
from bot.strategy.base import LONG, SHORT, Enter, Exit, MoveStop
from tests.helpers import MIN, T0, flat, md_from_rows, run
from tests.strategies import Scripted

SLIP, TAKER, MAKER, PEN = 0.0002, 0.00055, 0.0002, 0.25
FILL_IN = 40 * (1 + SLIP)                    # вход по открытию минуты 5


def rows_with(n, overrides, base=40.0):
    rows = flat(n, base)
    rows = list(rows)
    for i, r in overrides.items():
        rows[i] = r
    return rows


def one_trade(rows, script, **kw):
    res = run(md_from_rows(rows, **({"mark_rows": kw.pop("mark_rows")} if "mark_rows" in kw else {})),
              Scripted(script), **kw)
    return res, (res.trades.iloc[0] if len(res.trades) else None)


def test_entry_next_minute_open_and_market_exit_with_fees():
    res, t = one_trade(flat(40), {0: Enter(LONG, 39.2), 2: Exit()})
    assert t.decision_ts == T0 + 5 * MIN and t.entry_ts == T0 + 5 * MIN   # решение на закрытии, вход не раньше
    assert t.entry_price == pytest.approx(FILL_IN)
    assert t.qty == 0.58
    assert t.exit_ts == T0 + 15 * MIN and t.exit_price == pytest.approx(40 * (1 - SLIP))
    expected = 0.58 * (40 * (1 - SLIP) - FILL_IN) - 0.58 * FILL_IN * TAKER - 0.58 * 40 * (1 - SLIP) * TAKER
    assert t.net_pnl == pytest.approx(expected)
    assert res.final_equity == pytest.approx(10 + expected)
    assert abs(t.entry_price - t.liq_price) >= 2 * abs(t.entry_price - t.stop_initial)


def test_stop_fill_includes_penetration_and_slippage():
    rows = rows_with(40, {10: (39.9, 39.95, 38.8, 39.0)}, base=40.0)
    rows[11:] = flat(29, 39.0)
    _, t = one_trade(rows, {0: Enter(LONG, 39.2)})
    assert t.exit_reason == "stop" and t.exit_ts == T0 + 10 * MIN
    assert t.exit_price == pytest.approx((39.2 - PEN * (39.2 - 38.8)) * (1 - SLIP))
    assert t.exit_fee == pytest.approx(t.qty * t.exit_price * TAKER)


def test_gap_through_stop_fills_from_open():
    rows = rows_with(40, {10: (39.0, 39.05, 38.9, 38.95)})
    rows[11:] = flat(29, 39.0)
    _, t = one_trade(rows, {0: Enter(LONG, 39.2)})
    assert t.exit_price == pytest.approx((39.0 - PEN * (39.0 - 38.9)) * (1 - SLIP))


def test_short_stop_is_mirrored():
    rows = rows_with(40, {10: (40.1, 41.2, 40.05, 41.0)})
    rows[11:] = flat(29, 41.0)
    _, t = one_trade(rows, {0: Enter(SHORT, 40.8)})
    assert t.side == SHORT and t.exit_reason == "stop"
    assert t.exit_price == pytest.approx((40.8 + PEN * (41.2 - 40.8)) * (1 + SLIP))


def test_take_profit_needs_trade_through_and_pays_maker():
    rows = rows_with(40, {10: (40.5, 41.0, 40.4, 40.9), 11: (40.9, 41.01, 40.8, 41.0)})
    rows[12:] = flat(28, 41.0)
    _, t = one_trade(rows, {0: Enter(LONG, 39.2, take_profit=41.0)})
    assert t.exit_reason == "take_profit" and t.exit_ts == T0 + 11 * MIN
    assert t.exit_price == 41.0
    assert t.exit_fee == pytest.approx(t.qty * 41.0 * MAKER)


def test_stop_and_tp_in_same_minute_assumes_stop_first():
    rows = rows_with(40, {10: (40.5, 41.5, 39.0, 40.0)})
    _, t = one_trade(rows, {0: Enter(LONG, 39.2, take_profit=41.0)})
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx((39.2 - PEN * 0.2) * (1 - SLIP))


def test_gap_above_tp_fills_tp_even_if_minute_also_hits_stop():
    rows = rows_with(40, {10: (41.5, 41.6, 39.0, 40.0)})
    _, t = one_trade(rows, {0: Enter(LONG, 39.2, take_profit=41.0)})
    assert t.exit_reason == "take_profit" and t.exit_price == 41.0


def test_liquidation_by_mark_price_loses_whole_margin():
    rows = rows_with(40, {10: (40.0, 40.0, 39.5, 39.6)})       # последняя цена до стопа не дошла
    mark = rows_with(40, {10: (40.0, 40.0, 38.0, 39.6)})       # а mark-цена прошла ликвидацию
    res, t = one_trade(rows, {0: Enter(LONG, 39.2)}, mark_rows=mark)
    assert t.exit_reason == "liquidation"
    assert t.gross_pnl == pytest.approx(-t.margin)
    assert t.net_pnl == pytest.approx(-t.margin - t.entry_fee)


def test_gap_beyond_liquidation_is_capped_at_margin():
    rows = rows_with(40, {10: (37.0, 37.1, 36.5, 36.8)})
    rows[11:] = flat(29, 36.8)
    _, t = one_trade(rows, {0: Enter(LONG, 39.2)})
    assert t.exit_reason == "liquidation"
    assert t.gross_pnl == pytest.approx(-t.margin)


def test_funding_paid_on_mark_price_only_if_held_at_timestamp():
    t_start = T0 - 30 * MIN                       # 23:30; выплата в 00:00 = минута 30
    rate = 0.001
    f = (np.array([T0], dtype="int64"), np.array([rate]))
    mark = rows_with(60, {30: (40.1, 40.1, 40.1, 40.1)})
    # держим через 00:00
    res = run(md_from_rows(flat(60), t0=t_start, mark_rows=mark), Scripted({0: Enter(LONG, 39.2), 8: Exit()}), funding=f)
    t = res.trades.iloc[0]
    assert t.funding == pytest.approx(t.qty * 40.1 * rate)
    # вход ровно в момент выплаты — не платим
    res = run(md_from_rows(flat(60), t0=t_start, mark_rows=mark), Scripted({5: Enter(LONG, 39.2), 8: Exit()}), funding=f)
    assert res.trades.iloc[0].entry_ts == T0 and res.trades.iloc[0].funding == 0
    # выход ровно в момент выплаты — платим (позиция была до неё)
    res = run(md_from_rows(flat(60), t0=t_start, mark_rows=mark), Scripted({0: Enter(LONG, 39.2), 5: Exit()}), funding=f)
    assert res.trades.iloc[0].exit_ts == T0 and res.trades.iloc[0].funding == pytest.approx(res.trades.iloc[0].qty * 40.1 * rate)
    # шорт при положительной ставке получает
    res = run(md_from_rows(flat(60), t0=t_start, mark_rows=mark), Scripted({0: Enter(SHORT, 40.8), 8: Exit()}), funding=f)
    assert res.trades.iloc[0].funding == pytest.approx(-res.trades.iloc[0].qty * 40.1 * rate)


def test_funding_inside_a_long_bar_is_charged_at_its_minute():
    day = 1440
    f_ts = np.array([T0 + h * 3_600_000 for h in (0, 8, 16, 24, 32, 40, 48)], dtype="int64")
    f = (f_ts, np.full(len(f_ts), 0.0005))
    res = run(md_from_rows(flat(3 * day)), Scripted({0: Enter(LONG, 39.2), 1: Exit()}, timeframe="1d"), funding=f)
    t = res.trades.iloc[0]
    # вход в 24:00 после выплаты; выплаты 32:00, 40:00 внутри свечи и 48:00 перед выходом
    assert t.entry_ts == T0 + day * MIN and t.exit_ts == T0 + 2 * day * MIN
    assert t.funding == pytest.approx(3 * t.qty * 40 * 0.0005)


def test_below_minimum_is_skipped_without_upsizing():
    res = run(md_from_rows(flat(40)), Scripted({0: Enter(LONG, 38.0)}), equity=2.0)
    assert len(res.trades) == 0
    assert res.skips and "минимума" in res.skips[0][1]


def test_move_stop_only_tightens():
    rows = rows_with(60, {22: (39.6, 39.6, 39.4, 39.45)})
    rows[23:] = flat(37, 39.45)
    res, t = one_trade(rows, {0: Enter(LONG, 39.2), 2: MoveStop(39.0), 3: MoveStop(39.5)})
    assert t.stop_final == 39.5 and t.exit_reason == "stop" and t.exit_ts == T0 + 22 * MIN


def test_open_position_closed_at_end_of_data():
    _, t = one_trade(flat(40), {0: Enter(LONG, 39.2)})
    assert t.exit_reason == "end"


def test_no_entries_before_trade_start():
    res = run(md_from_rows(flat(60)), Scripted({0: Enter(LONG, 39.2), 4: Enter(LONG, 39.2)}),
              trade_start_ms=T0 + 20 * MIN)
    assert len(res.trades) == 1 and res.trades.iloc[0].entry_ts == T0 + 25 * MIN


def _stopouts(n_trades, per_day_minutes=None, total=None):
    """Каждые 20 минут: вход на свече k, провал на минуте через 10 минут."""
    total = total or n_trades * 20 + 40
    rows = list(flat(total))
    script = {}
    for j in range(n_trades):
        k = 4 * j                                 # свечи по 5 минут
        script[k] = Enter(LONG, 39.2)
        rows[5 * k + 10] = (40.0, 40.0, 38.9, 40.0)
    return rows, script


def test_daily_loss_limit_pauses_rest_of_day():
    risk = RiskCfg(starting_equity_usdt=10, risk_per_trade=0.2, daily_loss_limit=0.25, max_drawdown=0.8,
                   loss_streak_pause_trades=10, loss_streak_pause_hours=24)
    rows, script = _stopouts(3)
    res = run(md_from_rows(rows), Scripted(script), risk=risk)
    assert len(res.trades) == 2                    # −20%, затем ≈−36% → пауза
    assert any(e.kind == "daily_limit" for e in res.events)
    assert any("дневной" in why for _, why in res.skips)


def test_four_losses_trigger_24h_pause():
    rows, script = _stopouts(5)
    res = run(md_from_rows(rows), Scripted(script))
    assert len(res.trades) == 4
    assert any(e.kind == "streak_pause" for e in res.events)
    assert any("серия" in why for _, why in res.skips)


def test_max_drawdown_halts_trading_for_good():
    risk = RiskCfg(starting_equity_usdt=10, risk_per_trade=0.2, daily_loss_limit=0.5, max_drawdown=0.5,
                   loss_streak_pause_trades=10, loss_streak_pause_hours=24)
    rows, script = _stopouts(6)
    res = run(md_from_rows(rows), Scripted(script), risk=risk)
    assert res.halted
    assert any(e.kind == "max_drawdown" for e in res.events)
    assert len(res.trades) < 6
    assert res.final_equity > 0


def test_intrabar_drawdown_floor_closes_position_before_stop():
    # порог просадки 40%: после двух стопов капитал ≈ 6.13 при пороге 6.0
    risk = RiskCfg(starting_equity_usdt=10, risk_per_trade=0.2, daily_loss_limit=0.4, max_drawdown=0.4,
                   loss_streak_pause_trades=10, loss_streak_pause_hours=24)
    rows, script = _stopouts(2)
    rows += list(flat(60))
    script[8] = Enter(LONG, 39.2)
    rows[5 * 8 + 10] = (40.0, 40.0, 39.8, 39.9)   # стоп 39.2 не задет
    res = run(md_from_rows(rows), Scripted(script), risk=risk)
    assert len(res.trades) == 3
    last = res.trades.iloc[-1]
    assert last.exit_reason == "max_drawdown" and last.exit_price > 39.2
    assert res.halted
    assert [e.kind for e in res.events].count("max_drawdown") == 1
    # капитал у порога (с поправкой на проскальзывание), а не на полном стопе
    assert res.final_equity == pytest.approx(6.0, rel=0.02)


def test_rounding_of_qty_and_prices():
    res = run(md_from_rows(flat(40, 40.0)), Scripted({0: Enter(LONG, 39.237), 2: Exit()}))
    t = res.trades.iloc[0]
    assert t.stop_initial == 39.23
    assert round(t.qty / 0.01, 9).is_integer()
