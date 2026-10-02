from bot.config import RiskCfg
from bot.risk.guards import HOUR_MS, RiskGuard, RiskState

CFG = RiskCfg(starting_equity_usdt=10, risk_per_trade=0.05, daily_loss_limit=0.25,
              max_drawdown=0.8, loss_streak_pause_trades=4, loss_streak_pause_hours=24)
DAY0 = 1_767_225_600_000  # 2026-01-01 00:00 UTC
DAY_MS = 24 * HOUR_MS


def guard(equity=10.0, t=DAY0):
    return RiskGuard(CFG, equity, t)


def test_daily_limit_pauses_until_next_utc_day():
    g = guard()
    g.update_equity(DAY0 + 5 * HOUR_MS, 7.6)          # −24% — ещё можно
    assert g.can_open(DAY0 + 5 * HOUR_MS)[0]
    ev = g.update_equity(DAY0 + 6 * HOUR_MS, 7.5)     # −25% — пауза
    assert [e.kind for e in ev] == ["daily_limit"]
    assert not g.can_open(DAY0 + 23 * HOUR_MS)[0]
    g.update_equity(DAY0 + DAY_MS, 7.5)               # новый день: база = 7.5
    assert g.can_open(DAY0 + DAY_MS)[0]
    assert g.state.day_start_equity == 7.5


def test_max_drawdown_from_peak_halts_permanently():
    g = guard()
    g.update_equity(DAY0 + HOUR_MS, 20.0)             # новый пик
    g.update_equity(DAY0 + 2 * DAY_MS, 4.01)          # просадка 79.95% — ещё не остановка
    assert not g.state.halted
    ev = g.update_equity(DAY0 + 3 * DAY_MS, 4.0)      # 20 × (1 − 0.8) = 4
    assert any(e.kind == "max_drawdown" for e in ev)
    g.update_equity(DAY0 + 10 * DAY_MS, 30.0)         # даже рост не снимает остановку
    ok, why = g.can_open(DAY0 + 10 * DAY_MS)
    assert not ok and "остановлен" in why


def test_drawdown_measured_from_peak_not_from_start():
    g = guard()
    g.update_equity(DAY0 + DAY_MS, 50.0)
    g.update_equity(DAY0 + 2 * DAY_MS, 11.0)          # выше старта, но −78% от пика 50
    assert not g.state.halted
    g.update_equity(DAY0 + 3 * DAY_MS, 9.9)           # −80.2% от пика → остановка
    assert g.state.halted


def test_four_losses_in_a_row_pause_24h():
    g = guard()
    t = DAY0 + HOUR_MS
    for i in range(3):
        assert g.on_trade_closed(t + i, -0.1) == []
    assert g.can_open(t + 3)[0]
    ev = g.on_trade_closed(t + 10, -0.1)
    assert ev and ev[0].kind == "streak_pause"
    assert not g.can_open(t + 10 + 23 * HOUR_MS)[0]
    assert g.can_open(t + 10 + 24 * HOUR_MS)[0]
    assert g.state.loss_streak == 0


def test_win_resets_streak():
    g = guard()
    for _ in range(3):
        g.on_trade_closed(DAY0, -0.1)
    g.on_trade_closed(DAY0, 0.05)
    for _ in range(3):
        g.on_trade_closed(DAY0, -0.1)
    assert g.can_open(DAY0)[0]


def test_blockers_prevent_new_positions():
    g = guard()
    g.set_blocker(DAY0, "ws_public", "нет данных 30 с")
    ok, why = g.can_open(DAY0)
    assert not ok and "ws_public" in why
    g.set_blocker(DAY0, "ws_public", None)
    assert g.can_open(DAY0)[0]


def test_manual_halt_and_state_roundtrip():
    g = guard()
    g.halt(DAY0, "аварийная остановка командой kill")
    restored = RiskGuard(CFG, 10, DAY0, RiskState.from_dict(g.state.to_dict()))
    ok, why = restored.can_open(DAY0 + DAY_MS)
    assert not ok and "kill" in why
