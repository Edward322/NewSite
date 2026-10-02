"""Движок на имитаторе биржи: совпадение с бэктестом, перезапуск, сбои связи, сверка, остановки."""
import numpy as np
import pandas as pd
import pytest

from bot.data.panel import DAY_MS
from tests.engine_env import make_env as _make_env, sim_config  # noqa: F401
from tests.helpers import T0
from tests.test_basket_strategies import synth_panel


def _zero_funding_panel():
    p = synth_panel(seed=0)
    for sd in p.sym.values():
        sd.funding_rate = np.zeros_like(sd.funding_rate)
    return p


PANEL0 = _zero_funding_panel()


def make_env(tmp, **kw):
    """Без финансирования: движок и бэктест должны совпасть до последнего знака. С финансированием
    движок видит капитал уже после выплаты в момент закрытия свечи, а бэктест — до неё (см. отдельный тест)."""
    kw.setdefault("panel", PANEL0)
    return _make_env(tmp, **kw)

START_DAY, MID_DAY, END_DAY = 108, 125, 150


def _trades(env):
    return pd.DataFrame(env.db.trades())


def _compare(eng_tr: pd.DataFrame, bt_tr: pd.DataFrame):
    key = ["symbol", "side", "decision_ts"]
    a = eng_tr.set_index(key).sort_index()
    b = bt_tr.set_index(key).sort_index()
    return a, b


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("full")
    env = make_env(tmp, start_day=START_DAY)
    assert env.engine.start()
    first_bar = int(env.db.get("last_bar_close")) + 1
    env.run_until(T0 + END_DAY * DAY_MS)
    return env, first_bar


def test_engine_matches_backtest_trade_for_trade(full_run):
    env, first_bar = full_run
    tr = _trades(env)
    bt = env.backtest(first_bar)
    assert len(tr) >= 8, len(tr)
    a, b = _compare(tr, bt.trades)
    assert list(a.index) == list(b.index)                       # те же сигналы
    np.testing.assert_allclose(a["qty"], b["qty"], rtol=1e-12)    # те же объёмы
    np.testing.assert_allclose(a["entry_price"], b["entry_price"], rtol=1e-12)
    assert list(a["exit_reason"]) == list(b["exit_reason"])       # те же выходы
    np.testing.assert_allclose(a["exit_price"], b["exit_price"], rtol=1e-12)
    np.testing.assert_allclose(a["net_pnl"], b["net_pnl"], rtol=1e-9, atol=1e-9)
    m15 = 15 * 60_000                                             # выход на той же 15m-свече
    assert list(a["exit_ts"] // m15) == list(b["exit_ts"] // m15)
    # капитал бота совпадает с бэктестом
    eq, _ = env.engine.equity()
    open_pos = env.db.positions()
    assert not env.db.events(levels=("error",))
    if not open_pos:
        assert eq == pytest.approx(bt.final_equity, rel=1e-9)


def test_every_position_had_exchange_stop(full_run):
    env, _ = full_run
    assert not env.db.events(kinds=("mismatch",))
    stop_events = env.db.events(kinds=("stop",))
    assert stop_events == []                                     # стоп ни разу не пропадал
    tr = _trades(env)
    assert set(tr["exit_reason"]) <= {"stop", "channel_exit"}
    # каждый вход был отправлен вместе со стопом
    orders = [o for o in env.db.orders() if o.purpose == "entry" and o.status == "filled"]
    assert orders and all(o.stop for o in orders)
    assert len({o.link_id for o in env.db.orders()}) == len(env.db.orders())


def test_restart_mid_run_gives_same_trades(tmp_path, full_run):
    ref_env, first_bar = full_run
    env = make_env(tmp_path, start_day=START_DAY)
    assert env.engine.start()
    assert int(env.db.get("last_bar_close")) + 1 == first_bar
    env.run_until(T0 + MID_DAY * DAY_MS)
    assert env.db.positions(), "к моменту перезапуска должны быть открытые позиции"
    env.new_engine()
    assert env.engine.start()
    env.run_until(T0 + END_DAY * DAY_MS)
    a, b = _compare(_trades(env), _trades(ref_env))
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(a["net_pnl"], b["net_pnl"], rtol=1e-9, atol=1e-9)


# ------------------------------------------------------------------ сбои
def _first_entry(ref_env):
    tr = _trades(ref_env).sort_values("decision_ts")
    return tr.iloc[0]


def test_lost_order_response_is_recovered_without_double_entry(tmp_path, full_run):
    ref_env, _ = full_run
    t = _first_entry(ref_env)
    env = make_env(tmp_path, start_day=START_DAY)
    assert env.engine.start()
    env.run_until(int(t.decision_ts) - 60_000)
    env.sim.fail_next["place_order"] = "after"          # ордер исполнен, ответ потерян
    env.run_until(int(t.decision_ts) + 3_600_000)
    entries = [o for o in env.db.orders() if o.purpose == "entry" and o.symbol == t.symbol]
    assert len(entries) == 1 and entries[0].status == "filled"
    assert sum(1 for o in env.sim.orders if o["orderLinkId"] == entries[0].link_id) == 1
    assert t.symbol in env.db.positions() and t.symbol in env.sim.pos
    env.run_until(T0 + END_DAY * DAY_MS)
    a, b = _compare(_trades(env), _trades(ref_env))
    assert list(a.index) == list(b.index)
    np.testing.assert_allclose(a["net_pnl"], b["net_pnl"], rtol=1e-9, atol=1e-9)


def test_request_lost_before_exchange_skips_entry_safely(tmp_path, full_run):
    ref_env, _ = full_run
    t = _first_entry(ref_env)
    env = make_env(tmp_path, start_day=START_DAY)
    assert env.engine.start()
    env.run_until(int(t.decision_ts) - 60_000)
    env.sim.fail_next["place_order"] = "before"
    env.run_until(int(t.decision_ts) + 3_600_000)
    failed = [o for o in env.db.orders() if o.status == "failed"]
    assert len(failed) == 1 and failed[0].purpose == "entry"
    assert failed[0].symbol not in env.db.positions() and failed[0].symbol not in env.sim.pos
    assert not any(o["orderLinkId"] == failed[0].link_id for o in env.sim.orders)
    assert env.engine.reconcile()                        # расхождений нет


def test_reprocessing_same_bar_does_not_duplicate_orders(tmp_path, full_run):
    ref_env, _ = full_run
    t = _first_entry(ref_env)
    env = make_env(tmp_path, start_day=START_DAY)
    assert env.engine.start()
    env.run_until(int(t.decision_ts) + 600_000)
    n_orders = len(env.sim.orders)
    env.db.set("last_bar_close", int(t.decision_ts) - 4 * 3_600_000)   # как будто свеча не отмечена
    env.engine.process_bar(int(t.decision_ts))
    assert len(env.sim.orders) == n_orders
    assert len(env.sim.pos) == len(env.db.positions())


def _with_position(tmp_path, ref_env):
    t = _first_entry(ref_env)
    env = make_env(tmp_path, start_day=START_DAY)
    assert env.engine.start()
    env.run_until(int(t.decision_ts) + 600_000)
    assert t.symbol in env.sim.pos
    return env, t.symbol


def test_missing_exchange_stop_is_restored(tmp_path, full_run):
    env, sym = _with_position(tmp_path, full_run[0])
    env.sim.drop_stop(sym)
    env.engine.reconcile()
    assert env.sim.pos[sym].stop == env.db.positions()[sym].stop
    assert any("стоп" in e["message"] for e in env.db.events(kinds=("stop",)))


def test_position_closed_if_stop_cannot_be_set(tmp_path, full_run):
    env, sym = _with_position(tmp_path, full_run[0])
    env.sim.drop_stop(sym)
    env.sim.reject_next["set_trading_stop"] = (10001, "внутренняя ошибка")
    env.engine.reconcile()
    assert sym not in env.sim.pos and sym not in env.db.positions()
    tr = _trades(env)
    assert tr.iloc[-1]["exit_reason"] == "no_stop"


def test_manual_close_on_exchange_is_recorded(tmp_path, full_run):
    env, sym = _with_position(tmp_path, full_run[0])
    env.sim.close_externally(sym)
    env.engine.reconcile()
    assert sym not in env.db.positions()
    assert _trades(env).iloc[-1]["exit_reason"] == "external"
    assert env.engine.guard.can_open(env.clock.now_ms())[0]


def test_foreign_position_blocks_new_entries(tmp_path, full_run):
    from bot.exchange.sim import SimPos
    env, sym = _with_position(tmp_path, full_run[0])
    other = next(s for s in env.cfg.strategy.tradable if s not in env.sim.pos)
    env.sim.pos[other] = SimPos(other, 1, 1.0, env.sim.price(other), 5.0)
    assert not env.engine.reconcile()
    ok, why = env.engine.guard.can_open(env.clock.now_ms())
    assert not ok and "mismatch" in why
    assert env.db.events(kinds=("mismatch",))
    del env.sim.pos[other]
    assert env.engine.reconcile()
    assert env.engine.guard.can_open(env.clock.now_ms())[0]


def test_stop_file_closes_everything_and_blocks_restart(tmp_path, full_run):
    env, sym = _with_position(tmp_path, full_run[0])
    (tmp_path / "STOP").write_text("")
    r = env.engine.step()
    assert r.killed
    assert not env.sim.pos and not env.db.positions()
    assert env.db.get("halted")
    assert set(_trades(env)["exit_reason"]) >= {"kill"}
    (tmp_path / "STOP").unlink()
    assert env.new_engine().start() is False


def test_drawdown_floor_halts_and_flattens(tmp_path, full_run):
    env, sym = _with_position(tmp_path, full_run[0])
    env.engine.guard.state.peak_equity = env.engine.equity()[0] / 0.55     # просадка 45 % от пика
    r = env.engine.step()
    assert r.halted and env.db.get("halted")
    assert not env.sim.pos and not env.db.positions()
    assert env.new_engine().start() is False


def test_stale_websocket_blocks_entries_but_not_exits(tmp_path, full_run):
    class Stale:
        def start(self): ...
        def public_age_s(self, now): return 10_000.0
        def pop_dirty(self): return False
        def restart(self): self.restarted = True
        def stop(self): ...
    ref_env, _ = full_run
    cfg = ref_env.cfg.model_copy(update={"engine": ref_env.cfg.engine.model_copy(update={"ws_required": True})})
    env = make_env(tmp_path, start_day=START_DAY, cfg=cfg, stream=Stale())
    assert env.engine.start()
    env.run_until(T0 + (START_DAY + 10) * DAY_MS)
    assert not env.sim.orders and not env.db.positions()
    skips = env.db.events(kinds=("skip",))
    assert skips and all("ws" in e["message"] for e in skips)
    assert env.engine.stream.restarted


def test_demo_balance_is_capped_to_bot_capital(tmp_path):
    env = make_env(tmp_path, start_day=START_DAY, equity=25.0, wallet=50_000.0)
    assert env.engine.start()
    eq, wallet = env.engine.equity()
    assert eq == pytest.approx(25.0) and wallet == pytest.approx(50_000.0)


def test_live_mode_requires_explicit_flag(tmp_path):
    from bot.engine.engine import Engine
    env = make_env(tmp_path, start_day=START_DAY)
    with pytest.raises(PermissionError):
        Engine(env.cfg, env.client, env.db, env.clock, env.data, mode="live", base_dir=tmp_path)


def test_with_funding_engine_matches_backtest_within_one_qty_step(tmp_path):
    env = _make_env(tmp_path, start_day=START_DAY)          # с финансированием
    assert env.engine.start()
    first_bar = int(env.db.get("last_bar_close")) + 1
    env.run_until(T0 + END_DAY * DAY_MS)
    a, b = _compare(_trades(env), env.backtest(first_bar).trades)
    assert len(a) >= 8 and list(a.index) == list(b.index)
    assert list(a["exit_reason"]) == list(b["exit_reason"])
    assert (np.abs(a["qty"] - b["qty"]) <= 0.02 + 1e-9).all()
    np.testing.assert_allclose(a["net_pnl"], b["net_pnl"], rtol=0.02, atol=0.5)
