"""Реальный счёт: включается только флагом; автоостановка новых входов при исполнении хуже демо."""
from bot.data.panel import DAY_MS
from bot.engine.engine import Engine
from tests.engine_env import make_env, sim_config
from tests.helpers import T0


def test_live_engine_needs_flag_and_takes_backtest_baseline_without_demo(tmp_path):
    cfg = sim_config(25.0)
    cfg = cfg.model_copy(update={"live": cfg.live.model_copy(update={"enabled": True})})
    env = make_env(tmp_path, start_day=110, equity=25.0, cfg=cfg)
    eng = Engine(cfg, env.client, env.db, env.clock, env.data, mode="live", base_dir=tmp_path)
    assert eng.prefix == "hbl"
    assert eng.start()
    b = env.db.get("exec_baseline")
    assert b["source"] in ("допущения бэктеста", "демо")


def test_bad_execution_blocks_new_entries(tmp_path):
    env = make_env(tmp_path, start_day=110, equity=25.0)
    assert env.engine.start()
    env.run_until(T0 + 128 * DAY_MS)
    assert len(env.db.trades()) >= 3
    # база «как на демо» = фактическое исполнение имитатора → блокировки нет
    env.db.set("exec_baseline", {"slip": 0.0002, "stop": 0.0, "n": 10, "source": "демо"})
    env.engine.check_execution()
    assert "execution" not in env.engine.guard.state.blockers
    # демо исполнялось намного лучше → реальное исполнение «заметно хуже» → новые входы запрещены
    env.db.set("exec_baseline", {"slip": -0.002, "stop": -0.01, "n": 10, "source": "демо"})
    env.engine.check_execution()
    ok, why = env.engine.guard.can_open(env.clock.now_ms())
    assert not ok and "execution" in why and "хуже демо" in why
    # блокировка переживает перезапуск
    env.engine._save_guard()
    env.new_engine()
    assert env.engine.start()
    assert "execution" in env.engine.guard.state.blockers
