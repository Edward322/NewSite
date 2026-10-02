"""Доказательства отсутствия заглядывания в будущее.

1. Движок: решение принимается на закрытии свечи, исполнение — не раньше
   открытия следующей минуты, по цене этой минуты.
2. Обрезка данных и подмена будущего не меняют прошлые решения и сделки.
3. Проверки действительно ловят утечку: стратегии с явным и скрытым
   заглядыванием в будущее их не проходят.
"""
import numpy as np
import pytest

from bot.backtest.causality import perturbation_check, truncation_check
from bot.config import load_config
from bot.data.bars import load_minutes, load_funding
from tests.helpers import MIN, T0, random_walk, run
from tests.strategies import Cheater, NormalizedLeak, SmaCross

MD = random_walk(20 * 1440, seed=11)
CUTS = [T0 + d * 1440 * MIN + o for d, o in ((5, 0), (9, 7 * MIN), (13, 33 * MIN), (17, 0))]


def runner(factory):
    return lambda md: run(md, factory())


def test_fills_never_precede_decisions_and_use_next_minute_open():
    res = run(MD, SmaCross())
    t = res.trades
    assert len(t) > 20
    assert (t["entry_ts"] >= t["decision_ts"]).all()
    # вход исполняется ровно по открытию минуты, следующей за моментом решения
    idx = np.searchsorted(MD.ts, t["entry_ts"].to_numpy())
    expected = MD.open[idx] * (1 + 0.0002 * t["side"].to_numpy())
    np.testing.assert_allclose(t["entry_price"].to_numpy(), expected)
    assert (t["entry_ts"] == t["decision_ts"]).all()   # решение в момент закрытия = открытие следующей минуты
    assert (t["exit_ts"] >= t["entry_ts"]).all()


def test_honest_strategy_passes_truncation_check():
    assert truncation_check(runner(SmaCross), MD, CUTS) == []


def test_honest_strategy_passes_future_perturbation_check():
    assert perturbation_check(runner(SmaCross), MD, CUTS) == []


@pytest.mark.parametrize("leaky", [Cheater, NormalizedLeak])
def test_checks_catch_lookahead(leaky):
    problems = truncation_check(runner(leaky), MD, CUTS) + perturbation_check(runner(leaky), MD, CUTS)
    assert problems, f"{leaky.__name__} должна была провалить проверку"


def test_cheater_would_look_great_which_is_why_checks_matter():
    honest = run(MD, SmaCross()).final_equity
    cheat = run(MD, Cheater()).final_equity
    assert cheat > honest


def _real_data_available():
    cfg = load_config()
    return (cfg.data_dir() / cfg.symbol / "kline_last_1m.parquet").exists()


@pytest.mark.skipif(not _real_data_available(), reason="нет локальных данных Bybit")
def test_truncation_on_real_hype_data():
    from bot.backtest.engine import Backtester, EngineConfig
    from bot.market import Instrument
    cfg = load_config()
    md = load_minutes(cfg.data_dir(), cfg.symbol, "2025-01-01", "2025-07-01")
    f_ts, f_r = load_funding(cfg.data_dir(), cfg.symbol)
    inst = Instrument.from_files(cfg.data_dir(), cfg.symbol)

    def run_real(m):
        ec = EngineConfig(risk=cfg.risk, costs=cfg.costs, initial_equity=10.0)
        return Backtester(m, f_ts, f_r, inst, SmaCross(timeframe="1h", fast=12, slow=48), ec).run()

    cuts = [int(md.ts[0] + f * (md.ts[-1] - md.ts[0])) for f in (0.3, 0.55, 0.8)]
    assert truncation_check(run_real, md, cuts) == []
    assert perturbation_check(run_real, md, cuts) == []
