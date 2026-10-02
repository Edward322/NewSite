import numpy as np
import pytest

from bot.backtest.audit import audit_trades
from bot.backtest.metrics import buy_and_hold, path_metrics
from tests.helpers import COSTS, MIN, T0, random_walk, run
from tests.strategies import SmaCross


def test_engine_accounting_matches_independent_recalculation():
    md = random_walk(30 * 1440, seed=5)
    f_ts = np.arange(T0, T0 + 30 * 86_400_000, 8 * 3_600_000, dtype="int64")
    f_r = np.random.default_rng(1).normal(0.0001, 0.0003, len(f_ts))
    res = run(md, SmaCross(), funding=(f_ts, f_r))
    a = audit_trades(res, md, f_ts, f_r, COSTS, 10.0)
    assert len(a) > 30
    assert np.abs(a["d_entry"]).max() < 1e-9
    assert np.abs(a["d_funding"]).max() < 1e-9
    assert np.abs(a["d_net"]).max() < 1e-9
    assert a.attrs["final_equity_recomputed"] == pytest.approx(a.attrs["final_equity_engine"], abs=1e-9)


def test_path_metrics_on_known_series():
    import pandas as pd
    eq = pd.Series([100, 110, 99, 120], index=pd.date_range("2026-01-01", periods=4, tz="UTC"))
    m = path_metrics(eq)
    assert m["total_return"] == pytest.approx(0.2)
    assert m["max_drawdown"] == pytest.approx(0.1)
    r = eq.pct_change().dropna()
    assert m["sharpe"] == pytest.approx(r.mean() / r.std() * np.sqrt(365))


def test_buy_and_hold_pays_funding():
    ts = T0 + np.arange(3 * 1440, dtype="int64") * MIN
    close = np.full(len(ts), 50.0)
    f_ts = np.array([T0 + 8 * 3_600_000, T0 + 16 * 3_600_000], dtype="int64")
    m = buy_and_hold(ts, close, f_ts, np.array([0.001, 0.001]), int(ts[0]), int(ts[-1]) + MIN)
    assert m["total_return"] == pytest.approx(-0.002)
