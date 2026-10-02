from __future__ import annotations

import numpy as np

from bot.backtest.engine import Backtester, EngineConfig
from bot.config import CostsCfg, RiskCfg
from bot.data.bars import MinuteData
from tests.instruments import hype

MIN = 60_000
T0 = 1_767_225_600_000  # 2026-01-01 00:00 UTC

COSTS = CostsCfg(maker_fee=0.0002, taker_fee=0.00055, slippage=0.0002, stop_penetration=0.25)
RISK = RiskCfg(starting_equity_usdt=10, risk_per_trade=0.05, daily_loss_limit=0.25, max_drawdown=0.8,
               loss_streak_pause_trades=4, loss_streak_pause_hours=24)


def md_from_rows(rows, t0=T0, mark_rows=None) -> MinuteData:
    """rows: список (o, h, l, c) по минутам."""
    a = np.asarray(rows, dtype=float)
    m = np.asarray(mark_rows, dtype=float) if mark_rows is not None else a
    n = len(a)
    return MinuteData(ts=t0 + np.arange(n, dtype="int64") * MIN, open=a[:, 0], high=a[:, 1], low=a[:, 2],
                      close=a[:, 3], volume=np.full(n, 100.0), mark_open=m[:, 0], mark_high=m[:, 1],
                      mark_low=m[:, 2], mark_close=m[:, 3])


def flat(n, price=40.0):
    return [(price, price, price, price)] * n


def random_walk(n, seed=0, p0=40.0, t0=T0) -> MinuteData:
    rng = np.random.default_rng(seed)
    c = p0 * np.exp(np.cumsum(rng.normal(0, 0.0015, n)))
    o = np.r_[p0, c[:-1]]
    sp = np.abs(rng.normal(0, 0.0008, n))
    h, l = np.maximum(o, c) * (1 + sp), np.minimum(o, c) * (1 - sp)
    mo, mc = o * (1 + rng.normal(0, 0.0002, n)), c * (1 + rng.normal(0, 0.0002, n))
    return MinuteData(ts=t0 + np.arange(n, dtype="int64") * MIN, open=o, high=h, low=l, close=c,
                      volume=rng.uniform(10, 1000, n), mark_open=mo, mark_high=np.maximum(h, np.maximum(mo, mc)),
                      mark_low=np.minimum(l, np.minimum(mo, mc)), mark_close=mc)


def run(md, strategy, funding=None, risk=RISK, costs=COSTS, equity=10.0, **kw):
    f_ts, f_r = funding if funding is not None else (np.array([], dtype="int64"), np.array([]))
    cfg = EngineConfig(risk=risk, costs=costs, initial_equity=equity, **kw)
    return Backtester(md, f_ts, f_r, hype(), strategy, cfg).run()
