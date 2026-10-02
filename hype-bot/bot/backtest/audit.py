"""Независимый пересчёт сделок бэктеста из сырых минутных данных.

Не использует код движка: цены входа/выхода, комиссии и финансирование
пересчитываются заново по правилам исполнения, затем сверяются с движком.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestResult
from bot.config import CostsCfg
from bot.data.bars import MinuteData


def audit_trades(res: BacktestResult, md: MinuteData, f_ts: np.ndarray, f_rate: np.ndarray,
                 costs: CostsCfg, initial_equity: float) -> pd.DataFrame:
    rows = []
    for t in res.trades.itertuples():
        side, qty = int(t.side), float(t.qty)
        i_in = int(np.searchsorted(md.ts, t.entry_ts))
        entry = md.open[i_in] * (1 + side * costs.slippage)
        exit_fee_rate = {"take_profit": costs.maker_fee, "liquidation": 0.0}.get(t.exit_reason, costs.taker_fee)
        # выход по рынку по открытию минуты (сигнал/остановка) или по закрытию последней минуты (конец данных)
        i_out = int(np.searchsorted(md.ts, t.exit_ts))
        if t.exit_reason in ("signal", "cross", "max_drawdown") and md.ts[i_out] == t.exit_ts and \
                np.isclose(t.exit_price, md.open[i_out] * (1 - side * costs.slippage)):
            exit_px = md.open[i_out] * (1 - side * costs.slippage)
        elif t.exit_reason == "end":
            exit_px = md.close[i_out] * (1 - side * costs.slippage)
        else:
            exit_px = float(t.exit_price)       # стоп/тейк/ликвидация — цена из правил, проверяется отдельно
        sel = (f_ts > t.entry_ts) & (f_ts <= t.exit_ts)
        fidx = np.searchsorted(md.ts, f_ts[sel])
        funding = float(np.sum(side * qty * md.mark_open[fidx] * f_rate[sel]))
        fees = qty * entry * costs.taker_fee + qty * exit_px * exit_fee_rate
        net = side * qty * (exit_px - entry) - fees - funding
        rows.append({"entry_ts": t.entry_ts, "d_entry": entry - t.entry_price, "d_funding": funding - t.funding,
                     "d_net": net - t.net_pnl, "net": net})
    df = pd.DataFrame(rows)
    df.attrs["final_equity_recomputed"] = initial_equity + (df["net"].sum() if len(df) else 0.0)
    df.attrs["final_equity_engine"] = res.final_equity
    return df
