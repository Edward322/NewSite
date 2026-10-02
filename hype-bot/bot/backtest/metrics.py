"""Метрики результата бэктеста и сравнение с buy & hold.

Доходности дневные (по закрытию дня UTC), годовой множитель √365 —
криптовалюты торгуются без выходных.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestResult

DAY_MS = 86_400_000


def _daily(ts_ms: np.ndarray, values: np.ndarray, start_value: float, start_ms: int) -> pd.Series:
    s = pd.Series(values, index=pd.to_datetime(ts_ms, unit="ms", utc=True))
    daily = s.resample("1D").last().ffill()
    first_day = pd.Timestamp(start_ms, unit="ms", tz="UTC").floor("D") - pd.Timedelta(days=1)
    return pd.concat([pd.Series([start_value], index=[first_day]), daily])


def path_metrics(daily_equity: pd.Series) -> dict:
    eq = daily_equity.astype(float)
    r = eq.pct_change().dropna()
    days = max(len(eq) - 1, 1)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (365 / days) - 1 if eq.iloc[-1] > 0 else -1.0
    dd = (eq / eq.cummax() - 1).min()
    sd = r.std(ddof=1)
    downside = math.sqrt((np.minimum(r, 0) ** 2).mean()) if len(r) else float("nan")
    return {
        "total_return": total,
        "cagr": cagr,
        "max_drawdown": -dd,
        "sharpe": r.mean() / sd * math.sqrt(365) if sd > 0 else float("nan"),
        "sortino": r.mean() / downside * math.sqrt(365) if downside > 0 else float("nan"),
        "calmar": cagr / -dd if dd < 0 else float("nan"),
        "days": days,
    }


def compute_metrics(res: BacktestResult, initial_equity: float,
                    start_ms: int | None = None, end_ms: int | None = None) -> dict:
    eq = res.equity
    tr = res.trades
    if start_ms is not None:
        prior = eq[eq["ts"] < start_ms]
        start_value = float(prior["equity"].iloc[-1]) if len(prior) else initial_equity
        eq = eq[eq["ts"] >= start_ms]
        if len(tr):
            tr = tr[tr["entry_ts"] >= start_ms]
    else:
        start_value, start_ms = initial_equity, int(eq["ts"].iloc[0]) if len(eq) else 0
    if end_ms is not None:
        eq = eq[eq["ts"] <= end_ms]
        if len(tr):
            tr = tr[tr["entry_ts"] < end_ms]
    if len(eq) == 0:
        return {"trades": 0}
    m = path_metrics(_daily(eq["ts"].to_numpy(), eq["equity"].to_numpy(), start_value, start_ms))
    m["exposure"] = float((eq["position"] != 0).mean())
    n = len(tr)
    m["trades"] = n
    if n:
        pnl = tr["net_pnl"]
        wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        m.update(
            win_rate=float((pnl > 0).mean()),
            profit_factor=float(wins / losses) if losses > 0 else float("inf"),
            avg_trade_usdt=float(pnl.mean()),
            avg_trade_r=float(tr["r_multiple"].mean()),
            # средняя сделка в % от капитала перед сделкой
            avg_trade_pct=float((pnl / (tr["equity_after"] - pnl)).mean()),
            fees_usdt=float((tr["entry_fee"] + tr["exit_fee"]).sum()),
            funding_usdt=float(tr["funding"].sum()),
            avg_bars_held=float(tr["bars_held"].mean()),
            exit_reasons=tr["exit_reason"].value_counts().to_dict(),
        )
    m["final_equity"] = float(eq["equity"].iloc[-1])
    m["halted"] = res.halted
    return m


def buy_and_hold(ts_ms: np.ndarray, close: np.ndarray, funding_ts: np.ndarray, funding_rate: np.ndarray,
                 start_ms: int, end_ms: int, taker_fee: float = 0.0) -> dict:
    """Лонг 1x бессрочного контракта на весь период: цена + уплаченное финансирование."""
    sel = (ts_ms >= start_ms) & (ts_ms < end_ms)
    t, p = ts_ms[sel], close[sel]
    if len(p) == 0:
        return {}
    rel = p / p[0]
    fsel = (funding_ts >= start_ms) & (funding_ts < end_ms)
    f_t, f_r = funding_ts[fsel], funding_rate[fsel]
    idx = np.clip(np.searchsorted(t, f_t, "left"), 0, len(p) - 1)
    pay = np.zeros(len(p))
    np.add.at(pay, idx, f_r * rel[idx])
    equity = (1 - taker_fee) * rel - np.cumsum(pay)
    equity[-1] -= taker_fee * rel[-1]
    return path_metrics(_daily(t, equity, 1.0, start_ms))
