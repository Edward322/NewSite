"""Проверки отсутствия заглядывания в будущее.

1. Обрезка: бэктест на данных до момента C обязан дать те же решения и те же
   завершённые сделки до C, что и бэктест на полных данных.
2. Подмена будущего: если все минуты начиная с C заменить случайным
   блужданием, решения и сделки до C не должны измениться.

Любая стратегия, использующая будущие свечи (сдвиг назад, нормировку по всей
выборке, центрированные окна и т. п.), не проходит хотя бы одну из проверок.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from bot.backtest.engine import BacktestResult
from bot.data.bars import MinuteData

Runner = Callable[[MinuteData], BacktestResult]


def _closed_trades(res: BacktestResult, before_ms: int) -> pd.DataFrame:
    t = res.trades
    if len(t) == 0:
        return t
    t = t[(t["exit_ts"] < before_ms) & (t["exit_reason"] != "end")]
    return t.drop(columns=["equity_after"]).reset_index(drop=True)


def _compare(full: BacktestResult, part: BacktestResult, horizon_ms: int, label: str) -> list[str]:
    problems = []
    fd = [d for d in full.decisions if d[0] <= horizon_ms]
    pdx = [d for d in part.decisions if d[0] <= horizon_ms]
    if fd != pdx:
        first = next((i for i, (a, b) in enumerate(zip(fd, pdx)) if a != b), min(len(fd), len(pdx)))
        problems.append(f"{label}: решения расходятся, первое отличие #{first}: "
                        f"{fd[first] if first < len(fd) else None} vs {pdx[first] if first < len(pdx) else None}")
    a, b = _closed_trades(full, horizon_ms), _closed_trades(part, horizon_ms)
    if len(a) != len(b) or (len(a) and not a.equals(b)):
        problems.append(f"{label}: сделки до горизонта расходятся ({len(a)} vs {len(b)})")
    return problems


def truncation_check(run: Runner, md: MinuteData, cuts_ms: list[int]) -> list[str]:
    full = run(md)
    problems: list[str] = []
    for cut in cuts_ms:
        part = run(md.slice(None, cut))
        if len(part.bars) == 0:
            continue
        horizon = int(part.bars.close_ts[-1])
        problems += _compare(full, part, horizon, f"обрезка {pd.Timestamp(cut, unit='ms', tz='UTC')}")
    return problems


def perturb_after(md: MinuteData, cut_ms: int, seed: int = 0) -> MinuteData:
    """Минуты с ts ≥ cut заменяются случайным блужданием от последней цены до cut."""
    i = int(np.searchsorted(md.ts, cut_ms, "left"))
    if i == 0 or i >= len(md):
        return md
    rng = np.random.default_rng(seed)
    n = len(md) - i
    close = md.close[i - 1] * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    open_ = np.r_[md.close[i - 1], close[:-1]]
    spread = np.abs(rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = rng.uniform(10, 1000, n)
    cat = lambda a, b: np.r_[a[:i], b]  # noqa: E731
    return MinuteData(ts=md.ts.copy(), open=cat(md.open, open_), high=cat(md.high, high),
                      low=cat(md.low, low), close=cat(md.close, close), volume=cat(md.volume, vol),
                      mark_open=cat(md.mark_open, open_), mark_high=cat(md.mark_high, high),
                      mark_low=cat(md.mark_low, low), mark_close=cat(md.mark_close, close))


def perturbation_check(run: Runner, md: MinuteData, cuts_ms: list[int], seed: int = 0) -> list[str]:
    full = run(md)
    problems: list[str] = []
    for j, cut in enumerate(cuts_ms):
        alt = run(perturb_after(md, cut, seed + j))
        problems += _compare(full, alt, cut, f"подмена после {pd.Timestamp(cut, unit='ms', tz='UTC')}")
    return problems
