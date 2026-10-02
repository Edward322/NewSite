"""Walk-forward: подбор параметров на обучающем окне, проверка на следующем.

- Окна: 180 дней подбора → 60 дней проверки, сдвиг 60 дней, только период разработки.
- Подбор идёт на «справочном» капитале (REF_EQUITY), где минимальный ордер биржи
  не мешает: так меряется сам перевес стратегии в R. Проверка — и на справочном
  капитале, и на реальных 10 USDT.
- Критерий выбора — SQN = среднее R / ст. откл. R × √min(n, 100), медиана по
  соседям в сетке (плато, а не пик). Конфигурации с < MIN_TRADES сделками за
  окно подбора не допускаются (иначе не будет 100 сделок за полгода).
- Состояние ограничителей риска (пик, серия, пауза, остановка) переходит из
  окна в окно, как при непрерывной работе.
"""
from __future__ import annotations

import math
import multiprocessing as mp
from dataclasses import dataclass

import numpy as np
import pandas as pd

from bot.backtest.engine import Backtester, BacktestResult, EngineConfig
from bot.config import Config, CostsCfg
from bot.data.bars import TF_MINUTES, MinuteData
from bot.market import Instrument
from bot.risk.guards import RiskState
from bot.strategy.candidates import FAMILIES, grid_configs, make

DAY_MS = 86_400_000
TRAIN_DAYS, TEST_DAYS = 180, 60
REF_EQUITY = 1000.0
MIN_TRADES = 90
WARMUP_BARS = 400


@dataclass
class Ctx:
    md: MinuteData
    f_ts: np.ndarray
    f_r: np.ndarray
    inst: Instrument
    cfg: Config
    data_start_ms: int          # раньше этого момента данные не используются даже для прогрева


_CTX: Ctx | None = None


def set_ctx(ctx: Ctx) -> None:
    global _CTX
    _CTX = ctx


def folds(dev_start: int, dev_end: int) -> list[tuple[int, int, int]]:
    out, t = [], dev_start + TRAIN_DAYS * DAY_MS
    while t < dev_end:
        out.append((t - TRAIN_DAYS * DAY_MS, t, min(t + TEST_DAYS * DAY_MS, dev_end)))
        t += TEST_DAYS * DAY_MS
    return out


def run_window(family: str, tf: str, params: dict, start: int, end: int, equity: float,
               costs: CostsCfg | None = None, state: RiskState | None = None,
               ctx: Ctx | None = None) -> BacktestResult:
    c = ctx or _CTX
    w0 = max(c.data_start_ms, start - WARMUP_BARS * TF_MINUTES[tf] * 60_000)
    ec = EngineConfig(risk=c.cfg.risk, costs=costs or c.cfg.costs, initial_equity=equity,
                      trade_start_ms=start, trade_end_ms=end, initial_risk_state=state)
    return Backtester(c.md.slice(w0, end), c.f_ts, c.f_r, c.inst, make(family, tf, params), ec).run()


def trade_stats(trades: pd.DataFrame) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0, "mean_r": np.nan, "std_r": np.nan, "sqn": np.nan, "pf": np.nan, "total_r": 0.0,
                "win_rate": np.nan}
    r = trades["r_multiple"].to_numpy()
    pnl = trades["net_pnl"].to_numpy()
    sd = r.std(ddof=1) if n > 1 else np.nan
    loss = -pnl[pnl < 0].sum()
    return {"trades": n, "mean_r": float(r.mean()), "std_r": float(sd),
            "sqn": float(r.mean() / sd * math.sqrt(min(n, 100))) if sd and sd > 0 else np.nan,
            "pf": float(pnl[pnl > 0].sum() / loss) if loss > 0 else np.inf,
            "total_r": float(r.sum()), "win_rate": float((pnl > 0).mean())}


def _eval_one(args) -> dict:
    family, tf, params, start, end = args
    res = run_window(family, tf, params, start, end, REF_EQUITY)
    return {**params, **trade_stats(res.trades)}


def evaluate_grid(family: str, tf: str, start: int, end: int, pool=None) -> pd.DataFrame:
    jobs = [(family, tf, p, start, end) for p in grid_configs(family)]
    rows = pool.map(_eval_one, jobs) if pool else [_eval_one(j) for j in jobs]
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, family: str) -> tuple[dict | None, float]:
    """Лучшее плато по SQN среди допустимых конфигураций. None — торговать нечем."""
    grid = FAMILIES[family].grid
    keys = list(grid)
    idx = {k: {v: i for i, v in enumerate(grid[k])} for k in keys}
    elig = df["trades"] >= MIN_TRADES
    adj = np.where(elig, df["sqn"].fillna(0.0), np.minimum(df["sqn"].fillna(0.0), 0.0))
    pos = {tuple(idx[k][row[k]] for k in keys): j for j, row in df[keys].iterrows()}
    best, best_score, best_tie = None, 0.0, -np.inf
    for coord, j in pos.items():
        if not elig.iloc[j]:
            continue
        neigh = [j]
        for d in range(len(keys)):
            for step in (-1, 1):
                nc = list(coord)
                nc[d] += step
                if tuple(nc) in pos:
                    neigh.append(pos[tuple(nc)])
        # Медиана по окрестности: одиночный выброс не «поднимает» соседей.
        score = float(np.median(adj[neigh]))
        tie = float(np.mean(adj[neigh]))
        if score > best_score or (best is not None and score == best_score and tie > best_tie):
            best, best_score, best_tie = {k: df.iloc[j][k] for k in keys}, score, tie
    if best is not None:
        best = {k: (int(v) if float(v).is_integer() and isinstance(grid[k][0], int) else float(v))
                for k, v in best.items()}
    return best, best_score


@dataclass
class WFResult:
    family: str
    tf: str
    fold_rows: list[dict]
    trades_ref: pd.DataFrame
    trades_real: pd.DataFrame
    equity_ref: pd.DataFrame
    equity_real: pd.DataFrame
    grids: list[pd.DataFrame]


def walk_forward(family: str, tf: str, dev_start: int, dev_end: int, pool=None) -> WFResult:
    fold_rows, grids = [], []
    tr_ref, tr_real, eq_ref, eq_real = [], [], [], []
    e_ref, e_real = REF_EQUITY, _CTX.cfg.risk.starting_equity_usdt
    s_ref = s_real = None
    for k, (a, b, c) in enumerate(folds(dev_start, dev_end)):
        g = evaluate_grid(family, tf, a, b, pool)
        g["fold"] = k
        grids.append(g)
        params, score = select(g, family)
        row = {"fold": k, "train_start": a, "test_start": b, "test_end": c, "params": params,
               "train_score": score}
        if params is not None:
            r1 = run_window(family, tf, params, b, c, e_ref, state=s_ref)
            r2 = run_window(family, tf, params, b, c, e_real, state=s_real)
            e_ref, s_ref = r1.final_equity, r1.risk_state
            e_real, s_real = r2.final_equity, r2.risk_state
            for res, trs, eqs in ((r1, tr_ref, eq_ref), (r2, tr_real, eq_real)):
                if len(res.trades):
                    trs.append(res.trades.assign(fold=k))
                e = res.equity
                eqs.append(e[(e["ts"] > b) & (e["ts"] <= c)])
            row.update({f"oos_{key}": v for key, v in trade_stats(r1.trades).items()})
            row["oos_real_trades"] = len(r2.trades)
        else:
            # торговать нечем: капитал стоит на месте
            for eqs, e in ((eq_ref, e_ref), (eq_real, e_real)):
                eqs.append(pd.DataFrame({"ts": [c], "equity": [e], "cash": [e], "position": [0]}))
        row["equity_ref_end"], row["equity_real_end"] = e_ref, e_real
        fold_rows.append(row)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return WFResult(family, tf, fold_rows, cat(tr_ref), cat(tr_real), cat(eq_ref), cat(eq_real), grids)


def make_pool(ctx: Ctx, processes: int | None = None):
    set_ctx(ctx)
    return mp.get_context("fork").Pool(processes or mp.cpu_count(), initializer=set_ctx, initargs=(ctx,))
