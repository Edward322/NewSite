"""Walk-forward для корзины (протокол docs/BASKET_PROTOCOL.md, раздел 4).

Окна проверки — кварталы 2024Q1…2026Q1; подбор на 12 месяцах перед кварталом по
медиане SQN в окрестности конфигурации; ≥ MIN_TRADES сделок в окне подбора. Капитал
и состояние ограничителей переходят из квартала в квартал.

Запуск: python -m research.basket_wf [--min-trades N] [--prefix P] [семейство:тф ...]
(без комбинаций — все 11). Результаты: reports/basket/<P>_<семейство>_<тф>.pkl.
--min-trades/--prefix — только для справочных раундов (отклонение от протокола, помечается в отчёте).
"""
from __future__ import annotations

import math
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig, PortfolioResult
from bot.config import PROJECT_ROOT, Config, CostsCfg, load_config
from bot.data.bars import to_ms
from bot.data.panel import Panel, load_basket_dev
from bot.risk.guards import RiskState
from bot.strategy.basket import COMBOS, FAMILIES, grid_configs, make

DAY_MS = 86_400_000
REF_EQUITY = 1000.0
MIN_TRADES = 200
WARMUP_DAYS = 120
MAX_POSITIONS, MAX_OPEN_RISK = 4, 0.20
QUARTERS = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01",
            "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01"]
OUT_DIR = PROJECT_ROOT / "reports" / "basket"

_PANEL: Panel | None = None
_CFG: Config | None = None


def set_ctx(panel: Panel, cfg: Config) -> None:
    global _PANEL, _CFG
    _PANEL, _CFG = panel, cfg


def folds() -> list[tuple[int, int, int]]:
    out = []
    for a, b in zip(QUARTERS[:-1], QUARTERS[1:]):
        ts = pd.Timestamp(a, tz="UTC")
        out.append((to_ms(ts - pd.DateOffset(months=12)), to_ms(ts), to_ms(b)))
    return out


def run_window(family: str, tf: str, params: dict, start: int, end: int, equity: float,
               costs: CostsCfg | None = None, state: RiskState | None = None,
               panel: Panel | None = None, cfg: Config | None = None) -> PortfolioResult:
    P, C = panel or _PANEL, cfg or _CFG
    w0 = max(int(P.ts[0]), start - WARMUP_DAYS * DAY_MS)
    pc = PortfolioConfig(risk=C.risk, costs=costs or C.costs, initial_equity=equity,
                         max_positions=MAX_POSITIONS, max_open_risk=MAX_OPEN_RISK,
                         trade_start_ms=start, trade_end_ms=end, initial_risk_state=state)
    return PortfolioBacktester(P.slice(w0, end), make(family, tf, params), pc).run()


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
    return {**params, **trade_stats(res.trades), "halted": res.halted}


def evaluate_grid(family: str, tf: str, start: int, end: int, pool=None) -> pd.DataFrame:
    jobs = [(family, tf, p, start, end) for p in grid_configs(family)]
    rows = pool.map(_eval_one, jobs) if pool else [_eval_one(j) for j in jobs]
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, grid: dict, min_trades: int = MIN_TRADES) -> tuple[dict | None, float]:
    """Лучшее плато: медиана SQN по конфигурации и её соседям (±1 шаг по одному параметру)."""
    keys = list(grid)
    idx = {k: {v: i for i, v in enumerate(grid[k])} for k in keys}
    elig = (df["trades"] >= min_trades).to_numpy()
    sqn = df["sqn"].fillna(0.0).to_numpy()
    adj = np.where(elig, sqn, np.minimum(sqn, 0.0))
    pos = {tuple(idx[k][row[k]] for k in keys): j for j, row in enumerate(df[keys].to_dict("records"))}
    best, best_score, best_tie = None, 0.0, -np.inf
    for coord, j in pos.items():
        if not elig[j]:
            continue
        neigh = [j]
        for d in range(len(keys)):
            for step in (-1, 1):
                nc = list(coord)
                nc[d] += step
                if tuple(nc) in pos:
                    neigh.append(pos[tuple(nc)])
        score, tie = float(np.median(adj[neigh])), float(np.mean(adj[neigh]))
        if score > best_score or (best is not None and score == best_score and tie > best_tie):
            best, best_score, best_tie = j, score, tie
    if best is None:
        return None, best_score
    row = df.iloc[best]
    return {k: (type(grid[k][0])(row[k]) if not isinstance(grid[k][0], str) else str(row[k])) for k in keys}, best_score


@dataclass
class WF:
    family: str
    tf: str
    fold_rows: list[dict]
    trades_ref: pd.DataFrame
    trades_real: pd.DataFrame
    equity_ref: pd.DataFrame
    equity_real: pd.DataFrame
    grids: list[pd.DataFrame] = field(default_factory=list)
    halted_real: bool = False
    halted_ref: bool = False


def walk_forward(family: str, tf: str, pool=None, costs: CostsCfg | None = None,
                 min_trades: int = MIN_TRADES) -> WF:
    grid = FAMILIES[family].grid
    rows, grids, tr_ref, tr_real, eq_ref, eq_real = [], [], [], [], [], []
    e_ref, e_real = REF_EQUITY, _CFG.risk.starting_equity_usdt
    s_ref = s_real = None
    h_ref = h_real = False
    for k, (a, b, c) in enumerate(folds()):
        g = evaluate_grid(family, tf, a, b, pool)
        g["fold"] = k
        grids.append(g)
        params, score = select(g, grid, min_trades)
        row = {"fold": k, "train_start": a, "test_start": b, "test_end": c, "params": params, "train_score": score}
        if params is not None:
            r1 = run_window(family, tf, params, b, c, e_ref, costs, s_ref)
            r2 = run_window(family, tf, params, b, c, e_real, costs, s_real)
            e_ref, s_ref, h_ref = r1.final_equity, r1.risk_state, r1.halted
            e_real, s_real, h_real = r2.final_equity, r2.risk_state, r2.halted
            for res, trs, eqs in ((r1, tr_ref, eq_ref), (r2, tr_real, eq_real)):
                if len(res.trades):
                    trs.append(res.trades.assign(fold=k))
                e = res.equity
                eqs.append(e[(e["ts"] > b) & (e["ts"] <= c)])
            row.update({f"oos_{key}": v for key, v in trade_stats(r1.trades).items()})
            row["oos_real_trades"] = len(r2.trades)
        else:
            for eqs, e in ((eq_ref, e_ref), (eq_real, e_real)):
                eqs.append(pd.DataFrame({"ts": [c], "equity": [e], "cash": [e], "position": [0]}))
        row["equity_ref_end"], row["equity_real_end"] = e_ref, e_real
        rows.append(row)
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    return WF(family, tf, rows, cat(tr_ref), cat(tr_real), cat(eq_ref), cat(eq_real), grids, h_real, h_ref)


def load_pickle(path) -> WF:
    """Результаты, сохранённые при запуске `python -m research.basket_wf`, ссылаются на __main__.WF."""
    import __main__
    if not hasattr(__main__, "WF"):
        __main__.WF = WF
    with open(path, "rb") as fh:
        return pickle.load(fh)


def make_pool(panel: Panel, cfg: Config, processes: int | None = None):
    set_ctx(panel, cfg)
    return mp.get_context("fork").Pool(processes or mp.cpu_count(), initializer=set_ctx, initargs=(panel, cfg))


def fmt_params(p: dict | None) -> str:
    return "—" if p is None else ", ".join(f"{k}={v}" for k, v in p.items())


def main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-trades", type=int, default=MIN_TRADES)
    ap.add_argument("--prefix", default="wf")
    ap.add_argument("combos", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config()
    panel = load_basket_dev(cfg)
    combos = [tuple(a.split(":")) for a in args.combos] or COMBOS
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pool = make_pool(panel, cfg)
    try:
        for family, tf in combos:
            t0 = time.time()
            wf = walk_forward(family, tf, pool, min_trades=args.min_trades)
            with open(OUT_DIR / f"{args.prefix}_{family}_{tf}.pkl", "wb") as fh:
                pickle.dump(wf, fh)
            st = trade_stats(wf.trades_ref)
            print(f"{family:12s} {tf:3s}: OOS сделок {st['trades']:4d}, PF {st['pf']:.2f}, ср. R {st['mean_r']:+.3f}, "
                  f"справочный {wf.fold_rows[-1]['equity_ref_end']:.0f}, 10 USDT → {wf.fold_rows[-1]['equity_real_end']:.2f}"
                  f"{' (остановлен)' if wf.halted_real else ''}  [{time.time() - t0:.0f} с]", flush=True)
            for r in wf.fold_rows:
                print(f"    квартал {r['fold']}: {fmt_params(r['params'])} | подбор {r['train_score']:.2f} | "
                      f"OOS сделок {r.get('oos_trades', 0)}, PF {r.get('oos_pf', float('nan')):.2f}, "
                      f"ср. R {r.get('oos_mean_r', float('nan')):+.3f}", flush=True)
    finally:
        pool.close()


if __name__ == "__main__":
    main(sys.argv[1:])
