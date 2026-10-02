"""Этап 3: walk-forward кандидатов на периоде разработки.

Запуск:
    python -m research.stage3_walkforward r1        # раунд 1: 4 типа × 15m/30m/1h/4h
    python -m research.stage3_walkforward r2        # раунд 2: fade/pullback/shock × 15m/30m/1h
    python -m research.stage3_walkforward slow      # справочно: медленные без требования частоты
Результаты: reports/stage3/<раунд>_summary.csv, _folds.csv, _grids.csv.gz, сделки и капитал.
Отложенные данные НЕ загружаются.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from bot.backtest.metrics import _daily, buy_and_hold, path_metrics
from bot.config import PROJECT_ROOT, load_config
from bot.data.bars import load_funding
from bot.data.split import dev_range, load_dev
from bot.market import Instrument
from bot.strategy.candidates import FAMILIES, ROUND1, ROUND2, TIMEFRAMES, TIMEFRAMES_R2, grid_configs
from research import wf

OUT = PROJECT_ROOT / "reports" / "stage3"


def oos_metrics(res: wf.WFResult, which: str, start_equity: float, start_ms: int) -> dict:
    eq = res.equity_ref if which == "ref" else res.equity_real
    tr = res.trades_ref if which == "ref" else res.trades_real
    m = path_metrics(_daily(eq["ts"].to_numpy(), eq["equity"].to_numpy(), start_equity, start_ms))
    st = wf.trade_stats(tr)
    return {f"{which}_{k}": v for k, v in {**m, **st}.items()}


ROUNDS = {
    "r1": ([c.family for c in ROUND1], TIMEFRAMES, wf.MIN_TRADES),
    "r2": ([c.family for c in ROUND2], TIMEFRAMES_R2, wf.MIN_TRADES),
    # Справочно: медленные варианты без требования 100 сделок за полгода (15 сделок за окно подбора).
    "slow": ([c.family for c in ROUND1], ["1h", "4h"], 15),
    # Проверка решения исключить декабрь 2024: те же раунды, история с первой свечи.
    "r1dec": ([c.family for c in ROUND1], TIMEFRAMES, wf.MIN_TRADES),
    "r2dec": ([c.family for c in ROUND2], TIMEFRAMES_R2, wf.MIN_TRADES),
}
FIRST_CANDLE = "2024-12-05T12:55:00Z"


def main(tag: str) -> None:
    families, timeframes, min_trades = ROUNDS[tag]
    wf.MIN_TRADES = min_trades
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    d0, d1 = dev_range(cfg)
    md = load_dev(cfg)
    if tag.endswith("dec"):
        from bot.data.bars import load_minutes, to_ms
        d0 = to_ms(FIRST_CANDLE)
        md = load_minutes(cfg.data_dir(), cfg.symbol, d0, d1)
    f_ts, f_r = load_funding(cfg.data_dir(), cfg.symbol)
    inst = Instrument.from_files(cfg.data_dir(), cfg.symbol)
    ctx = wf.Ctx(md=md, f_ts=f_ts, f_r=f_r, inst=inst, cfg=cfg, data_start_ms=d0)
    pool = wf.make_pool(ctx)
    fl = wf.folds(d0, d1)
    oos_start, oos_end = fl[0][1], fl[-1][2]
    oos_days = (oos_end - oos_start) / wf.DAY_MS
    print(f"Окна: {[(pd.Timestamp(a, unit='ms').date(), pd.Timestamp(b, unit='ms').date(), pd.Timestamp(c, unit='ms').date()) for a, b, c in fl]}")
    print(f"Вне выборки (OOS) всего: {oos_days:.0f} дн.")
    n_configs = sum(len(grid_configs(f)) for f in families) * len(timeframes)
    print(f"Раунд {tag}: {families} × {timeframes}; минимум сделок в окне подбора {min_trades}")
    print(f"Конфигураций в переборе: {n_configs} × {len(fl)} окон = {n_configs * len(fl)} прогонов подбора")

    summary, fold_rows, grids = [], [], []
    for family in families:
        for tf in timeframes:
            t0 = time.perf_counter()
            res = wf.walk_forward(family, tf, d0, d1, pool)
            row = {"family": family, "tf": tf,
                   "folds_traded": sum(r["params"] is not None for r in res.fold_rows),
                   **oos_metrics(res, "ref", wf.REF_EQUITY, oos_start),
                   **oos_metrics(res, "real", cfg.risk.starting_equity_usdt, oos_start)}
            row["ref_trades_per_6m"] = row["ref_trades"] / oos_days * 182.5
            row["real_halted"] = bool(res.equity_real["equity"].iloc[-1] <= 0.2 * res.equity_real["equity"].max()) \
                if len(res.equity_real) else False
            summary.append(row)
            for r in res.fold_rows:
                fold_rows.append({"family": family, "tf": tf, **{k: (json.dumps(v) if k == "params" else v)
                                                                  for k, v in r.items()}})
            for g in res.grids:
                grids.append(g.assign(family=family, tf=tf))
            for which, tr in (("ref", res.trades_ref), ("real", res.trades_real)):
                if len(tr):
                    tr.to_parquet(OUT / f"{tag}_trades_{which}_{family}_{tf}.parquet", index=False)
            for which, eq in (("ref", res.equity_ref), ("real", res.equity_real)):
                eq.to_parquet(OUT / f"{tag}_equity_{which}_{family}_{tf}.parquet", index=False)
            print(f"{family:9s} {tf:4s} | окон с торговлей {row['folds_traded']}/{len(fl)} | "
                  f"OOS сделок {row['ref_trades']:4d} ({row['ref_trades_per_6m']:.0f}/6мес) | PF {row['ref_pf']:.2f} | "
                  f"ср.R {row['ref_mean_r']:+.3f} | SQN {row['ref_sqn']:+.2f} | "
                  f"10 USDT → {row['real_total_return']:+.1%} (DD {row['real_max_drawdown']:.0%}, "
                  f"сделок {row['real_trades']}) | {time.perf_counter() - t0:.0f} с", flush=True)
    pool.close()

    bh = buy_and_hold(md.ts, md.close, f_ts, f_r, oos_start, oos_end, cfg.costs.taker_fee)
    print(f"\nBuy & hold за тот же OOS-период: доходность {bh['total_return']:+.1%}, DD {bh['max_drawdown']:.0%}, "
          f"Sharpe {bh['sharpe']:.2f}, Calmar {bh['calmar']:.2f}")
    pd.DataFrame(summary).to_csv(OUT / f"{tag}_summary.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / f"{tag}_folds.csv", index=False)
    pd.concat(grids).to_csv(OUT / f"{tag}_grids.csv.gz", index=False)
    (OUT / f"{tag}_meta.json").write_text(json.dumps({
        "folds": fl, "oos_start": oos_start, "oos_end": oos_end, "n_configs": n_configs,
        "train_days": wf.TRAIN_DAYS, "test_days": wf.TEST_DAYS, "min_trades": min_trades,
        "ref_equity": wf.REF_EQUITY, "buy_hold": bh}, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "r1")
