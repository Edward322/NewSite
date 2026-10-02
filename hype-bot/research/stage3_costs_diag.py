"""Диагностика: есть ли у кандидатов перевес ДО издержек (только период разработки).

Запуск: python -m research.stage3_costs_diag
Все конфигурации на всём периоде разработки (справочный капитал) — с обычными и
с нулевыми издержками. Это не подбор параметров, а ответ на вопрос «что съедает
результат».
"""
from __future__ import annotations

import pandas as pd

from bot.config import CostsCfg, PROJECT_ROOT, load_config
from bot.data.bars import load_funding
from bot.data.split import dev_range, load_dev
from bot.market import Instrument
from bot.strategy.candidates import FAMILIES, TIMEFRAMES, grid_configs
from research import wf

ZERO = CostsCfg(maker_fee=0, taker_fee=0, slippage=0, stop_penetration=0)


def _job(args):
    family, tf, p, zero = args
    c = wf._CTX
    d0, d1 = dev_range(c.cfg)
    res = wf.run_window(family, tf, p, d0, d1, wf.REF_EQUITY, costs=ZERO if zero else None)
    st = wf.trade_stats(res.trades)
    return {"family": family, "tf": tf, **{f"p_{k}": v for k, v in p.items()}, "zero_costs": zero,
            "trades": st["trades"], "mean_r": st["mean_r"], "pf": st["pf"]}


def main():
    cfg = load_config()
    d0, d1 = dev_range(cfg)
    ctx = wf.Ctx(load_dev(cfg), *load_funding(cfg.data_dir(), cfg.symbol),
                 Instrument.from_files(cfg.data_dir(), cfg.symbol), cfg, d0)
    pool = wf.make_pool(ctx)
    jobs = [(f, tf, p, z) for f in FAMILIES for tf in TIMEFRAMES for p in grid_configs(f) for z in (False, True)]
    df = pd.DataFrame(pool.map(_job, jobs, chunksize=8))
    pool.close()
    out = PROJECT_ROOT / "reports" / "stage3"
    df.to_csv(out / "costs_diag.csv.gz", index=False)
    days = (d1 - d0) / wf.DAY_MS
    df["trades_per_6m"] = df["trades"] / days * 182.5
    piv = df.groupby(["family", "tf", "zero_costs"]).agg(
        med_trades_6m=("trades_per_6m", "median"), med_mean_r=("mean_r", "median"),
        share_pos=("mean_r", lambda s: float((s > 0).mean()))).unstack("zero_costs")
    pd.set_option("display.width", 200)
    print(f"Период разработки целиком ({days:.0f} дн.), все {len(df) // 2} конфигураций; медианы по сетке")
    print(piv.round(3).to_string())
    freq = df[df["trades_per_6m"] >= 100]
    print("\nТолько частые конфигурации (≥100 сделок за 6 мес):")
    print(freq.groupby("zero_costs").agg(configs=("mean_r", "size"), med_mean_r=("mean_r", "median"),
                                         share_pos=("mean_r", lambda s: float((s > 0).mean())),
                                         best_mean_r=("mean_r", "max")).round(3).to_string())


if __name__ == "__main__":
    main()
