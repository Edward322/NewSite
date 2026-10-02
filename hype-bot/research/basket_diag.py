"""Диагностика (не отбор): есть ли у семейства перевес хоть в какой-то части сетки.

Каждая конфигурация каждой комбинации прогоняется без подбора на всех 27 месяцах
проверочных кварталов (2024-01-01 — 2026-04-01), справочный капитал, ограничители
как в бою, но без остановки по просадке (max_drawdown 0,99), чтобы измерить перевес
по всем сделкам. Метрики — в R (не зависят от наращивания капитала):
среднее R, PF по R, число сделок. Медиана по сетке показывает, есть ли у идеи
перевес «в целом», а не у одной удачной конфигурации.

Результат: reports/basket_diag.txt и reports/basket/diag_grid.csv.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bot.config import PROJECT_ROOT, load_config
from bot.data.bars import to_ms
from bot.data.panel import load_basket_dev
from research import basket_wf as W
from bot.strategy.basket import COMBOS, grid_configs


def _one(args):
    fam, tf, prm, a, b, costs = args
    res = W.run_window(fam, tf, prm, a, b, W.REF_EQUITY, costs=costs)
    tr = res.trades
    if len(tr) == 0:
        return {"family": fam, "tf": tf, **{f"p_{k}": v for k, v in prm.items()}, "trades": 0}
    r = tr["r_multiple"].to_numpy()
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    long_r = tr.loc[tr["side"] == 1, "r_multiple"]
    short_r = tr.loc[tr["side"] == -1, "r_multiple"]
    return {"family": fam, "tf": tf, **{f"p_{k}": v for k, v in prm.items()}, "trades": len(tr),
            "mean_r": float(r.mean()), "pf_r": float(gain / loss) if loss > 0 else np.inf,
            "gross_mean_r": float(((tr["gross_pnl"]) / tr["planned_loss"]).mean()),
            "long_n": len(long_r), "long_mean_r": float(long_r.mean()) if len(long_r) else np.nan,
            "short_n": len(short_r), "short_mean_r": float(short_r.mean()) if len(short_r) else np.nan,
            "halted": res.halted}


def main() -> None:
    cfg = load_config()
    risk = cfg.risk.model_copy(update={"max_drawdown": 0.99, "daily_loss_limit": 0.99})
    cfg = cfg.model_copy(update={"risk": risk})
    panel = load_basket_dev(cfg)
    a, b = to_ms(W.QUARTERS[0]), to_ms(W.QUARTERS[-1])
    jobs = [(f, tf, p, a, b, None) for f, tf in COMBOS for p in grid_configs(f)]
    pool = W.make_pool(panel, cfg)
    t0 = time.time()
    try:
        rows = pool.map(_one, jobs, chunksize=2)
    finally:
        pool.close()
    df = pd.DataFrame(rows)
    df.to_csv(W.OUT_DIR / "diag_grid.csv", index=False)
    lines = ["Диагностика сетки без подбора: 2024-01-01 — 2026-04-01, справочный капитал, без остановки по просадке",
             f"({len(jobs)} прогонов, {time.time() - t0:.0f} с). R — результат сделки в единицах запланированного риска,",
             "после всех издержек; «до издержек» — без комиссий, проскальзывания и финансирования.", ""]
    for (f, tf), g in df.groupby(["family", "tf"], sort=False):
        g = g[g["trades"] > 0]
        if not len(g):
            lines.append(f"{f:12s} {tf:3s}: сделок нет")
            continue
        best = g.sort_values("mean_r").iloc[-1]
        lines.append(
            f"{f:12s} {tf:3s}: конфигураций {len(g):3d}; сделок медиана {g['trades'].median():5.0f}; "
            f"среднее R: медиана {g['mean_r'].median():+.3f}, доля > 0 {(g['mean_r'] > 0).mean() * 100:3.0f}%, "
            f"до издержек медиана {g['gross_mean_r'].median():+.3f}; PF по R медиана {g['pf_r'].median():.2f}; "
            f"лучшая {best['mean_r']:+.3f} ({best['trades']:.0f} сделок)")
        lines.append(f"{'':17s}лонги: ср. R медиана {g['long_mean_r'].median():+.3f}; "
                     f"шорты: ср. R медиана {g['short_mean_r'].median():+.3f}")
    # почему дневные стратегии почти не торгуют: типичный дневной ATR против предела стопа 9,5 %
    import warnings

    from bot.data.panel import aggregate_panel
    from bot.strategy import indicators as ind
    b = aggregate_panel(panel.slice(a, b), "1d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        atr = [np.nanmedian(ind.atr(b.high[r], b.low[r], b.close[r], 14) / b.close[r]) for r in range(b.n_trade)]
    lines += ["", f"Медианный дневной ATR монет корзины: {np.median(atr) * 100:.1f}% цены "
                  f"(по монетам {min(atr) * 100:.1f}–{max(atr) * 100:.1f}%); стоп 2–3 ATR = "
                  f"{2 * np.median(atr) * 100:.0f}–{3 * np.median(atr) * 100:.0f}% при пределе 9,5%."]
    out = PROJECT_ROOT / "reports" / "basket_diag.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
