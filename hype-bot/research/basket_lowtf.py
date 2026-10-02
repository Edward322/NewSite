"""Разведка по просьбе пользователя: «агрессивнее и на более низком таймфрейме».

Трендовые и межмонетные семейства корзины на 1h и 15m (для сравнения — пробой 4h).
Процедура walk-forward та же (подбор раз в квартал на 12 месяцах, SQN, плато, ≥ 200 сделок),
кварталы 2024Q1 — 2026Q3 (11 кварталов), подбор на риске 2 %. Затем по тем же параметрам
по кварталам — счёт 25 USDT при риске 2 % и 5 %.

Отложенные данные уже израсходованы финальной проверкой; здесь они используются как
обычная история (обращение пишется в журнал). Это разведка, а не чистая проверка.

Запуск: python -m research.basket_lowtf → reports/basket_lowtf.txt
"""
from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd

from bot.backtest.metrics import path_metrics
from bot.config import PROJECT_ROOT, load_config
from bot.data.bars import to_ms
from bot.data.panel import load_basket_holdout
from bot.data.split import HOLDOUT_CONFIRM
from research import basket_wf as W
from research.basket_gates import basket_bh, daily_returns

QUARTERS = ["2024-01-01", "2024-04-01", "2024-07-01", "2024-10-01", "2025-01-01", "2025-04-01",
            "2025-07-01", "2025-10-01", "2026-01-01", "2026-04-01", "2026-07-01", "2026-10-01"]
COMBOS = [("breakout", "4h"), ("breakout", "1h"), ("breakout", "15m"), ("tsmom", "1h"), ("tsmom", "15m"),
          ("xsmom", "1h"), ("xsmom", "15m")]
DEPOSIT = 25.0
SPLIT = "2026-04-01"   # граница: до неё — период разработки, после — бывшие отложенные данные


def chain(panel, cfg, fold_rows, equity):
    trades, eqs, state, halted = [], [], None, False
    for r in fold_rows:
        if r["params"] is None:
            eqs.append(pd.DataFrame({"ts": [r["test_end"]], "equity": [equity], "cash": [equity], "position": [0]}))
            continue
        res = W.run_window(r["family"], r["tf"], r["params"], r["test_start"], r["test_end"], equity, state=state,
                           panel=panel, cfg=cfg)
        equity, state, halted = res.final_equity, res.risk_state, res.halted or halted
        if len(res.trades):
            trades.append(res.trades)
        e = res.equity
        eqs.append(e[(e["ts"] > r["test_start"]) & (e["ts"] <= r["test_end"])])
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return tr, pd.concat(eqs, ignore_index=True), equity, halted


def pf(tr) -> float:
    if not len(tr):
        return float("nan")
    p = tr["net_pnl"]
    loss = -p[p < 0].sum()
    return float(p[p > 0].sum() / loss) if loss > 0 else float("inf")


def equity_at(eq: pd.DataFrame, t_ms: int, default: float) -> float:
    prior = eq[eq["ts"] <= t_ms]
    return float(prior["equity"].iloc[-1]) if len(prior) else default


def main() -> None:
    base = load_config()
    panel = load_basket_holdout(base, HOLDOUT_CONFIRM, "после финальной проверки: разведка 1h/15m (данные уже не чистые)")
    W.QUARTERS = QUARTERS
    cfg2 = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": 0.02,
                                                                         "starting_equity_usdt": DEPOSIT})})
    start, end, split = to_ms(QUARTERS[0]), to_ms(QUARTERS[-1]), to_ms(SPLIT)
    bh = basket_bh(start, end, base.costs.taker_fee, panel)
    bh = pd.concat([pd.Series([1.0], index=[bh.index[0] - pd.Timedelta(days=1)]), bh])
    bhm = path_metrics(bh)
    bh_split = float(bh[bh.index <= pd.Timestamp(split, unit="ms", tz="UTC")].iloc[-1])
    L = ["Разведка: таймфреймы 1h и 15m против 4h, счёт 25 USDT (данные 2024-01 — 2026-10, не чистая проверка)",
         "Подбор параметров раз в квартал по процедуре протокола; R — результат сделки в единицах риска.",
         f"Buy & hold корзины: весь период {bhm['total_return'] * 100:+.0f}% (макс. просадка {bhm['max_drawdown'] * 100:.0f}%); "
         f"2024-01 — 2026-04 {(bh_split - 1) * 100:+.0f}%; 2026-04 — 2026-10 {(bh.iloc[-1] / bh_split - 1) * 100:+.0f}%", ""]
    saved = {}
    pool = W.make_pool(panel, cfg2)
    try:
        for fam, tf in COMBOS:
            t0 = time.time()
            wf = W.walk_forward(fam, tf, pool)
            rows = [{**r, "family": fam, "tf": tf} for r in wf.fold_rows]
            ref = wf.trades_ref
            gross = (ref["gross_pnl"] / ref["planned_loss"]).mean() if len(ref) else np.nan
            costs_r = ((ref["entry_fee"] + ref["exit_fee"] + ref["funding"]) / ref["planned_loss"]).mean() if len(ref) else np.nan
            L.append(f"{fam:9s} {tf:3s}: кварталов с торговлей {sum(r['params'] is not None for r in rows)} из {len(rows)}; "
                     f"сделок {len(ref)}; средний R {ref['r_multiple'].mean() if len(ref) else float('nan'):+.3f} "
                     f"(до издержек {gross:+.3f}, издержки {costs_r:.3f} R на сделку)  [{time.time() - t0:.0f} с]")
            for rpt in (0.02, 0.05):
                cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": rpt,
                                                                                    "starting_equity_usdt": DEPOSIT})})
                tr, eq, fin, halted = chain(panel, cfg, rows, DEPOSIT)
                e = eq["equity"]
                dd = float((1 - e / e.cummax()).max()) if len(e) else 0.0
                mid = equity_at(eq, split, DEPOSIT)
                m = path_metrics(daily_returns(eq, start, DEPOSIT)) if len(eq) else {"sharpe": np.nan}
                L.append(f"    риск {rpt * 100:.0f}%: 25 → {mid:7.2f} к 2026-04 → {fin:7.2f} к 2026-10; сделок {len(tr)}, "
                         f"PF {pf(tr):.2f}, макс. просадка {dd * 100:.0f}%, Шарп {m['sharpe']:.2f}"
                         f"{', ОСТАНОВЛЕН по просадке 80 %' if halted else ''}")
                saved[(fam, tf, rpt)] = {"trades": tr, "equity": eq}
    finally:
        pool.close()
    out = PROJECT_ROOT / "reports" / "basket_lowtf.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    with open(W.OUT_DIR / "lowtf.pkl", "wb") as fh:
        pickle.dump(saved, fh)
    print("\n".join(L))


if __name__ == "__main__":
    main()
