"""Независимая проверка портфельных сделок на реальных данных → reports/basket_audit.txt.

Для нескольких стратегий корзины (без подбора, первая конфигурация сетки) на всём
периоде разработки пересчитываются заново, без кода движка:
  - цена входа (открытие 15m-свечи ± проскальзывание), комиссии, финансирование
    (выплаты в (вход, выход]), итоговый результат сделки;
  - цена выхода по стопу/тейку лежит внутри диапазона свечи выхода (с проскальзыванием);
  - не больше 4 позиций одновременно, итоговый капитал = начальный + Σ результатов.
Это проверка учёта, а не оценка стратегий.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig
from bot.config import PROJECT_ROOT, load_config
from bot.data.panel import load_basket_dev
from bot.strategy.basket import COMBOS, grid_configs, make


def audit(res, panel, costs, equity0: float) -> dict:
    tr = res.trades
    worst = {"entry": 0.0, "funding": 0.0, "net": 0.0}
    bad_range = 0
    for t in tr.itertuples():
        r = panel.row(t.symbol)
        sd = panel.sym[t.symbol]
        i_in = int(np.searchsorted(panel.ts, t.entry_ts))
        entry = panel.open[r, i_in] * (1 + t.side * costs.slippage)
        sel = (sd.funding_ts > t.entry_ts) & (sd.funding_ts <= t.exit_ts)
        fidx = np.searchsorted(panel.ts, sd.funding_ts[sel])
        funding = float(np.sum(t.side * t.qty * panel.open[r, fidx] * sd.funding_rate[sel]))
        fee_out = {"take_profit": costs.maker_fee, "liquidation": 0.0}.get(t.exit_reason, costs.taker_fee)
        net = t.side * t.qty * (t.exit_price - entry) - t.qty * entry * costs.taker_fee \
            - t.qty * t.exit_price * fee_out - funding
        worst["entry"] = max(worst["entry"], abs(entry - t.entry_price) / t.entry_price)
        worst["funding"] = max(worst["funding"], abs(funding - t.funding))
        worst["net"] = max(worst["net"], abs(net - t.net_pnl))
        i_out = int(np.searchsorted(panel.ts, t.exit_ts))
        lo = panel.low[r, i_out] * (1 - costs.slippage) - 1e-12
        hi = panel.high[r, i_out] * (1 + costs.slippage) + 1e-12
        if t.exit_reason in ("stop", "take_profit") and not (lo <= t.exit_price <= hi):
            bad_range += 1
    # одновременно открытые позиции
    ev = pd.concat([pd.DataFrame({"t": tr["entry_ts"], "d": 1}), pd.DataFrame({"t": tr["exit_ts"], "d": -1})])
    ev = ev.sort_values(["t", "d"])          # выход раньше входа в одну и ту же минуту
    max_open = int(ev["d"].cumsum().max()) if len(ev) else 0
    recon = equity0 + float(tr["net_pnl"].sum()) if len(tr) else equity0
    return {"trades": len(tr), "max_open": max_open, "recon_diff": abs(recon - res.final_equity),
            "bad_exit_range": bad_range, **{f"max_d_{k}": v for k, v in worst.items()}}


def main() -> None:
    cfg = load_config()
    panel = load_basket_dev(cfg)
    lines = ["Проверка учёта портфельного движка на реальных данных корзины (период разработки)", ""]
    for fam, tf in COMBOS:
        prm = grid_configs(fam)[0]
        for eq in (10.0, 1000.0):
            pc = PortfolioConfig(risk=cfg.risk, costs=cfg.costs, initial_equity=eq)
            res = PortfolioBacktester(panel, make(fam, tf, prm), pc).run()
            a = audit(res, panel, cfg.costs, eq)
            ok = a["max_open"] <= 4 and a["recon_diff"] < 1e-6 and a["bad_exit_range"] == 0 \
                and a["max_d_entry"] < 1e-12 and a["max_d_funding"] < 1e-9 and a["max_d_net"] < 1e-9
            lines.append(f"{fam:12s} {tf:3s} капитал {eq:6.0f}: сделок {a['trades']:5d}, макс. открыто {a['max_open']}, "
                         f"|Δ итога| {a['recon_diff']:.1e}, |Δ входа| {a['max_d_entry']:.1e}, "
                         f"|Δ фин.| {a['max_d_funding']:.1e}, |Δ сделки| {a['max_d_net']:.1e}, "
                         f"выход вне свечи {a['bad_exit_range']} → {'OK' if ok else 'ОШИБКА'}")
            assert ok, (fam, tf, eq, a)
    lines += ["", "Все проверки пройдены."]
    out = PROJECT_ROOT / "reports" / "basket_audit.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
