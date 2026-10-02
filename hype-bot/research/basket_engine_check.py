"""Проверка портфельного движка на реальных данных → reports/basket_engine_check.txt.

1. Для каждой торгуемой монеты корзины: стратегии одной монеты прогоняются старым
   движком (bot/backtest/engine.py) и портфельным (одна монета в панели) — сделки
   должны совпасть полностью.
2. Независимый пересчёт итогового капитала из сделок (сумма net_pnl).
Только период разработки.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from bot.backtest.engine import Backtester, EngineConfig
from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig, SingleSymbol
from bot.config import PROJECT_ROOT, load_config
from bot.data.panel import load_basket_dev
from bot.strategy.candidates import make

CASES = [("donchian", "4h", {"n": 40, "k_stop": 2.5, "k_trail": 3.0, "er_min": 0.0}),
         ("meanrev", "1h", {"n": 50, "z_in": 2.5, "k_stop": 2.5, "er_max": 1.0}),
         ("momentum", "1h", {"L": 48, "z_in": 1.5, "k_stop": 3.0, "er_min": 0.0})]


def main() -> None:
    cfg = load_config()
    full = load_basket_dev(cfg)
    lines = ["Портфельный движок против движка одной монеты (реальные 15m-данные, период разработки)", ""]
    worst = 0.0
    for sym in full.symbols:
        sub = load_basket_dev(cfg)
        sub.symbols, sub.signal_only = [sym], []
        r = full.row(sym)
        for c in ("open", "high", "low", "close", "volume", "turnover"):
            setattr(sub, c, getattr(full, c)[r:r + 1])
        ok = ~np.isnan(sub.close[0])
        first = int(np.argmax(ok))
        sub = sub.slice(int(sub.ts[first]), None)
        md, sd = sub.minute_data(sym), sub.sym[sym]
        for fam, tf, prm in CASES:
            for eq in (10.0, 1000.0):
                ec = EngineConfig(risk=cfg.risk, costs=cfg.costs, initial_equity=eq)
                t0 = time.time()
                old = Backtester(md, sd.funding_ts, sd.funding_rate, sd.inst, make(fam, tf, prm), ec).run()
                t1 = time.time()
                pc = PortfolioConfig(risk=cfg.risk, costs=cfg.costs, initial_equity=eq)
                new = PortfolioBacktester(sub, SingleSymbol(make(fam, tf, prm)), pc).run()
                t2 = time.time()
                cols = list(old.trades.columns)
                same = len(old.trades) == len(new.trades) and (
                    len(old.trades) == 0 or np.allclose(old.trades[cols].select_dtypes("number").to_numpy(),
                                                        new.trades[cols].select_dtypes("number").to_numpy(),
                                                        rtol=1e-10, atol=1e-12, equal_nan=True))
                diff = abs(new.final_equity - old.final_equity)
                worst = max(worst, diff)
                recon = eq + (new.trades["net_pnl"].sum() if len(new.trades) else 0.0)
                lines.append(f"{sym:13s} {fam:9s} {tf:3s} капитал {eq:6.0f}: сделок {len(old.trades):4d}/{len(new.trades):4d} "
                             f"совпали={same} итог {old.final_equity:12.4f}/{new.final_equity:12.4f} "
                             f"пересчёт из сделок {recon:12.4f}  ({t1 - t0:.1f} с / {t2 - t1:.1f} с)")
                assert same and diff < 1e-8 and abs(recon - new.final_equity) < 1e-6, (sym, fam, tf, eq)
    lines += ["", f"Все комбинации совпали; макс. расхождение итогового капитала {worst:.2e} USDT."]
    out = PROJECT_ROOT / "reports" / "basket_engine_check.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    main()
