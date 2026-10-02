"""Бэктест основной стратегии и теневых вариантов на живых данных с начала работы бота.

Ордеров нет: варианты считаются тем же портфельным бэктестером по 15m-свечам, которые бот
скачал сам, с тем же капиталом, риском и ограничителями, что у основной стратегии.
Прогон «основная» — это и есть эталон для сравнения сделок движка с бэктестом (этап 4).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from bot.backtest.portfolio import PortfolioBacktester, PortfolioResult
from bot.config import Config
from bot.data.panel import load_panel
from bot.runtime import make_strategy, portfolio_config

MAIN = "основная"


@dataclass
class VariantResult:
    name: str
    res: PortfolioResult
    equity0: float

    @property
    def trades(self) -> pd.DataFrame:
        return self.res.trades

    def last_equity(self) -> float:
        e = self.res.equity
        return float(e["equity"].iloc[-1]) if len(e) else self.equity0

    def max_dd(self) -> float:
        e = pd.concat([pd.Series([self.equity0]), self.res.equity["equity"]])
        return float((1 - e / e.cummax()).max())

    def pf(self) -> float:
        if not len(self.trades):
            return float("nan")
        p = self.trades["net_pnl"]
        loss = -p[p < 0].sum()
        return float(p[p > 0].sum() / loss) if loss > 0 else float("inf")

    def line(self) -> str:
        n = len(self.trades)
        net = float(self.trades["net_pnl"].sum()) if n else 0.0
        ret = self.last_equity() / self.equity0 - 1
        return (f"{self.name}: сделок {n}, закрытые {net:+.2f} USDT, капитал {self.last_equity():.2f} ({ret * 100:+.1f}%), "
                f"макс. просадка {self.max_dd() * 100:.1f}%, PF {self.pf():.2f}"
                f"{', ОСТАНОВЛЕН' if self.res.halted else ''}")


def run_variants(cfg: Config, data_dir: Path, anchor_ms: int, start_ms: int, end_ms: int, equity: float,
                 shadows: bool = True) -> list[VariantResult]:
    panel = load_panel(Path(data_dir), cfg.strategy.tradable, cfg.strategy.signal_only, anchor_ms, end_ms)
    variants = [(MAIN, {})] + (list(cfg.strategy.shadows.items()) if shadows else [])
    out = []
    for name, extra in variants:
        pc = portfolio_config(cfg, equity, start_ms=start_ms, close_at_end=False)
        res = PortfolioBacktester(panel, make_strategy(cfg, extra), pc).run()
        out.append(VariantResult(name, res, equity))
    return out
