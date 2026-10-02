"""Базовая линия: случайные входы (только период разработки).

Запуск: python -m research.stage3_random_baseline > reports/stage3_random_baseline.txt
Случайная сторона с вероятностью 10% на каждой свече, стоп 2×ATR, выход через 8 свечей.
Без издержек средний R должен быть ≈ 0 (движок не искажает результат); с издержками —
отрицательный: столько стоит торговля без перевеса.
"""
from __future__ import annotations

import numpy as np

from bot.backtest.engine import Backtester, EngineConfig
from bot.config import CostsCfg, load_config
from bot.data.bars import load_funding
from bot.data.split import load_dev
from bot.market import Instrument
from bot.strategy import indicators as ind
from bot.strategy.base import LONG, SHORT, Enter, Exit, Strategy


class RandomEntry(Strategy):
    name = "random"

    def __init__(self, tf: str, seed: int, k_stop: float = 2.0, hold: int = 8, p: float = 0.1):
        self.timeframe, self.seed, self.k, self.hold, self.p = tf, seed, k_stop, hold, p
        self.warmup_bars = 20

    def prepare(self, bars):
        rng = np.random.default_rng(self.seed)
        self.u, self.s = rng.random(len(bars)), rng.random(len(bars))
        self.atr, self.c = ind.atr(bars.high, bars.low, bars.close, 14), bars.close

    def on_bar(self, i, pos):
        if np.isnan(self.atr[i]):
            return None
        if pos is None:
            if self.u[i] < self.p:
                side = LONG if self.s[i] < 0.5 else SHORT
                return Enter(side, self.c[i] - side * self.k * self.atr[i])
            return None
        return Exit("time") if pos.bars_held >= self.hold else None


def main():
    cfg = load_config()
    md = load_dev(cfg)
    f_ts, f_r = load_funding(cfg.data_dir(), cfg.symbol)
    inst = Instrument.from_files(cfg.data_dir(), cfg.symbol)
    zero = CostsCfg(maker_fee=0, taker_fee=0, slippage=0, stop_penetration=0)
    for tf in ("15m", "1h"):
        for costs, lab in ((cfg.costs, "обычные"), (zero, "нулевые")):
            rs = []
            for seed in range(20):
                ec = EngineConfig(risk=cfg.risk, costs=costs, initial_equity=1000.0)
                rs.append(Backtester(md, f_ts, f_r, inst, RandomEntry(tf, seed), ec).run().trades["r_multiple"].to_numpy())
            allr, means = np.concatenate(rs), [x.mean() for x in rs]
            print(f"Случайные входы {tf}, издержки {lab}: сделок {len(allr)}, средний R {allr.mean():+.3f}, "
                  f"разброс по 20 сидам [{min(means):+.3f}, {max(means):+.3f}]")


if __name__ == "__main__":
    main()
