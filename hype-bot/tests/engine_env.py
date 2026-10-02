"""Окружение для тестов движка: синтетическая корзина → имитатор биржи → движок."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bot.backtest.portfolio import PortfolioBacktester
from bot.config import Config, LiquidityCfg, load_bot_config
from bot.data.panel import DAY_MS, load_panel
from bot.engine.data import LiveData
from bot.engine.engine import Engine
from bot.engine.state import StateDB
from bot.exchange.client import BybitClient
from bot.exchange.sim import SimBybit, SimClock, write_market_dir
from bot.runtime import make_strategy, portfolio_config
from tests.helpers import T0
from tests.instruments import HYPE_INSTRUMENT, HYPE_TIERS
from tests.test_basket_strategies import synth_panel

INSTRUMENT = {**HYPE_INSTRUMENT, "priceFilter": {"tickSize": "0.0001"}}
TRADABLE = [f"C{i}USDT" for i in range(6)]
SIGNAL = ["BTCUSDT"]


def sim_config(equity: float = 1000.0, **risk_over) -> Config:
    base = load_bot_config()
    risk = base.risk.model_copy(update={"starting_equity_usdt": equity, **risk_over})
    return base.model_copy(update={
        "risk": risk,
        "strategy": base.strategy.model_copy(update={"tradable": TRADABLE, "signal_only": SIGNAL,
                                                     "params": {**base.strategy.params, "n": 20}}),
        "portfolio": base.portfolio.model_copy(update={"liquidity": LiquidityCfg(min_turnover=0)}),
        "engine": base.engine.model_copy(update={"ws_required": False, "loop_s": 300, "reconcile_every_s": 300}),
    })


@dataclass
class Env:
    cfg: Config
    clock: SimClock
    sim: SimBybit
    client: BybitClient
    db: StateDB
    data: LiveData
    engine: Engine
    tmp: Path

    def new_engine(self, stream=None, notifier=None) -> Engine:
        """Как после перезапуска процесса: новые объекты, то же состояние на диске и та же биржа."""
        self.db.close()
        self.db = StateDB(self.tmp / "state.sqlite")
        self.data = LiveData(self.client, self.tmp / "live", TRADABLE, SIGNAL, sleep=self.clock.sleep)
        self.engine = Engine(self.cfg, self.client, self.db, self.clock, self.data, mode="sim", base_dir=self.tmp,
                             stream=stream, notifier=notifier)
        return self.engine

    def run_until(self, t_ms: int) -> None:
        while self.clock.now_ms() < t_ms:
            self.engine.step()
            self.clock.sleep(self.cfg.engine.loop_s)

    def backtest(self, first_bar: int):
        anchor = int(self.db.get("anchor_ms"))
        panel = load_panel(self.tmp / "live", TRADABLE, SIGNAL, anchor, self.clock.now_ms())
        pc = portfolio_config(self.cfg, self.cfg.risk.starting_equity_usdt, start_ms=first_bar, close_at_end=False)
        return PortfolioBacktester(panel, make_strategy(self.cfg), pc).run()


def make_env(tmp: Path, start_day: float = 100, equity: float = 1000.0, wallet: float | None = None,
             panel=None, cfg: Config | None = None, seed: int = 0, stream=None, notifier=None) -> Env:
    panel = panel or synth_panel(seed=seed)
    write_market_dir(panel, tmp / "market", INSTRUMENT, HYPE_TIERS)
    cfg = cfg or sim_config(equity)
    clock = SimClock(T0 + int(start_day * DAY_MS) + 7 * 60_000)
    c = cfg.costs
    sim = SimBybit.from_dir(tmp / "market", TRADABLE + SIGNAL, clock, equity=wallet if wallet else equity,
                            taker_fee=c.taker_fee, maker_fee=c.maker_fee, slippage=c.slippage,
                            stop_penetration=c.stop_penetration)
    client = BybitClient(sim, sleep=clock.sleep)
    db = StateDB(tmp / "state.sqlite")
    data = LiveData(client, tmp / "live", TRADABLE, SIGNAL, sleep=clock.sleep)
    eng = Engine(cfg, client, db, clock, data, mode="sim", base_dir=tmp, stream=stream, notifier=notifier)
    return Env(cfg, clock, sim, client, db, data, eng, tmp)
