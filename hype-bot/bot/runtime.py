"""Сборка стратегии и портфельного бэктеста из конфига — один путь для исследования,
боевого движка, теневых вариантов и скрипта сравнения с демо."""
from __future__ import annotations

from bot.backtest.portfolio import PortfolioConfig
from bot.config import Config
from bot.risk.guards import RiskState
from bot.strategy.basket import Breakout, make_breakout


def make_strategy(cfg: Config, extra: dict | None = None) -> Breakout:
    return make_breakout(cfg.strategy.timeframe, {**cfg.strategy.params, **(extra or {})})


def portfolio_config(cfg: Config, equity: float, start_ms: int | None = None, end_ms: int | None = None,
                     state: RiskState | None = None, close_at_end: bool = True) -> PortfolioConfig:
    rules = cfg.portfolio.rules()
    return PortfolioConfig(risk=cfg.risk, costs=cfg.costs, initial_equity=equity,
                           max_positions=rules.max_positions, max_open_risk=rules.max_open_risk,
                           trade_start_ms=start_ms, trade_end_ms=end_ms, close_at_end=close_at_end,
                           initial_risk_state=state, rules=rules)


def warmup_days(cfg: Config) -> int:
    """Сколько дней истории нужно до первой свечи решения (индикаторы, корреляция, ликвидность)."""
    p = cfg.portfolio
    need = [120, p.corr_days + 2, (p.liquidity.days + 2) if p.liquidity else 0, 51]   # 51 — EMA BTC 50 дн. у S2
    if p.vol_scaling:
        need.append(p.vol_ref_days + p.vol_days + 2)
    return max(need)
