"""Загрузка и проверка конфигурации из YAML.

Все параметры риска живут в конфиге; код не содержит «магических» чисел риска.
Ключи API в конфиге не хранятся — только в .env (см. bot/envfile.py).
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"
EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "config.example.yaml"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExchangeCfg(_Strict):
    domain: Literal["bybit", "bytick"] = "bybit"
    tld: str = "com"
    recv_window_ms: int = Field(5000, ge=1000, le=60000)
    http_timeout_s: float = Field(10, gt=0, le=60)


class DataCfg(_Strict):
    dir: str = "data/raw"
    extra_symbols: list[str] = Field(default_factory=list)
    extra_history_days: int = Field(730, ge=30)
    request_pause_s: float = Field(0.08, ge=0.02)


class BasketCfg(_Strict):
    history_start: str = "2023-01-01"
    kline_interval: str = "15"
    include: list[str] = Field(default_factory=lambda: ["HYPEUSDT"])
    signal_only: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    n_max: int = Field(24, ge=1, le=100)
    listed_before: str = "2024-01-01"
    min_turnover_24h: float = Field(30_000_000, ge=0)
    max_min_order_usdt: float = Field(5.5, gt=0)


class ResearchCfg(_Strict):
    dev_start: str = "2025-01-01"
    holdout_months: int = Field(6, ge=1, le=24)
    holdout_start: str = "2026-04-02"
    holdout_end: str = "2026-10-02T09:28:00Z"


class CostsCfg(_Strict):
    maker_fee: float = Field(ge=0, le=0.01)
    taker_fee: float = Field(ge=0, le=0.01)
    slippage: float = Field(0.0002, ge=0, le=0.05)
    stop_penetration: float = Field(0.25, ge=0, le=1)


class RiskCfg(_Strict):
    starting_equity_usdt: float = Field(gt=0)
    max_leverage: float | None = Field(None, ge=1, le=200)  # None — без собственного лимита
    risk_per_trade: float = Field(gt=0, le=0.2)
    daily_loss_limit: float = Field(gt=0, lt=1)
    max_drawdown: float = Field(gt=0, lt=1)
    loss_streak_pause_trades: int = Field(ge=1)
    loss_streak_pause_hours: float = Field(gt=0)
    day_boundary: Literal["UTC"] = "UTC"
    margin_mode: Literal["ISOLATED_MARGIN"] = "ISOLATED_MARGIN"
    position_mode: Literal["one_way"] = "one_way"
    sl_trigger_by: Literal["LastPrice", "MarkPrice"] = "LastPrice"
    liq_distance_vs_stop_min: float = Field(2.0, ge=1.2)

    @model_validator(mode="after")
    def _limits_consistent(self) -> "RiskCfg":
        # Дневной лимит должен вмещать хотя бы одну полную потерю по стопу,
        # иначе первая же убыточная сделка останавливает торговлю на день.
        if self.daily_loss_limit < self.risk_per_trade:
            raise ValueError("daily_loss_limit меньше risk_per_trade")
        if self.max_drawdown < self.daily_loss_limit:
            raise ValueError("max_drawdown меньше daily_loss_limit")
        return self


class LiveCfg(_Strict):
    enabled: bool = False
    min_size_only: bool = True


class Config(_Strict):
    symbol: str = "HYPEUSDT"
    category: Literal["linear"] = "linear"
    exchange: ExchangeCfg = ExchangeCfg()
    data: DataCfg = DataCfg()
    basket: BasketCfg = BasketCfg()
    research: ResearchCfg = ResearchCfg()
    costs: CostsCfg
    risk: RiskCfg
    live: LiveCfg = LiveCfg()

    def data_dir(self) -> Path:
        p = Path(self.data.dir)
        return p if p.is_absolute() else PROJECT_ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    """Читает config/config.yaml, а если его нет — шаблон config.example.yaml."""
    if path is None:
        path = DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else EXAMPLE_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return Config.model_validate(raw)
