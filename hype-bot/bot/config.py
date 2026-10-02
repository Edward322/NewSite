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
BOT_CONFIG = PROJECT_ROOT / "config" / "bot.yaml"
BOT_EXAMPLE_CONFIG = PROJECT_ROOT / "config" / "bot.example.yaml"


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
    # Пауза после серии убытков; None — правило выключено (решение пользователя для корзины).
    loss_streak_pause_trades: int | None = Field(None, ge=1)
    loss_streak_pause_hours: float = Field(24, gt=0)
    # Ступени снижения риска по просадке от пика: [[0.20, 0.5]] — при просадке ≥ 20 % риск × 0.5.
    drawdown_steps: list[tuple[float, float]] = Field(default_factory=list)
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
        for dd, factor in self.drawdown_steps:
            if not (0 < dd < self.max_drawdown) or not (0 < factor <= 1):
                raise ValueError("ступень просадки должна быть меньше max_drawdown, множитель — в (0, 1]")
        return self


class LiquidityCfg(_Strict):
    days: int = Field(30, ge=5)
    max_tick_frac: float = Field(0.0004, gt=0)
    min_turnover: float = Field(10_000_000, ge=0)
    max_zero_volume_frac: float = Field(0.01, ge=0, le=1)


class PortfolioCfg(_Strict):
    """Портфельные правила риска (docs/RISK_PROTOCOL.md); общие для бэктеста и движка."""
    max_positions: int = Field(4, ge=1, le=20)
    max_open_risk: float = Field(0.06, gt=0, le=1)
    downsize_to_fit: bool = True
    corr_cap: float | None = Field(None, gt=0, le=1)
    corr_days: int = Field(30, ge=5)
    vol_scaling: bool = False
    vol_days: int = Field(30, ge=5)
    vol_ref_days: int = Field(365, ge=30)
    vol_ref_min_days: int = Field(180, ge=10)
    liquidity: LiquidityCfg | None = LiquidityCfg()

    @model_validator(mode="after")
    def _caps(self) -> "PortfolioCfg":
        if self.corr_cap is not None and self.corr_cap > self.max_open_risk:
            raise ValueError("corr_cap не может быть больше max_open_risk")
        return self

    def rules(self):
        from bot.risk.portfolio import LiquidityRule, PortfolioRules
        liq = LiquidityRule(**self.liquidity.model_dump()) if self.liquidity else None
        d = self.model_dump(exclude={"liquidity"})
        return PortfolioRules(**d, liquidity=liq)


MAIN_PARAMS = {"n": 38, "k_stop": 2.5, "k_trail": 4.0, "sides": "both", "regime": "none"}
BASKET = ["XRPUSDT", "NEARUSDT", "MOVRUSDT", "QNTUSDT", "DOGEUSDT", "AAVEUSDT", "WLDUSDT", "1000PEPEUSDT",
          "ADAUSDT", "SANDUSDT", "LINKUSDT", "UNIUSDT", "AVAXUSDT", "HBARUSDT", "HYPEUSDT"]


class StrategyCfg(_Strict):
    """Пробой канала на корзине с фиксированными параметрами (центр плато) и теневые варианты."""
    timeframe: str = "4h"
    params: dict = Field(default_factory=lambda: dict(MAIN_PARAMS))
    tradable: list[str] = Field(default_factory=lambda: list(BASKET))
    signal_only: list[str] = Field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    shadows: dict[str, dict] = Field(default_factory=lambda: {
        "S1 боковой рынок (ADX ≥ 20)": {"adx_min": 20.0},
        "S2 только лонг в бычьем режиме": {"bull_long_only": True},
        "S3 оба фильтра": {"adx_min": 20.0, "bull_long_only": True},
    })

    @model_validator(mode="after")
    def _tf(self) -> "StrategyCfg":
        from bot.data.bars import TF_MINUTES
        if TF_MINUTES.get(self.timeframe, 0) < 240:
            raise ValueError("таймфрейм стратегии не ниже 4h")
        if len(self.shadows) > 3:
            raise ValueError("не больше 3 теневых вариантов")
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
    portfolio: PortfolioCfg = PortfolioCfg()
    strategy: StrategyCfg = StrategyCfg()

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


def load_bot_config(path: str | Path | None = None) -> Config:
    """Конфиг торгового бота: config/bot.yaml, а если его нет — шаблон config/bot.example.yaml."""
    if path is None:
        path = BOT_CONFIG if BOT_CONFIG.exists() else BOT_EXAMPLE_CONFIG
    return load_config(path)
