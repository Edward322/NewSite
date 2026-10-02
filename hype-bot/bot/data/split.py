"""Разделение истории: данные для разработки и отложенные данные.

Отложенный период (holdout) открывается только явным вызовом load_holdout с
подтверждением; каждое обращение пишется в reports/holdout_access.log, чтобы
было видно, сколько раз и когда отложенные данные использовались.
"""
from __future__ import annotations

from datetime import datetime, timezone

from bot.config import PROJECT_ROOT, Config
from bot.data.bars import MinuteData, load_minutes, to_ms

HOLDOUT_CONFIRM = "FINAL-HOLDOUT-CHECK"
ACCESS_LOG = PROJECT_ROOT / "reports" / "holdout_access.log"


def dev_range(cfg: Config) -> tuple[int, int]:
    return to_ms(cfg.research.dev_start), to_ms(cfg.research.holdout_start)


def holdout_range(cfg: Config) -> tuple[int, int]:
    return to_ms(cfg.research.holdout_start), to_ms(cfg.research.holdout_end)


def load_dev(cfg: Config, symbol: str | None = None, warmup_days: int = 0) -> MinuteData:
    """Данные разработки. warmup_days — история ДО начала только для прогрева индикаторов."""
    start, end = dev_range(cfg)
    return load_minutes(cfg.data_dir(), symbol or cfg.symbol, start - warmup_days * 86_400_000, end)


def load_holdout(cfg: Config, confirm: str, reason: str, symbol: str | None = None,
                 warmup_days: int = 0) -> MinuteData:
    if confirm != HOLDOUT_CONFIRM:
        raise PermissionError("Отложенные данные открываются только для финальной проверки")
    ACCESS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ACCESS_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{symbol or cfg.symbol}\t{reason}\n")
    start, end = holdout_range(cfg)
    return load_minutes(cfg.data_dir(), symbol or cfg.symbol, start - warmup_days * 86_400_000, end)
