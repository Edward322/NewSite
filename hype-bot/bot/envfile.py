"""Минимальное чтение .env без сторонних зависимостей.

Значения из .env не логируются и не печатаются.
"""
from __future__ import annotations

import os
from pathlib import Path

from bot.config import PROJECT_ROOT


def load_env(path: str | Path | None = None, override: bool = False) -> dict[str, str]:
    """Читает KEY=VALUE из .env в os.environ. Возвращает прочитанные пары."""
    path = Path(path) if path else PROJECT_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        loaded[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return loaded


def get_keys(mode: str) -> tuple[str, str]:
    """Ключи для режима 'demo' или 'live'. Бросает ошибку, если ключей нет."""
    prefix = {"demo": "BYBIT_DEMO", "live": "BYBIT_LIVE"}[mode]
    key = os.environ.get(f"{prefix}_API_KEY", "").strip()
    secret = os.environ.get(f"{prefix}_API_SECRET", "").strip()
    if not key or not secret:
        raise RuntimeError(
            f"Не заданы {prefix}_API_KEY / {prefix}_API_SECRET в файле .env"
        )
    return key, secret
