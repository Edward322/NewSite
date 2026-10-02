"""Уведомления. Без настроенного Telegram — только запись в журнал."""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class LogNotifier:
    def send(self, text: str, important: bool = False) -> None:
        log.info("[уведомление] %s", text)

    def start_commands(self, engine) -> None: ...


def make_notifier(db, mode: str):
    return LogNotifier()
