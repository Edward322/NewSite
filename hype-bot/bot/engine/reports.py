"""Периодические задачи основного цикла: команды Telegram, ежедневное «жив», недельный отчёт."""
from __future__ import annotations

import logging
from datetime import datetime, timezone

from bot.engine.control import status_text

log = logging.getLogger(__name__)
HELP = ("Команды: /status — состояние бота; /stop — АВАРИЙНАЯ ОСТАНОВКА (закрыть позиции, отменить ордера, "
        "выключить бота; запуск снова — windows\\resume.bat на компьютере).")


def _equity(eng):
    try:
        return eng.equity()[0] if eng.started else None
    except Exception:
        return None


def handle_commands(eng) -> None:
    for cmd in eng.notifier.pop_commands():
        if cmd.startswith("/status"):
            eng.notifier.send(status_text(eng.db, eng.mode, _equity(eng)))
        elif cmd.startswith("/stop"):
            eng.stop_file().write_text("команда /stop из Telegram\n", encoding="utf-8")
            eng.notifier.send("Принято: аварийная остановка. Закрываю позиции, отменяю ордера, выключаюсь.",
                              important=True)
        else:
            eng.notifier.send(HELP)


def weekly_path(base_dir, mode: str, now_ms: int):
    d = base_dir / "reports" / "live"
    d.mkdir(parents=True, exist_ok=True)
    day = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    return d / f"weekly_{mode}_{day}.txt"


def periodic(eng) -> None:
    handle_commands(eng)
    E = eng.cfg.engine
    now = eng.now()
    dt = datetime.fromtimestamp(now / 1000, tz=timezone.utc)
    if dt.hour < E.daily_alive_hour_utc:
        return
    day = dt.strftime("%Y-%m-%d")
    if eng.db.get("last_alive_day") != day:
        eng.db.set("last_alive_day", day)
        eng.notifier.send("Жив. " + status_text(eng.db, eng.mode, _equity(eng)))
    week = dt.strftime("%G-W%V")
    if dt.weekday() == E.weekly_report_weekday and eng.db.get("last_weekly") != week:
        eng.db.set("last_weekly", week)
        from bot.report.weekly import weekly_text
        text = weekly_text(eng.cfg, eng.db, eng.data.dir, eng.mode, now, _equity(eng))
        weekly_path(eng.base_dir, eng.mode, now).write_text(text + "\n", encoding="utf-8")
        eng.notifier.send(text)


def install_hooks(engine) -> None:
    engine.hooks.append(periodic)
