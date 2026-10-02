"""Запуск торгового движка.

    python -m bot.engine run    [--mode demo|live]   работать круглосуточно
    python -m bot.engine kill   [--mode demo|live]   аварийная остановка: закрыть позиции, отменить ордера
    python -m bot.engine resume [--mode demo|live]   снять остановку (после разбора причины)
    python -m bot.engine status [--mode demo|live]   состояние

Live включается только при `live.enabled: true` в config/bot.yaml И `--mode live`.
Коды выхода: 0 — штатная остановка (kill), 2 — ошибка настройки, 3 — бот уже запущен.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
import traceback

from bot.config import PROJECT_ROOT, load_bot_config
from bot.envfile import get_keys, load_env

log = logging.getLogger("bot.engine")


def _build(cfg, mode: str, with_stream: bool = True):
    from bot.engine.control import RealClock, make_session, paths
    from bot.engine.data import LiveData
    from bot.engine.engine import Engine, NullStream
    from bot.engine.notify import make_notifier
    from bot.engine.state import StateDB
    from bot.engine.stream import BybitStream
    from bot.exchange.client import BybitClient
    key, secret = get_keys(mode)
    client = BybitClient(make_session(cfg, mode, key, secret))
    db_path, data_dir = paths(cfg, mode)
    db = StateDB(db_path)
    clock = RealClock()
    data = LiveData(client, data_dir, cfg.strategy.tradable, cfg.strategy.signal_only)
    stream = BybitStream(cfg.strategy.tradable + cfg.strategy.signal_only, mode, key, secret, cfg.exchange.domain,
                         cfg.exchange.tld) if (with_stream and cfg.engine.ws_required) else NullStream()
    notifier = make_notifier(db, mode)
    eng = Engine(cfg, client, db, clock, data, mode=mode, notifier=notifier, stream=stream, base_dir=PROJECT_ROOT)
    return eng


def cmd_run(mode: str) -> int:
    from bot.engine.control import InstanceLock, paths, setup_logging
    cfg = load_bot_config()
    load_env()
    setup_logging(mode)
    db_path, _ = paths(cfg, mode)
    lock = InstanceLock(db_path.with_suffix(".lock"))
    if not lock.acquire():
        log.error("Бот в режиме %s уже запущен — второй экземпляр не нужен", mode)
        return 3
    try:
        eng = _build(cfg, mode)
    except (RuntimeError, PermissionError) as e:
        log.error("%s", e)
        return 2
    from bot.engine.reports import install_hooks
    install_hooks(eng)
    eng.notifier.start_commands(eng)
    started, errors, last_alert, last_sync = False, 0, 0.0, 0.0
    log.info("Запуск в режиме %s", mode)
    while True:
        try:
            if time.time() - last_sync > 3600:
                off = eng.clock.sync(eng.client.server_time_ms())
                last_sync = time.time()
                if abs(off) > 2000:
                    log.warning("Часы компьютера расходятся с биржей на %.1f с — включите синхронизацию времени Windows",
                                off / 1000)
            if not started:
                started = eng.start()
                if not started:
                    halted = eng.db.get("halted") or {}
                    if "аварийная" in str(halted.get("reason", "")):
                        log.info("Аварийная остановка — бот выключается")
                        return 0
                    for h in eng.hooks:                  # остановлен по просадке: только отчёты и команды
                        h(eng)
                    time.sleep(60)
                    continue
            res = eng.step()
            if res.killed:
                log.info("Аварийная остановка выполнена — бот выключается")
                return 0
            errors = 0
        except KeyboardInterrupt:
            log.info("Остановлено с клавиатуры (позиции остаются под биржевыми стопами)")
            return 0
        except Exception as e:  # движок не падает: ошибка в журнал и в Telegram, повтор с паузой
            errors += 1
            tb = traceback.format_exc()
            log.error("Ошибка цикла (%d подряд): %s\n%s", errors, e, tb)
            try:
                eng.db.event(eng.now(), "error", "loop", f"{e}\n{tb[-1500:]}")
            except Exception:
                pass
            if time.time() - last_alert > 1800 or errors == 1:
                eng.notifier.send(f"Ошибка бота ({errors} подряд): {str(e)[:300]}. Позиции защищены биржевыми "
                                  "стопами; бот повторит попытку.", important=True)
                last_alert = time.time()
            time.sleep(min(300, 10 * errors))
            continue
        time.sleep(cfg.engine.loop_s)


def cmd_kill(mode: str) -> int:
    from bot.engine.control import setup_logging
    cfg = load_bot_config()
    load_env()
    setup_logging(mode)
    (PROJECT_ROOT / cfg.engine.stop_file).write_text("аварийная остановка\n", encoding="utf-8")
    eng = _build(cfg, mode, with_stream=False)
    try:
        eng.data.refresh_instruments(eng.now())
    except Exception as e:
        log.warning("Параметры инструментов не обновлены: %s", e)
    eng.kill("команда kill")
    print("Аварийная остановка выполнена. Подробности — logs/bot_%s.log" % mode)
    return 0


def cmd_resume(mode: str, reset_peak: bool) -> int:
    from bot.engine.control import paths
    from bot.engine.state import StateDB
    cfg = load_bot_config()
    db_path, _ = paths(cfg, mode)
    db = StateDB(db_path)
    stop = PROJECT_ROOT / cfg.engine.stop_file
    if stop.exists():
        stop.unlink()
    h = db.get("halted")
    db.delete("halted")
    st = db.get("risk_state")
    if st and reset_peak:
        db.delete("risk_state")      # пик и дневной лимит начнутся заново от текущего капитала
    elif st:
        st["halted"], st["halt_reason"] = False, ""
        db.set("risk_state", st)
    print(f"Остановка снята ({h.get('reason') if h else 'не была включена'}).",
          "Пик капитала сброшен." if reset_peak else "")
    return 0


def cmd_status(mode: str) -> int:
    from bot.engine.control import paths, status_text
    from bot.engine.state import StateDB
    cfg = load_bot_config()
    db_path, _ = paths(cfg, mode)
    if not db_path.exists():
        print("Бот ещё не запускался в режиме", mode)
        return 0
    db = StateDB(db_path)
    rows = db.equity_rows()
    print(status_text(db, mode, rows[-1]["equity"] if rows else None))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m bot.engine")
    ap.add_argument("command", choices=["run", "kill", "resume", "status"])
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--reset-peak", action="store_true", help="resume: начать отсчёт пика заново")
    a = ap.parse_args(argv)
    if a.command == "run":
        return cmd_run(a.mode)
    if a.command == "kill":
        return cmd_kill(a.mode)
    if a.command == "resume":
        return cmd_resume(a.mode, a.reset_peak)
    return cmd_status(a.mode)


if __name__ == "__main__":
    sys.exit(main())
