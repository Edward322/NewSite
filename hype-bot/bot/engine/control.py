"""Служебное для запуска движка: часы, блокировка второго экземпляра, сборка объектов, статус."""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bot.config import PROJECT_ROOT, Config

log = logging.getLogger(__name__)


class RealClock:
    """Время UTC в мс с поправкой на часы биржи (sync)."""

    def __init__(self):
        self.offset_ms = 0

    def now_ms(self) -> int:
        return int(time.time() * 1000) + self.offset_ms

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)

    def sync(self, server_ms: int) -> int:
        self.offset_ms = int(server_ms - time.time() * 1000)
        return self.offset_ms


class InstanceLock:
    """Не даёт запустить второй экземпляр бота в том же режиме (Windows и Linux)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.fh = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                self.fh.seek(0)
                msvcrt.locking(self.fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            self.fh = None
            return False
        self.fh.seek(0)
        self.fh.truncate()
        self.fh.write(str(os.getpid()))
        self.fh.flush()
        return True

    def release(self) -> None:
        if self.fh:
            self.fh.close()
            self.fh = None


def setup_logging(mode: str) -> Path:
    logs = PROJECT_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / f"bot_{mode}.log"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=5_000_000, backupCount=10, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    logging.getLogger("pybit").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    return path


def paths(cfg: Config, mode: str) -> tuple[Path, Path]:
    E = cfg.engine
    db = Path(E.db_path.format(mode=mode))
    data = Path(E.data_dir) / mode
    return (db if db.is_absolute() else PROJECT_ROOT / db), (data if data.is_absolute() else PROJECT_ROOT / data)


def make_session(cfg: Config, mode: str, key: str | None, secret: str | None):
    from pybit.unified_trading import HTTP
    X = cfg.exchange
    return HTTP(testnet=False, demo=(mode == "demo"), api_key=key, api_secret=secret, domain=X.domain, tld=X.tld,
                timeout=X.http_timeout_s, recv_window=X.recv_window_ms, max_retries=1)


def utc(ms: int | None) -> str:
    if not ms:
        return "—"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def status_text(db, mode: str, equity: float | None = None, guard=None) -> str:
    """Короткий отчёт о состоянии (для /status, status.bat и ежедневного «жив»)."""
    L = [f"Бот ({mode})"]
    halted = db.get("halted")
    if halted:
        L.append(f"⛔ ОСТАНОВЛЕН: {halted.get('reason')} ({utc(halted.get('ts'))})")
    if equity is not None:
        st = db.get("risk_state") or {}
        peak = st.get("peak_equity") or equity
        dd = max(0.0, 1 - equity / peak) if peak else 0.0
        L.append(f"Капитал бота: {equity:.2f} USDT (пик {peak:.2f}, просадка {dd * 100:.1f}%)")
    pos = db.positions()
    if pos:
        L.append(f"Позиций: {len(pos)}")
        for p in pos.values():
            L.append(f"  {'лонг' if p.side > 0 else 'шорт'} {p.symbol} {p.qty:g} по {p.entry_price:.6g}, "
                     f"стоп {p.stop:.6g}")
    else:
        L.append("Позиций нет")
    st = db.get("risk_state") or {}
    if st.get("blockers"):
        L.append("Блокировки новых входов: " + "; ".join(f"{k}: {v}" for k, v in st["blockers"].items()))
    if st.get("pause_until_ms", 0) > time.time() * 1000:
        L.append(f"Пауза до {utc(st['pause_until_ms'])}: {st.get('pause_reason')}")
    L.append(f"Последняя свеча 4h: {utc(db.get('last_bar_close'))}")
    tr = db.trades()
    if tr:
        net = sum(t["net_pnl"] for t in tr)
        L.append(f"Сделок закрыто: {len(tr)}, итог {net:+.2f} USDT")
    errs = db.events(since_ms=int(time.time() * 1000) - 86_400_000, levels=("error",))
    if errs:
        L.append(f"Ошибок за сутки: {len(errs)}; последняя: {errs[-1]['message'][:200]}")
    return "\n".join(L)
