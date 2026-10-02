"""Состояние боевого движка в SQLite — переживает перезапуск, отключение питания и обрыв связи.

Таблицы:
  kv         — разные значения (json): режим, смещение капитала, якорь данных, последняя
               обработанная свеча, состояние ограничителей, флаг остановки;
  orders     — каждое намерение отправить ордер ДО отправки (orderLinkId, статус, исполнение);
  positions  — открытые позиции бота (как их видит бот);
  trades     — закрытые сделки (для отчётов и сверки с бэктестом);
  events     — журнал: решения, пропуски, сверка, ошибки, ограничители;
  equity     — снимки капитала.
Каждая запись фиксируется сразу (autocommit), журнал WAL.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS orders (
    link_id TEXT PRIMARY KEY, symbol TEXT NOT NULL, purpose TEXT NOT NULL, side INTEGER NOT NULL,
    qty REAL NOT NULL, ref_price REAL, stop REAL, decision_ts INTEGER, created_ms INTEGER NOT NULL,
    status TEXT NOT NULL, order_id TEXT, filled_qty REAL, avg_price REAL, fee REAL, error TEXT,
    extra TEXT);
CREATE TABLE IF NOT EXISTS positions (
    symbol TEXT PRIMARY KEY, side INTEGER NOT NULL, qty REAL NOT NULL, entry_price REAL NOT NULL,
    entry_ts INTEGER NOT NULL, decision_ts INTEGER NOT NULL, stop REAL NOT NULL, stop_initial REAL NOT NULL,
    leverage REAL NOT NULL, margin REAL NOT NULL, planned_loss REAL NOT NULL, ref_price REAL NOT NULL,
    entry_fee REAL NOT NULL, link_id TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side INTEGER, qty REAL, decision_ts INTEGER,
    entry_ts INTEGER, entry_price REAL, ref_price REAL, exit_ts INTEGER, exit_price REAL, exit_ref REAL,
    exit_reason TEXT, stop_initial REAL, stop_final REAL, planned_loss REAL, entry_fee REAL, exit_fee REAL,
    funding REAL, gross_pnl REAL, net_pnl REAL, r_multiple REAL, leverage REAL, entry_link TEXT, exit_link TEXT);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL,
    message TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS equity (ts INTEGER PRIMARY KEY, equity REAL NOT NULL, wallet REAL, positions INTEGER);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
"""


@dataclass
class DbPosition:
    symbol: str
    side: int
    qty: float
    entry_price: float
    entry_ts: int
    decision_ts: int
    stop: float
    stop_initial: float
    leverage: float
    margin: float
    planned_loss: float
    ref_price: float
    entry_fee: float
    link_id: str


@dataclass
class DbOrder:
    link_id: str
    symbol: str
    purpose: str            # entry | exit | kill
    side: int
    qty: float
    ref_price: float | None
    stop: float | None
    decision_ts: int | None
    created_ms: int
    status: str             # pending | sent | filled | rejected | failed | unknown
    order_id: str | None = None
    filled_qty: float | None = None
    avg_price: float | None = None
    fee: float | None = None
    error: str | None = None
    extra: str | None = None


class StateDB:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.c = sqlite3.connect(str(self.path), isolation_level=None, timeout=30)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.c.execute("PRAGMA synchronous=FULL")
        self.c.executescript(SCHEMA)

    def close(self) -> None:
        self.c.close()

    # ------------------------------------------------------------------ kv
    def get(self, key: str, default: Any = None) -> Any:
        row = self.c.execute("SELECT value FROM kv WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def set(self, key: str, value: Any) -> None:
        self.c.execute("INSERT INTO kv(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                       (key, json.dumps(value, ensure_ascii=False)))

    def delete(self, key: str) -> None:
        self.c.execute("DELETE FROM kv WHERE key=?", (key,))

    # -------------------------------------------------------------- orders
    def add_order(self, o: DbOrder) -> None:
        d = asdict(o)
        cols = ",".join(d)
        self.c.execute(f"INSERT INTO orders({cols}) VALUES({','.join('?' * len(d))})", tuple(d.values()))

    def update_order(self, link_id: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.c.execute(f"UPDATE orders SET {sets} WHERE link_id=?", (*fields.values(), link_id))

    def order(self, link_id: str) -> DbOrder | None:
        row = self.c.execute("SELECT * FROM orders WHERE link_id=?", (link_id,)).fetchone()
        return DbOrder(**dict(row)) if row else None

    def orders(self, status: tuple[str, ...] | None = None) -> list[DbOrder]:
        if status:
            q = f"SELECT * FROM orders WHERE status IN ({','.join('?' * len(status))}) ORDER BY created_ms"
            rows = self.c.execute(q, status).fetchall()
        else:
            rows = self.c.execute("SELECT * FROM orders ORDER BY created_ms").fetchall()
        return [DbOrder(**dict(r)) for r in rows]

    # ----------------------------------------------------------- positions
    def positions(self) -> dict[str, DbPosition]:
        return {r["symbol"]: DbPosition(**dict(r)) for r in self.c.execute("SELECT * FROM positions")}

    def put_position(self, p: DbPosition) -> None:
        d = asdict(p)
        self.c.execute(f"INSERT OR REPLACE INTO positions({','.join(d)}) VALUES({','.join('?' * len(d))})",
                       tuple(d.values()))

    def update_position(self, symbol: str, **fields) -> None:
        sets = ",".join(f"{k}=?" for k in fields)
        self.c.execute(f"UPDATE positions SET {sets} WHERE symbol=?", (*fields.values(), symbol))

    def remove_position(self, symbol: str) -> None:
        self.c.execute("DELETE FROM positions WHERE symbol=?", (symbol,))

    # -------------------------------------------------------------- trades
    def add_trade(self, t: dict) -> int:
        cols = ",".join(t)
        cur = self.c.execute(f"INSERT INTO trades({cols}) VALUES({','.join('?' * len(t))})", tuple(t.values()))
        return int(cur.lastrowid)

    def trades(self, since_ms: int = 0) -> list[dict]:
        return [dict(r) for r in self.c.execute("SELECT * FROM trades WHERE exit_ts>=? ORDER BY exit_ts, id",
                                                (since_ms,))]

    # -------------------------------------------------------------- events
    def event(self, ts: int, level: str, kind: str, message: str) -> None:
        self.c.execute("INSERT INTO events(ts, level, kind, message) VALUES(?,?,?,?)", (ts, level, kind, message))

    def events(self, since_ms: int = 0, kinds: tuple[str, ...] | None = None, levels: tuple[str, ...] | None = None
               ) -> list[dict]:
        q, args = "SELECT * FROM events WHERE ts>=?", [since_ms]
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += list(kinds)
        if levels:
            q += f" AND level IN ({','.join('?' * len(levels))})"
            args += list(levels)
        return [dict(r) for r in self.c.execute(q + " ORDER BY ts, id", args)]

    # -------------------------------------------------------------- equity
    def snapshot(self, ts: int, equity: float, wallet: float | None, positions: int) -> None:
        self.c.execute("INSERT OR REPLACE INTO equity(ts, equity, wallet, positions) VALUES(?,?,?,?)",
                       (ts, equity, wallet, positions))

    def equity_rows(self, since_ms: int = 0) -> list[dict]:
        return [dict(r) for r in self.c.execute("SELECT * FROM equity WHERE ts>=? ORDER BY ts", (since_ms,))]
