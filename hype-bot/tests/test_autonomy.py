"""Автономность: Telegram (/status, /stop), ежедневное «жив», недельный отчёт с теневыми вариантами,
одна копия бота, мастер настройки .env."""
import queue

import pytest

from bot.data.panel import DAY_MS
from bot.engine.control import InstanceLock, status_text
from bot.engine.notify import TelegramNotifier
from bot.engine.reports import install_hooks
from tests.engine_env import make_env
from tests.helpers import T0


class Notes:
    """Уведомитель для тестов: копит сообщения, команды подаются вручную."""

    def __init__(self):
        self.sent, self.cmds = [], []

    def send(self, text, important=False):
        self.sent.append(text)

    def start_commands(self, engine=None): ...

    def pop_commands(self):
        out, self.cmds = self.cmds, []
        return out


def test_telegram_accepts_commands_only_from_own_chat(monkeypatch):
    calls = []
    updates = [{"ok": True, "result": [
        {"update_id": 1, "message": {"chat": {"id": 999}, "text": "/stop"}},
        {"update_id": 2, "message": {"chat": {"id": 42}, "text": "/status@mybot"}},
        {"update_id": 3, "message": {"chat": {"id": 42}, "text": "привет"}}]}]

    def api(self, method, **p):
        calls.append((method, p))
        if method == "getUpdates":
            if updates:
                return updates.pop(0)
            raise SystemExit   # остановить поток опроса
        return {"ok": True}

    monkeypatch.setattr(TelegramNotifier, "_api", api)
    n = TelegramNotifier("t", "42", "demo")
    try:
        n._poll_loop()
    except SystemExit:
        pass
    assert n.pop_commands() == ["/status"]
    assert n.offset == 4
    n.send("проверка", important=True)
    for _ in range(100):
        if any(c[0] == "sendMessage" for c in calls):
            break
        import time
        time.sleep(0.01)
    msg = [p for m, p in calls if m == "sendMessage"][0]
    assert msg["chat_id"] == "42" and msg["text"].startswith("❗ [ДЕМО] проверка")
    assert isinstance(n.out, queue.Queue)


@pytest.fixture
def env(tmp_path):
    e = make_env(tmp_path, start_day=120, notifier=Notes())
    install_hooks(e.engine)
    assert e.engine.start()
    return e


def test_status_and_stop_commands(env):
    n = env.engine.notifier
    env.run_until(T0 + 124 * DAY_MS)
    n.cmds = ["/status"]
    env.engine.step()
    assert any(m.startswith("Бот (sim)") and "Капитал бота" in m for m in n.sent)
    n.cmds = ["/stop"]
    env.engine.step()                 # команда → файл STOP
    r = env.engine.step()             # следующий шаг — аварийная остановка
    assert r.killed and env.db.get("halted")
    assert not env.sim.pos


def test_daily_alive_once_per_day_and_weekly_report(env):
    n = env.engine.notifier
    env.run_until(T0 + 131 * DAY_MS)     # дни 120…130 = 2026-05-01 (пт) … 2026-05-11 (пн)
    alive = [m for m in n.sent if m.startswith("Жив.")]
    assert len(alive) == 11              # по одному в сутки, после 06:00 UTC
    weekly = [m for m in n.sent if m.startswith("Недельный отчёт")]
    assert len(weekly) == 2              # понедельники 2026-05-04 и 2026-05-11
    w = weekly[0]
    assert "основная:" in w and "S1 боковой рынок" in w and "S2 только лонг" in w and "S3 оба фильтра" in w
    assert "Исполнение" in w
    assert len(list((env.tmp / "reports" / "live").glob("weekly_sim_*.txt"))) == 2


def test_status_text_shows_halt_and_positions(env):
    env.run_until(T0 + 126 * DAY_MS)
    txt = status_text(env.db, "demo", env.engine.equity()[0])
    assert "Капитал бота" in txt and "Последняя свеча 4h" in txt
    env.db.set("halted", {"reason": "тест", "ts": T0})
    assert "ОСТАНОВЛЕН: тест" in status_text(env.db, "demo", 25.0)


def test_single_instance_lock(tmp_path):
    a, b = InstanceLock(tmp_path / "x.lock"), InstanceLock(tmp_path / "x.lock")
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_env_writer_keeps_other_lines(tmp_path):
    from bot.setup_env import set_env_value
    p = tmp_path / ".env"
    p.write_text("# комментарий\nBYBIT_DEMO_API_KEY=\nOTHER=1\n", encoding="utf-8")
    set_env_value(p, "BYBIT_DEMO_API_KEY", "abc")
    set_env_value(p, "TELEGRAM_CHAT_ID", "42")
    assert p.read_text(encoding="utf-8").splitlines() == ["# комментарий", "BYBIT_DEMO_API_KEY=abc", "OTHER=1",
                                                          "TELEGRAM_CHAT_ID=42"]
