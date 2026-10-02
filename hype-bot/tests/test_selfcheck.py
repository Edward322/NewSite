"""Скрипт самопроверки не падает и проходит на имитаторе биржи (настоящую биржу проверяет пользователь)."""
import bot.selfcheck as sc
from bot.data.panel import DAY_MS
from tests.engine_env import make_env


class FakeStream:
    def __init__(self, *a, **k):
        self._last_public = None
        self.priv = None

    def start(self):
        import time
        self._last_public = time.time()
        self.priv = object()

    def public_age_s(self, now_ms=None):
        return 0.5

    def stop(self): ...


def test_selfcheck_full_on_simulator(tmp_path, monkeypatch, capsys):
    env = make_env(tmp_path, start_day=130)
    sim = env.sim
    monkeypatch.setenv("BYBIT_DEMO_API_KEY", "abcd1234")
    monkeypatch.setenv("BYBIT_DEMO_API_SECRET", "secret")
    monkeypatch.setattr(sc, "load_bot_config", lambda: env.cfg)
    monkeypatch.setattr(sc, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sc, "TEST_SYMBOL", "C0USDT")
    monkeypatch.setattr("bot.engine.control.make_session", lambda *a, **k: sim)
    monkeypatch.setattr("bot.engine.stream.BybitStream", FakeStream)
    monkeypatch.setattr(sc.time, "sleep", lambda s: env.clock.sleep(s))
    monkeypatch.setattr(sc.time, "time", lambda: env.clock.now_ms() / 1000)
    import bot.engine.control as ctl
    monkeypatch.setattr(ctl.time, "time", lambda: env.clock.now_ms() / 1000)
    rc = sc.main(["--mode", "demo"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "secret" not in out
    assert "отклонён: код 110072" in out and "код 110017" in out
    assert "ЗАКРЫТА" in out and "Сухой прогон" in out
    assert not sim.pos                                   # тестовая позиция закрыта
    files = list((tmp_path / "reports" / "selfcheck").glob("selfcheck_demo_*.txt"))
    assert files and "ИТОГ" in files[0].read_text(encoding="utf-8")
    _ = DAY_MS
