"""Мастер ключей: явные ошибки вставки ловятся сразу, ключ проверяется на бирже до записи в .env."""
import bot.setup_env as se


def test_format_check_catches_failed_paste():
    assert se.check_format("tLs1YQ8B2OGOj5nbFx", "A" * 36) == []
    assert any("мало" in p for p in se.check_format("tLs1YQ8B2OGOj5nbFx", "\x16"))
    assert any("посторонние" in p for p in se.check_format("tLs1YQ8B2OGOj5nbFx", '"' + "A" * 36 + '"'))


def test_wrong_secret_is_not_saved_and_retry_works(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(se, "ENV", env)
    monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
    keys = iter(["tLs1YQ8B2OGOj5nbFx", "tLs1YQ8B2OGOj5nbFx"])
    secrets = iter(["B" * 36, "C" * 36])
    monkeypatch.setattr("builtins.input", lambda *_: next(keys))
    monkeypatch.setattr(se.getpass, "getpass", lambda *_: next(secrets))
    checked = []

    def check(mode, key, secret):
        checked.append(secret)
        return (secret == "C" * 36), ("ok" if secret == "C" * 36 else "код 10004")

    assert se.ask_keys("demo", check=check)
    assert checked == ["B" * 36, "C" * 36]
    text = env.read_text(encoding="utf-8")
    assert "BYBIT_DEMO_API_SECRET=" + "C" * 36 in text and "B" * 36 not in text


def test_gives_up_after_three_bad_attempts(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    monkeypatch.setattr(se, "ENV", env)
    monkeypatch.delenv("BYBIT_DEMO_API_KEY", raising=False)
    monkeypatch.delenv("BYBIT_DEMO_API_SECRET", raising=False)
    monkeypatch.setattr("builtins.input", lambda *_: "tLs1YQ8B2OGOj5nbFx")
    monkeypatch.setattr(se.getpass, "getpass", lambda *_: "\x16")
    assert not se.ask_keys("demo", check=lambda *a: (True, "ok"))
    assert not env.exists() or "BYBIT_DEMO_API_SECRET" not in env.read_text(encoding="utf-8")
