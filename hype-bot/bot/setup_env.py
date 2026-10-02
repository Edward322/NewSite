"""Мастер первой настройки (запускается установщиком windows/install.bat).

    python -m bot.setup_env [demo|live]  всё по шагам: конфиг, ключи счёта, Telegram
    python -m bot.setup_env keys demo  только ключи демо-счёта (или live)

Ключи пишутся только в файл .env в папке бота (он не попадает в git и никуда не отправляется).
"""
from __future__ import annotations

import getpass
import os
import shutil
import sys
from pathlib import Path

from bot.config import BOT_CONFIG, BOT_EXAMPLE_CONFIG, PROJECT_ROOT

ENV = PROJECT_ROOT / ".env"


def set_env_value(path: Path, key: str, value: str) -> None:
    path = Path(path)
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if line.split("=", 1)[0].strip() == key:
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.environ[key] = value


def ensure_config() -> None:
    if not BOT_CONFIG.exists():
        shutil.copyfile(BOT_EXAMPLE_CONFIG, BOT_CONFIG)
        print(f"Создан {BOT_CONFIG.relative_to(PROJECT_ROOT)} (параметры из отчёта этапа 1).")
    if not ENV.exists():
        shutil.copyfile(PROJECT_ROOT / ".env.example", ENV)


def check_format(key: str, secret: str) -> list[str]:
    """Явные ошибки вставки: пусто, слишком коротко, посторонние символы (например, Ctrl+V в скрытом поле)."""
    out = []
    for name, v, n_min in (("ключ", key, 10), ("секрет", secret, 20)):
        if len(v) < n_min:
            out.append(f"{name}: получено {len(v)} символов — слишком мало, похоже, вставка не сработала")
        elif not (v.isascii() and v.isalnum()):
            out.append(f"{name}: есть посторонние символы (пробелы, кавычки или служебные) — вставьте заново")
    return out


def verify(mode: str, key: str, secret: str) -> tuple[bool | None, str]:
    """Проверка ключа на бирже: True — годится, False — нет (с причиной), None — нет связи, проверить нельзя."""
    from bot.config import load_bot_config
    from bot.engine.control import make_session
    from bot.exchange.client import BybitClient, ExchangeError, NetworkError, explain
    try:
        client = BybitClient(make_session(load_bot_config(), mode, key, secret), get_attempts=2)
        info = client._get("get_api_key_information")
    except ExchangeError as e:
        return False, f"Биржа не приняла ключ (код {e.code}): {explain(e.code) or e.message}"
    except NetworkError as e:
        return None, f"Не удалось проверить ключ — нет связи с биржей ({e}). Ключи сохранены, их проверит самопроверка."
    perms = [x for v in (info.get("permissions") or {}).values() for x in (v or [])]
    if "Withdraw" in perms:
        return False, "У ключа есть право ВЫВОДА средств. Создайте ключ без него."
    if info.get("readOnly"):
        return False, "Ключ только для чтения. Нужны права «Контракты: Ордера, Позиции» (чтение и запись)."
    ips = info.get("ips") or []
    if mode == "live" and (not ips or ips == ["*"]):
        return False, "Ключ реального счёта должен быть привязан к вашему IP."
    return True, "Ключ проверен на бирже: подходит."


def ask_keys(mode: str = "demo", check=verify, attempts: int = 3) -> bool:
    prefix = {"demo": "BYBIT_DEMO", "live": "BYBIT_LIVE"}[mode]
    from bot.envfile import load_env
    load_env(ENV)
    if os.environ.get(f"{prefix}_API_KEY") and os.environ.get(f"{prefix}_API_SECRET"):
        ans = input(f"Ключи {'демо' if mode == 'demo' else 'реального'} счёта уже записаны. Заменить? (да/нет): ")
        if not ans.strip().lower().startswith("д"):
            return True
    if mode == "demo":
        print("\nКлючи ДЕМО-счёта: bybit.com → войти → переключатель «Демо-торговля» (Demo Trading) →")
        print("профиль → API → «Создать новый ключ» → «Системный» → права «Контракты: Ордера, Позиции»")
        print("(чтение и запись), «Единый торговый аккаунт»: Торговля. Вывод средств НЕ включать.")
    else:
        print("\nКлючи РЕАЛЬНОГО счёта — см. README, раздел «Реальные деньги».")
    print("Вставляйте ПРАВОЙ кнопкой мыши (Ctrl+V в этом окне может вставить не то).")
    for attempt in range(1, attempts + 1):
        key = input("API Key: ").strip()
        secret = getpass.getpass("API Secret (символы не видны — это нормально): ").strip()
        print(f"Получено: ключ — {len(key)} символов, секрет — {len(secret)} символов "
              "(обычно у Bybit 18 и 36).")
        problems = check_format(key, secret)
        if not problems:
            ok, msg = check(mode, key, secret)
            print(msg)
            if ok is not False:
                set_env_value(ENV, f"{prefix}_API_KEY", key)
                set_env_value(ENV, f"{prefix}_API_SECRET", secret)
                print("Ключи записаны в .env.")
                return True
        else:
            print("\n".join(problems))
        if attempt < attempts:
            print(f"Попробуйте ещё раз ({attempt + 1} из {attempts}).")
    print("Ключи не записаны.")
    return False


def main(argv: list[str]) -> int:
    ensure_config()
    if argv[:1] == ["keys"]:
        return 0 if ask_keys(argv[1] if len(argv) > 1 else "demo") else 1
    mode = argv[0] if argv[:1] and argv[0] in ("demo", "live") else "demo"
    if mode == "live":
        from bot.config import load_bot_config
        if not load_bot_config().live.enabled:
            print("Реальная торговля выключена (live.enabled: false в config/bot.yaml). Её включают только после "
                  "вашего явного решения — см. README, раздел «Реальные деньги».")
            return 1
    if not ask_keys(mode):
        return 1
    from bot.engine.notify import setup as tg_setup
    tg_setup(ENV)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
