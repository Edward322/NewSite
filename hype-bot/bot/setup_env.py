"""Мастер первой настройки (запускается установщиком windows/install.bat).

    python -m bot.setup_env            всё по шагам: конфиг, ключи демо-счёта, Telegram
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


def ask_keys(mode: str = "demo") -> bool:
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
    key = input("API Key: ").strip()
    secret = getpass.getpass("API Secret (при вставке символы не видны — это нормально): ").strip()
    if not key or not secret:
        print("Ключи не введены.")
        return False
    set_env_value(ENV, f"{prefix}_API_KEY", key)
    set_env_value(ENV, f"{prefix}_API_SECRET", secret)
    print("Ключи записаны в .env.")
    return True


def main(argv: list[str]) -> int:
    ensure_config()
    if argv[:1] == ["keys"]:
        return 0 if ask_keys(argv[1] if len(argv) > 1 else "demo") else 1
    if not ask_keys("demo"):
        return 1
    from bot.engine.notify import setup as tg_setup
    tg_setup(ENV)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
