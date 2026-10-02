"""Уведомления в Telegram и команды /status, /stop.

Отправка идёт из фоновой очереди с повторами: сбой Telegram не задерживает торговлю.
Команды принимает фоновый поток (long polling getUpdates), но выполняет основной цикл движка
(handle_commands из bot/engine/reports.py): к бирже обращается только один поток.
Принимаются сообщения только из чата TELEGRAM_CHAT_ID (ваш личный чат с ботом).

Без TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID в .env уведомления пишутся только в журнал.

Настройка: python -m bot.engine.notify setup
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time

log = logging.getLogger(__name__)
API = "https://api.telegram.org/bot{token}/{method}"
MAX_LEN = 4000


class LogNotifier:
    def send(self, text: str, important: bool = False) -> None:
        log.info("[уведомление] %s", text)

    def start_commands(self, engine) -> None: ...

    def pop_commands(self) -> list[str]:
        return []


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, mode: str, timeout: float = 15.0):
        self.token, self.chat_id = token, str(chat_id)
        self.prefix = {"demo": "[ДЕМО] ", "live": "[РЕАЛ] "}.get(mode, "")
        self.timeout = timeout
        self.out: queue.Queue = queue.Queue(maxsize=500)
        self.cmds: queue.Queue = queue.Queue()
        self._sender = threading.Thread(target=self._send_loop, daemon=True, name="tg-send")
        self._sender.start()
        self._poller: threading.Thread | None = None
        self.offset = 0

    def _api(self, method: str, **params):
        import requests
        r = requests.post(API.format(token=self.token, method=method), json=params, timeout=self.timeout + 5)
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------------- отправка
    def send(self, text: str, important: bool = False) -> None:
        log.info("[telegram] %s", text)
        text = (("❗ " if important else "") + self.prefix + text)[:MAX_LEN]
        try:
            self.out.put_nowait(text)
        except queue.Full:
            log.warning("Очередь Telegram переполнена, сообщение пропущено")

    def _send_loop(self) -> None:
        while True:
            text = self.out.get()
            for attempt in range(6):
                try:
                    self._api("sendMessage", chat_id=self.chat_id, text=text, disable_web_page_preview=True)
                    break
                except Exception as e:
                    log.warning("Telegram: не отправлено (%s), попытка %d", str(e)[:120], attempt + 1)
                    time.sleep(min(60, 5 * 2 ** attempt))

    # ---------------------------------------------------------------- команды
    def start_commands(self, engine=None) -> None:
        if self._poller is None:
            self._poller = threading.Thread(target=self._poll_loop, daemon=True, name="tg-poll")
            self._poller.start()

    def _poll_loop(self) -> None:
        while True:
            try:
                res = self._api("getUpdates", offset=self.offset, timeout=int(self.timeout))
                for u in res.get("result", []):
                    self.offset = max(self.offset, u["update_id"] + 1)
                    m = u.get("message") or {}
                    if str((m.get("chat") or {}).get("id")) != self.chat_id:
                        continue
                    txt = (m.get("text") or "").strip().split("@")[0].lower()
                    if txt.startswith("/"):
                        self.cmds.put(txt)
            except Exception as e:
                log.warning("Telegram getUpdates: %s", str(e)[:120])
                time.sleep(30)

    def pop_commands(self) -> list[str]:
        out = []
        while not self.cmds.empty():
            out.append(self.cmds.get_nowait())
        return out


def make_notifier(db=None, mode: str = "demo"):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if token and chat:
        return TelegramNotifier(token, chat, mode)
    return LogNotifier()


# ------------------------------------------------------------------ настройка
def setup(env_path=None) -> int:
    """Спрашивает токен бота, ждёт ваше сообщение боту, записывает TELEGRAM_* в .env."""
    import requests

    from bot.config import PROJECT_ROOT
    from bot.envfile import load_env
    from bot.setup_env import set_env_value
    env_path = env_path or PROJECT_ROOT / ".env"
    load_env(env_path)
    if os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID"):
        if not input("Telegram уже настроен. Настроить заново? (да/нет): ").strip().lower().startswith("д"):
            return 0
    print("\nНастройка Telegram (можно пропустить — нажмите Enter).")
    print("1) В Telegram найдите @BotFather, отправьте ему /newbot, придумайте имя бота.")
    print("2) BotFather пришлёт токен вида 123456789:AA...  Скопируйте его сюда.")
    token = input("Токен бота: ").strip() or os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        print("Telegram пропущен: уведомления будут только в журнале logs\\.")
        return 0
    try:
        me = requests.get(API.format(token=token, method="getMe"), timeout=20).json()
    except Exception as e:
        print(f"Нет связи с Telegram: {e}")
        return 1
    if not me.get("ok"):
        print("Токен не подошёл. Проверьте и запустите настройку ещё раз.")
        return 1
    name = me["result"].get("username")
    print(f"\n3) Откройте в Telegram бота @{name} и отправьте ему любое сообщение (например «привет»).")
    print("   Жду сообщение до 3 минут...")
    chat_id, offset, t0 = None, 0, time.time()
    while time.time() - t0 < 180 and chat_id is None:
        res = requests.get(API.format(token=token, method="getUpdates"), params={"offset": offset, "timeout": 20},
                           timeout=30).json()
        for u in res.get("result", []):
            offset = u["update_id"] + 1
            chat = ((u.get("message") or {}).get("chat") or {})
            if chat.get("type") == "private":
                chat_id = chat.get("id")
    if chat_id is None:
        print("Сообщение не пришло. Запустите настройку ещё раз.")
        return 1
    requests.get(API.format(token=token, method="getUpdates"), params={"offset": offset}, timeout=20)
    set_env_value(env_path, "TELEGRAM_BOT_TOKEN", token)
    set_env_value(env_path, "TELEGRAM_CHAT_ID", str(chat_id))
    requests.post(API.format(token=token, method="sendMessage"),
                  json={"chat_id": chat_id, "text": "Бот подключён. Команды: /status — состояние, /stop — аварийная "
                                                   "остановка (закрыть позиции и выключить бота)."}, timeout=20)
    print("Готово: Telegram подключён, проверочное сообщение отправлено.")
    return 0


if __name__ == "__main__":
    import sys
    if sys.argv[1:] == ["setup"]:
        sys.exit(setup())
    print("python -m bot.engine.notify setup")
