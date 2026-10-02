"""WebSocket Bybit V5 поверх pybit: свежесть рыночных данных и события счёта.

  - публичный поток kline.15 по монетам корзины (на демо — тоже основной сервер биржи:
    у демо-торговли рыночные данные общие) — по нему движок понимает, что данные живые;
  - приватный поток position / execution / order — любое событие запускает внеочередную сверку.

pybit сам переподключается при ошибках (restart_on_error) и заново подписывается на темы.
Если сообщений нет дольше порога, движок вызывает restart(): соединения создаются заново.
Колбэки pybit работают в своих потоках, поэтому здесь только отметки времени под замком.
"""
from __future__ import annotations

import logging
import threading
import time

log = logging.getLogger(__name__)


class BybitStream:
    def __init__(self, symbols: list[str], mode: str, api_key: str | None, api_secret: str | None,
                 domain: str = "bybit", tld: str = "com", min_restart_s: float = 60.0):
        self.symbols, self.mode = list(symbols), mode
        self.key, self.secret = api_key, api_secret
        self.domain, self.tld = domain, tld
        self.min_restart_s = min_restart_s
        self._lock = threading.Lock()
        self._last_public: float | None = None
        self._started: float | None = None
        self._last_restart = 0.0
        self._dirty = False
        self.pub = self.priv = None
        self.errors = 0

    # ------------------------------------------------------------ колбэки
    def _on_public(self, msg) -> None:
        with self._lock:
            self._last_public = time.time()

    def _on_private(self, msg) -> None:
        with self._lock:
            self._dirty = True

    # ------------------------------------------------------------- работа
    def start(self) -> None:
        from pybit.unified_trading import WebSocket
        self._started = time.time()
        self._last_restart = self._started
        try:
            self.pub = WebSocket(testnet=False, channel_type="linear", domain=self.domain, tld=self.tld,
                                 retries=3, restart_on_error=True, ping_interval=20, ping_timeout=10)
            for i in range(0, len(self.symbols), 10):          # не больше 10 тем в одном запросе подписки
                self.pub.kline_stream(interval=15, symbol=self.symbols[i:i + 10], callback=self._on_public)
        except Exception as e:      # сеть, DNS — движок увидит «нет данных» и перезапустит позже
            self.errors += 1
            log.warning("WebSocket публичный: %s", e)
        if self.key and self.secret:
            try:
                self.priv = WebSocket(testnet=False, channel_type="private", demo=self.mode == "demo",
                                      api_key=self.key, api_secret=self.secret, domain=self.domain, tld=self.tld,
                                      retries=3, restart_on_error=True, ping_interval=20, ping_timeout=10)
                self.priv.position_stream(callback=self._on_private)
                self.priv.execution_stream(callback=self._on_private)
                self.priv.order_stream(callback=self._on_private)
            except Exception as e:
                self.errors += 1
                log.warning("WebSocket приватный: %s", e)

    def public_age_s(self, now_ms: int | None = None) -> float | None:
        """Секунд с последнего сообщения (или с попытки подключения, если сообщений ещё не было)."""
        with self._lock:
            ref = self._last_public or self._started
        return None if ref is None else max(0.0, time.time() - ref)

    def pop_dirty(self) -> bool:
        with self._lock:
            d, self._dirty = self._dirty, False
        return d

    def restart(self) -> None:
        if time.time() - self._last_restart < self.min_restart_s:
            return
        log.warning("WebSocket: переподключение")
        self.stop()
        self.start()

    def stop(self) -> None:
        for ws in (self.pub, self.priv):
            try:
                if ws is not None:
                    ws.exit()
            except Exception:
                pass
        self.pub = self.priv = None
