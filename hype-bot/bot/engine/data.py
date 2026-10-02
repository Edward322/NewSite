"""Живые данные: бот сам докачивает 15m-свечи, финансирование и параметры инструментов через API
в каталог data/live в том же формате, что данные бэктеста (bot/data/store.py), и строит из них
Panel тем же кодом (bot/data/panel.py::load_panel). Ручных загрузок больше нет.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bot.data import store
from bot.data.downloader import Caller, DownloadError, download_funding, download_klines
from bot.data.panel import BASE_MS, Panel, load_panel
from bot.exchange.client import BybitClient

log = logging.getLogger(__name__)


class LiveData:
    def __init__(self, client: BybitClient, data_dir: Path, tradable: list[str], signal_only: list[str],
                 sleep=None):
        self.client, self.dir = client, Path(data_dir)
        self.tradable, self.signal_only = list(tradable), list(signal_only)
        self.symbols = self.tradable + self.signal_only
        self.call = Caller(pause_s=0.0, max_attempts=4, **({"sleep": sleep} if sleep else {}))
        self.dir.mkdir(parents=True, exist_ok=True)
        self._last: dict[str, int | None] = {}
        self._last_f: dict[str, int | None] = {}

    # ----------------------------------------------------------- инструменты
    def refresh_instruments(self, now_ms: int) -> None:
        for s in self.symbols:
            inst = self.client.instrument(s)
            tiers = self.client.risk_limits(s)
            store.write_json(store.json_path(self.dir, s, "instrument"), {"fetched_at_ms": now_ms, "instrument": inst})
            store.write_json(store.json_path(self.dir, s, "risk_limit"), {"fetched_at_ms": now_ms, "tiers": tiers})

    # ----------------------------------------------------------------- свечи
    def _path(self, s: str) -> Path:
        return store.kline_path(self.dir, s, store.KIND_LAST, "15")

    def last_ts(self, s: str) -> int | None:
        if s not in self._last:
            p = self._path(s)
            df = pd.read_parquet(p, columns=["ts"]) if p.exists() else None
            self._last[s] = int(df["ts"].max()) if df is not None and len(df) else None
        return self._last[s]

    def _last_funding(self, s: str) -> int | None:
        if s not in self._last_f:
            fp = store.funding_path(self.dir, s)
            df = pd.read_parquet(fp, columns=["ts"]) if fp.exists() else None
            self._last_f[s] = int(df["ts"].max()) if df is not None and len(df) else None
        return self._last_f[s]

    def update(self, start_ms: int, end_ms: int) -> int:
        """Докачивает закрытые к end_ms свечи (с start_ms для новых монет). Возвращает число новых свечей."""
        added = 0
        try:
            for s in self.symbols:
                last = self.last_ts(s)
                # последняя сохранённая свеча скачивается ещё раз: если биржа отдала её недозакрытой,
                # запись исправится (merge_and_write оставляет более свежую версию)
                a = start_ms if last is None else last
                if a + BASE_MS <= end_ms:
                    df = download_klines(self.client.s, self.call, s, store.KIND_LAST, a, end_ms, interval="15",
                                         progress_every=0)
                    if len(df):
                        store.merge_and_write(self._path(s), df, store.KLINE_COLUMNS)
                        self._last[s] = max(int(df["ts"].max()), last or 0)
                        added += len(df)
                lf = self._last_funding(s)
                fa = lf + 1 if lf is not None else start_ms
                if fa <= end_ms:
                    new = download_funding(self.client.s, self.call, s, fa, end_ms)
                    fp = store.funding_path(self.dir, s)
                    if len(new):
                        store.merge_and_write(fp, new, store.FUNDING_COLUMNS)
                        self._last_f[s] = max(int(new["ts"].max()), lf or 0)
                    elif not fp.exists():
                        store.merge_and_write(fp, new, store.FUNDING_COLUMNS)
        except DownloadError as e:
            from bot.exchange.client import NetworkError
            raise NetworkError(f"загрузка свечей: {e}") from e
        return added

    def missing(self, t_close: int) -> list[str]:
        """Монеты, у которых уже есть история, но нет последней 15m-свечи перед t_close."""
        need = t_close - BASE_MS
        out = []
        for s in self.symbols:
            last = self.last_ts(s)
            if last is None or last < need:
                out.append(s)
        return out

    def panel(self, start_ms: int, end_ms: int) -> Panel:
        return load_panel(self.dir, self.tradable, self.signal_only, start_ms, end_ms)
