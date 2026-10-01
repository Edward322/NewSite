"""Загрузка исторических данных Bybit V5 через официальный SDK pybit.

Только публичные эндпоинты (ключи не нужны), кроме комиссий (fetch_fee_rate).

Правила, проверенные по документации Bybit V5:
- /v5/market/kline и /v5/market/mark-price-kline: limit ≤ 1000, список
  отсортирован по убыванию startTime, start/end — по времени открытия свечи.
- /v5/market/funding/history: limit ≤ 200; передавать только startTime нельзя,
  поэтому идём назад по endTime.
- Лимит IP: 600 запросов / 5 с; 403 означает лимит IP или запрещённую страну.

Незакрытая (текущая) свеча никогда не сохраняется.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from bot.data import store

log = logging.getLogger(__name__)

MINUTE_MS = 60_000
DAY_MS = 86_400_000
KLINE_LIMIT = 1000
FUNDING_LIMIT = 200
CHUNK_REQUESTS = 100  # запросов на одну порцию сохранения (~69 дней минутных свечей)


class DownloadError(RuntimeError):
    pass


def interval_ms(interval: str) -> int:
    if interval.isdigit():
        return int(interval) * MINUTE_MS
    if interval == "D":
        return DAY_MS
    raise ValueError(f"Неподдерживаемый интервал для загрузки: {interval}")


def fmt_ts(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


@dataclass
class Caller:
    """Вызов метода pybit с повтором сетевых ошибок (публичные GET идемпотентны)."""

    pause_s: float = 0.12
    max_attempts: int = 5
    sleep: Callable[[float], None] = time.sleep
    requests_made: int = field(default=0, init=False)

    def __call__(self, fn: Callable[..., dict], **params: Any) -> dict:
        from pybit.exceptions import FailedRequestError, InvalidRequestError
        import requests

        delay = 1.0
        for attempt in range(1, self.max_attempts + 1):
            try:
                resp = fn(**params)
                self.requests_made += 1
                if self.pause_s:
                    self.sleep(self.pause_s)
                if resp.get("retCode", 0) != 0:
                    raise DownloadError(f"retCode={resp.get('retCode')} {resp.get('retMsg')}")
                return resp
            except FailedRequestError as e:
                if getattr(e, "status_code", None) == 403:
                    raise DownloadError(
                        "Bybit ответил 403: превышен лимит запросов с IP (бан ~10 минут) "
                        "или IP из страны, для которой API закрыт (США, материковый Китай)."
                    ) from e
                err: Exception = e
            except InvalidRequestError as e:
                # Ошибка параметров/бизнес-логики — повтор не поможет.
                raise DownloadError(f"Bybit отклонил запрос: {e}") from e
            except requests.exceptions.ProxyError as e:
                # Соединение запрещает прокси/файрвол на нашей стороне — до Bybit запрос не дошёл.
                raise DownloadError(
                    "Нет доступа к API Bybit: соединение блокирует сеть (прокси или файрвол), "
                    f"запрос до биржи не дошёл. Подробности: {e}"
                ) from e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                err = e
            if attempt == self.max_attempts:
                raise DownloadError(f"Сеть недоступна после {attempt} попыток: {err}") from err
            log.warning("Ошибка сети (%s), повтор через %.0f с", err, delay)
            self.sleep(delay)
            delay = min(delay * 2, 30)
        raise AssertionError("unreachable")


def server_time_ms(session, call: Caller) -> int:
    resp = call(session.get_server_time)
    return int(resp["time"])


def fetch_instrument(session, call: Caller, symbol: str, category: str = "linear") -> dict:
    resp = call(session.get_instruments_info, category=category, symbol=symbol)
    lst = resp["result"].get("list") or []
    if not lst:
        raise DownloadError(f"Инструмент {symbol} не найден в category={category}")
    return lst[0]


def fetch_risk_limits(session, call: Caller, symbol: str, category: str = "linear") -> list[dict]:
    tiers: list[dict] = []
    cursor = None
    while True:
        params = {"category": category, "symbol": symbol}
        if cursor:
            params["cursor"] = cursor
        resp = call(session.get_risk_limit, **params)
        tiers.extend(resp["result"].get("list") or [])
        cursor = resp["result"].get("nextPageCursor")
        if not cursor:
            return tiers


def fetch_fee_rate(auth_session, call: Caller, symbol: str) -> list[dict]:
    """Требует ключ API (только чтение достаточно). Возвращает ставки maker/taker."""
    resp = call(auth_session.get_fee_rates, category="linear", symbol=symbol)
    return resp["result"].get("list") or []


def download_klines(
    session,
    call: Caller,
    symbol: str,
    kind: str,
    start_ms: int,
    end_ms: int,
    interval: str = "1",
    category: str = "linear",
    progress_every: int = 100,
) -> pd.DataFrame:
    """Закрытые свечи с открытием в [start_ms, end_ms - interval].

    Идём вперёд фиксированными окнами ровно по KLINE_LIMIT свечей, поэтому
    результат не зависит от порядка сортировки ответа и не теряет свечи.
    """
    step = interval_ms(interval)
    window = step * KLINE_LIMIT
    fn = session.get_kline if kind == store.KIND_LAST else session.get_mark_price_kline
    columns = store.KLINE_COLUMNS if kind == store.KIND_LAST else store.MARK_COLUMNS
    ncol = len(columns)

    ws = (start_ms // step) * step
    last_open = ((end_ms // step) * step) - step  # последняя полностью закрытая свеча
    rows: list[list] = []
    n = 0
    while ws <= last_open:
        we = min(ws + window - 1, last_open)
        resp = call(fn, category=category, symbol=symbol, interval=interval,
                    start=ws, end=we, limit=KLINE_LIMIT)
        for item in resp["result"].get("list") or []:
            ts = int(item[0])
            if ws <= ts <= we:
                rows.append([ts] + [float(x) for x in item[1:ncol]])
        ws += window
        n += 1
        if progress_every and n % progress_every == 0:
            log.info("%s %s: загружено до %s (%d свечей)", symbol, kind, fmt_ts(min(ws, last_open)), len(rows))

    df = pd.DataFrame(rows, columns=columns)
    if len(df):
        df["ts"] = df["ts"].astype("int64")
        df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)
    return df


def download_funding(
    session, call: Caller, symbol: str, start_ms: int, end_ms: int, category: str = "linear"
) -> pd.DataFrame:
    rows: list[tuple[int, float]] = []
    cur_end = end_ms
    while cur_end >= start_ms:
        resp = call(session.get_funding_rate_history, category=category, symbol=symbol,
                    startTime=start_ms, endTime=cur_end, limit=FUNDING_LIMIT)
        lst = resp["result"].get("list") or []
        page = [(int(x["fundingRateTimestamp"]), float(x["fundingRate"])) for x in lst]
        page = [p for p in page if start_ms <= p[0] <= cur_end]
        if not page:
            break
        rows.extend(page)
        oldest = min(p[0] for p in page)
        if len(lst) < FUNDING_LIMIT:
            break
        cur_end = oldest - 1
    df = pd.DataFrame(rows, columns=store.FUNDING_COLUMNS)
    if len(df):
        df["ts"] = df["ts"].astype("int64")
        df = df.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return df


def _resume_from(path: Path, columns: list[str], default_start: int, step: int) -> int:
    existing = store.read_parquet(path, columns)
    if len(existing):
        return int(existing["ts"].iloc[-1]) + step
    return default_start


def download_symbol(
    session,
    call: Caller,
    data_dir: Path,
    symbol: str,
    history_days: int | None = None,
    kinds: tuple[str, ...] = (store.KIND_LAST, store.KIND_MARK),
    funding: bool = True,
    now_ms: int | None = None,
) -> dict:
    """Загружает (или дозагружает) всё по одному символу. Возвращает сводку."""
    now_ms = now_ms if now_ms is not None else server_time_ms(session, call)
    inst = fetch_instrument(session, call, symbol)
    inst_payload = {"fetched_at_ms": now_ms, "instrument": inst}
    store.write_json(store.json_path(data_dir, symbol, "instrument"), inst_payload)
    tiers = fetch_risk_limits(session, call, symbol)
    store.write_json(store.json_path(data_dir, symbol, "risk_limit"),
                     {"fetched_at_ms": now_ms, "tiers": tiers})

    launch_ms = int(inst.get("launchTime") or 0)
    start_ms = launch_ms
    if history_days is not None:
        start_ms = max(launch_ms, now_ms - history_days * DAY_MS)

    summary: dict = {"symbol": symbol, "launch_ms": launch_ms, "start_ms": start_ms, "now_ms": now_ms}
    chunk_ms = CHUNK_REQUESTS * KLINE_LIMIT * MINUTE_MS
    for kind in kinds:
        path = store.kline_path(data_dir, symbol, kind)
        columns = store.KLINE_COLUMNS if kind == store.KIND_LAST else store.MARK_COLUMNS
        resume = _resume_from(path, columns, start_ms, MINUTE_MS)
        log.info("%s %s: загрузка с %s", symbol, kind, fmt_ts(resume))
        added = 0
        df = store.read_parquet(path, columns)
        # Сохраняем частями: после обрыва связи повторный запуск продолжит с места остановки.
        chunk_start = (resume // MINUTE_MS) * MINUTE_MS
        while chunk_start + MINUTE_MS <= now_ms:
            chunk_end = min(chunk_start + chunk_ms, now_ms)
            new = download_klines(session, call, symbol, kind, chunk_start, chunk_end)
            df = store.merge_and_write(path, new, columns)
            added += len(new)
            log.info("%s %s: сохранено до %s, всего %d свечей", symbol, kind,
                     fmt_ts(min(chunk_end, now_ms)), len(df))
            chunk_start = chunk_end
        summary[f"kline_{kind}_rows"] = len(df)
        summary[f"kline_{kind}_new"] = added
    if funding:
        path = store.funding_path(data_dir, symbol)
        resume = _resume_from(path, store.FUNDING_COLUMNS, start_ms, 1)
        new = download_funding(session, call, symbol, resume, now_ms)
        df = store.merge_and_write(path, new, store.FUNDING_COLUMNS)
        summary["funding_rows"] = len(df)
    return summary


def write_manifest(data_dir: Path, manifest_path: Path, symbols: list[str]) -> dict:
    """Сводка файлов (строки, диапазон, sha256) — коммитится в git вместо самих данных."""
    entries = {}
    for sym in symbols:
        sdir = Path(data_dir) / sym
        if not sdir.exists():
            continue
        for f in sorted(sdir.iterdir()):
            if f.suffix not in (".parquet", ".json"):
                continue
            info: dict = {"sha256": store.file_sha256(f), "bytes": f.stat().st_size}
            if f.suffix == ".parquet":
                df = pd.read_parquet(f, columns=["ts"])
                info.update(rows=len(df),
                            first=fmt_ts(int(df["ts"].iloc[0])) if len(df) else None,
                            last=fmt_ts(int(df["ts"].iloc[-1])) if len(df) else None)
            entries[f"{sym}/{f.name}"] = info
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "files": entries}
    store.write_json(manifest_path, payload)
    return payload


def make_public_session(domain: str = "bybit", tld: str = "com", timeout: float = 10):
    from pybit.unified_trading import HTTP

    # max_retries=1: повторы делает Caller, чтобы контролировать паузы и 403.
    return HTTP(testnet=False, domain=domain, tld=tld, timeout=timeout, max_retries=1)
