"""Данные для корзины монет: отбор монет по правилу и загрузка всех рядов.

Отбор (правило фиксируется в конфиге, результат — в universe.json с причинами):
  - бессрочный USDT-контракт на криптовалюту (не акции/металлы/форекс, не стейблкоин);
  - в торговле с даты listed_before или раньше (есть история);
  - оборот за 24 ч ≥ min_turnover_24h;
  - минимальный ордер (max(minNotional, minQty × цена)) ≤ max_min_order_usdt —
    иначе депозит 10 USDT не потянет позицию с риском 5 %;
  - не больше n_max монет по обороту; монеты из include добавляются всегда.
Внимание: отбор по ТЕКУЩЕМУ обороту даёт смещение выживших (делистинги в выборку
не попадают). Это ограничение исследования, оно указывается в отчёте.

Ряды на монету: свечи последней цены (15m), премиальный индекс (базис, 1h),
открытый интерес (1h), соотношение лонгов/шортов по счетам (1h), финансирование,
параметры инструмента и уровни риска.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bot.data import store
from bot.data.downloader import (DAY_MS, MINUTE_MS, Caller, DownloadError, download_funding, download_klines,
                                 fetch_instrument, fetch_risk_limits, fmt_ts)

log = logging.getLogger(__name__)
HOUR_MS = 3_600_000
STABLES = {"USDC", "USDE", "FDUSD", "DAI", "TUSD", "USDD", "PYUSD", "USD1", "RLUSD"}
OI_LIMIT, LSR_LIMIT = 200, 500

PREMIUM_COLUMNS = ["ts", "open", "high", "low", "close"]
OI_COLUMNS = ["ts", "oi"]
LSR_COLUMNS = ["ts", "buy_ratio"]


def series_path(data_dir: Path, symbol: str, name: str) -> Path:
    return store.symbol_dir(data_dir, symbol) / f"{name}.parquet"


# ------------------------------------------------------------------ universe
def all_linear_instruments(session, call: Caller) -> list[dict]:
    out, cursor = [], None
    while True:
        params = {"category": "linear", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        resp = call(session.get_instruments_info, **params)
        out += resp["result"].get("list") or []
        cursor = resp["result"].get("nextPageCursor")
        if not cursor:
            return out


def select_universe(instruments: list[dict], tickers: list[dict], rule: dict) -> tuple[list[str], pd.DataFrame]:
    tk = {t["symbol"]: t for t in tickers}
    listed_before = int(pd.Timestamp(rule["listed_before"], tz="UTC").value // 1_000_000)
    rows = []
    for i in instruments:
        sym = i["symbol"]
        t = tk.get(sym, {})
        price = float(t.get("lastPrice") or 0)
        lot = i.get("lotSizeFilter", {})
        min_order = max(float(lot.get("minNotionalValue") or 0), float(lot.get("minOrderQty") or 0) * price)
        reasons = []
        if i.get("contractType") != "LinearPerpetual" or i.get("quoteCoin") != "USDT" or i.get("settleCoin") != "USDT":
            reasons.append("не USDT-бессрочный")
        if i.get("symbolType") or i.get("underlyingTicker") or i.get("marketRegion"):
            reasons.append("не криптовалюта")
        if i.get("baseCoin") in STABLES:
            reasons.append("стейблкоин")
        if i.get("status") != "Trading" or i.get("isPreListing"):
            reasons.append("не торгуется")
        if int(i.get("launchTime") or 0) > listed_before:
            reasons.append("листинг позже порога")
        if float(t.get("turnover24h") or 0) < rule["min_turnover_24h"]:
            reasons.append("малый оборот")
        if price <= 0 or min_order > rule["max_min_order_usdt"]:
            reasons.append(f"минимальный ордер {min_order:.2f} USDT")
        rows.append({"symbol": sym, "turnover24h": float(t.get("turnover24h") or 0), "last_price": price,
                     "min_order_usdt": min_order, "launch": fmt_ts(int(i.get("launchTime") or 0)),
                     "ok": not reasons, "reasons": "; ".join(reasons)})
    df = pd.DataFrame(rows).sort_values("turnover24h", ascending=False).reset_index(drop=True)
    chosen = [s for s in df[df["ok"]]["symbol"]][: rule["n_max"]]
    for s in rule.get("include", []):
        if s not in chosen:
            chosen.append(s)
    df["selected"] = df["symbol"].isin(chosen)
    return chosen, df


# ------------------------------------------------------------ hourly series
def download_windows(fn, call: Caller, symbol: str, start_ms: int, end_ms: int, step_ms: int, limit: int,
                     time_keys: tuple[str, str], extra: dict, parse) -> pd.DataFrame:
    """Окнами по (limit − 1) точек: ответ гарантированно помещается в одну страницу."""
    rows = []
    span = (limit - 1) * step_ms
    ws = start_ms // step_ms * step_ms
    while ws < end_ms:
        we = min(ws + span, end_ms)
        resp = call(fn, category="linear", symbol=symbol, **{time_keys[0]: ws, time_keys[1]: we}, limit=limit, **extra)
        for item in resp["result"].get("list") or []:
            row = parse(item)
            if ws <= row[0] <= we:
                rows.append(row)
        ws = we + 1
    return pd.DataFrame(rows)


def download_symbol_basket(session, call: Caller, data_dir: Path, symbol: str, start_ms: int, now_ms: int,
                           kline_interval: str = "15") -> dict:
    inst = fetch_instrument(session, call, symbol)
    store.write_json(store.json_path(data_dir, symbol, "instrument"), {"fetched_at_ms": now_ms, "instrument": inst})
    store.write_json(store.json_path(data_dir, symbol, "risk_limit"),
                     {"fetched_at_ms": now_ms, "tiers": fetch_risk_limits(session, call, symbol)})
    start_ms = max(start_ms, int(inst.get("launchTime") or 0))
    summary = {"symbol": symbol}
    step = int(kline_interval) * MINUTE_MS

    def resume(path, columns, default, inc):
        old = store.read_parquet(path, columns)
        return (int(old["ts"].iloc[-1]) + inc) if len(old) else default

    # свечи последней цены — порциями, чтобы обрыв не терял прогресс
    path = store.kline_path(data_dir, symbol, store.KIND_LAST, kline_interval)
    cur = resume(path, store.KLINE_COLUMNS, start_ms, step)
    chunk = 100 * 1000 * step
    while cur + step <= now_ms:
        end = min(cur + chunk, now_ms)
        df = download_klines(session, call, symbol, store.KIND_LAST, cur, end, interval=kline_interval,
                             progress_every=0)
        store.merge_and_write(path, df, store.KLINE_COLUMNS)
        log.info("%s свечи %sm: до %s", symbol, kline_interval, fmt_ts(end))
        cur = end
    summary["klines"] = len(store.read_parquet(path, store.KLINE_COLUMNS))

    # финансирование
    fpath = store.funding_path(data_dir, symbol)
    new = download_funding(session, call, symbol, resume(fpath, store.FUNDING_COLUMNS, start_ms, 1), now_ms)
    summary["funding"] = len(store.merge_and_write(fpath, new, store.FUNDING_COLUMNS))

    # премиальный индекс (базис) 1h
    ppath = series_path(data_dir, symbol, "premium_1h")
    cur = resume(ppath, PREMIUM_COLUMNS, start_ms, HOUR_MS)
    if cur + HOUR_MS <= now_ms:
        new = _premium(session, call, symbol, cur, now_ms)
        store.merge_and_write(ppath, new, PREMIUM_COLUMNS)
    summary["premium"] = len(store.read_parquet(ppath, PREMIUM_COLUMNS))

    # открытый интерес и соотношение лонгов/шортов, 1h — необязательные: ошибка не останавливает загрузку
    for name, columns, fn, limit, keys, extra, parse in (
        ("oi_1h", OI_COLUMNS, session.get_open_interest, OI_LIMIT, ("startTime", "endTime"),
         {"intervalTime": "1h"}, lambda x: (int(x["timestamp"]), float(x["openInterest"]))),
        ("lsr_1h", LSR_COLUMNS, session.get_long_short_ratio, LSR_LIMIT, ("startTime", "endTime"),
         {"period": "1h"}, lambda x: (int(x["timestamp"]), float(x["buyRatio"]))),
    ):
        spath = series_path(data_dir, symbol, name)
        cur = resume(spath, columns, start_ms, HOUR_MS)
        try:
            if cur < now_ms:
                new = download_windows(fn, call, symbol, cur, now_ms, HOUR_MS, limit, keys, extra, parse)
                if len(new):
                    new.columns = columns
                store.merge_and_write(spath, new if len(new) else pd.DataFrame(columns=columns), columns)
            summary[name] = len(store.read_parquet(spath, columns))
        except DownloadError as e:
            log.warning("%s %s: пропущено (%s)", symbol, name, e)
            summary[name] = f"ошибка: {e}"
    return summary


def _premium(session, call, symbol, start_ms, end_ms) -> pd.DataFrame:
    from types import SimpleNamespace
    # тот же формат ответа, что у свечей mark-цены: [start, open, high, low, close]
    s = SimpleNamespace(get_mark_price_kline=session.get_premium_index_price_kline)
    return download_klines(s, call, symbol, store.KIND_MARK, start_ms, end_ms, interval="60", progress_every=0)


# --------------------------------------------------------- внешние данные
def download_fear_greed(path: Path, timeout: float = 20) -> int:
    """Индекс страха и жадности (alternative.me), дневной, вся история. Необязательный."""
    import requests

    r = requests.get("https://api.alternative.me/fng/", params={"limit": 0, "format": "json"}, timeout=timeout)
    r.raise_for_status()
    data = r.json().get("data") or []
    df = pd.DataFrame({"ts": [int(x["timestamp"]) * 1000 for x in data],
                       "value": [float(x["value"]) for x in data]}).sort_values("ts")
    df.to_parquet(path, index=False)
    return len(df)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


__all__ = ["select_universe", "download_symbol_basket", "download_fear_greed", "all_linear_instruments",
           "series_path", "DAY_MS"]
