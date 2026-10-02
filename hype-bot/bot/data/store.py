"""Локальное хранилище рыночных данных (parquet + json).

Раскладка:
    <data_dir>/<SYMBOL>/kline_last_1m.parquet   свечи по последней цене
    <data_dir>/<SYMBOL>/kline_mark_1m.parquet   свечи по mark-цене
    <data_dir>/<SYMBOL>/funding.parquet         история ставок финансирования
    <data_dir>/<SYMBOL>/instrument.json         параметры инструмента
    <data_dir>/<SYMBOL>/risk_limit.json         уровни лимита риска (MMR)

Время везде — int64, миллисекунды UTC, время ОТКРЫТИЯ свечи.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

KLINE_COLUMNS = ["ts", "open", "high", "low", "close", "volume", "turnover"]
MARK_COLUMNS = ["ts", "open", "high", "low", "close"]
FUNDING_COLUMNS = ["ts", "rate"]

KIND_LAST = "last"
KIND_MARK = "mark"


def symbol_dir(data_dir: Path, symbol: str) -> Path:
    d = Path(data_dir) / symbol
    d.mkdir(parents=True, exist_ok=True)
    return d


def kline_path(data_dir: Path, symbol: str, kind: str, interval: str = "1") -> Path:
    suffix = f"{interval}m" if interval.isdigit() else interval
    return symbol_dir(data_dir, symbol) / f"kline_{kind}_{suffix}.parquet"


def funding_path(data_dir: Path, symbol: str) -> Path:
    return symbol_dir(data_dir, symbol) / "funding.parquet"


def json_path(data_dir: Path, symbol: str, name: str) -> Path:
    return symbol_dir(data_dir, symbol) / f"{name}.json"


def read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame({c: pd.Series(dtype="int64" if c == "ts" else "float64") for c in columns})
    return pd.read_parquet(path)


def merge_and_write(path: Path, new: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Объединяет с уже сохранёнными данными, убирает дубли по ts, пишет атомарно."""
    old = read_parquet(path, columns)
    frames = [f for f in (old, new) if len(f)]
    df = pd.concat(frames, ignore_index=True) if frames else new
    df = df[columns]
    # При совпадении ts побеждает более свежая загрузка (new идёт вторым).
    df = df.drop_duplicates(subset="ts", keep="last").sort_values("ts").reset_index(drop=True)
    df["ts"] = df["ts"].astype("int64")
    tmp = Path(str(path) + ".tmp")
    df.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(path)
    return df


def write_json(path: Path, payload: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
