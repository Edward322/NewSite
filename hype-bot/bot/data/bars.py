"""Минутные данные в numpy и сборка свечей старшего таймфрейма.

Свеча таймфрейма tf с временем открытия T покрывает минуты [T, T + tf) и
считается закрытой в момент T + tf — как kline у Bybit (границы по UTC).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bot.data import store

MINUTE_MS = 60_000
TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
              "6h": 360, "8h": 480, "12h": 720, "1d": 1440}


def to_ms(t: str | pd.Timestamp | int | None) -> int | None:
    if t is None:
        return None
    if isinstance(t, (int, np.integer)):
        return int(t)
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.value // 1_000_000)


@dataclass
class MinuteData:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    mark_open: np.ndarray
    mark_high: np.ndarray
    mark_low: np.ndarray
    mark_close: np.ndarray
    base_ms: int = MINUTE_MS    # шаг ряда: 1 минута (HYPE) или 15 минут (корзина)

    def __len__(self) -> int:
        return len(self.ts)

    def slice(self, start_ms: int | None = None, end_ms: int | None = None) -> "MinuteData":
        lo = 0 if start_ms is None else int(np.searchsorted(self.ts, start_ms, "left"))
        hi = len(self.ts) if end_ms is None else int(np.searchsorted(self.ts, end_ms, "left"))
        return MinuteData(**{k: (v[lo:hi] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})

    def select(self, mask: np.ndarray) -> "MinuteData":
        return MinuteData(**{k: (v[mask] if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()})

    @classmethod
    def from_frames(cls, last: pd.DataFrame, mark: pd.DataFrame | None = None) -> "MinuteData":
        last = last.sort_values("ts")
        if mark is not None and len(mark):
            m = last[["ts"]].merge(mark, on="ts", how="left")
            # минута без mark-цены (бывает только на краях) — берём последнюю цену
            for c in ("open", "high", "low", "close"):
                m[c] = np.where(m[c].isna(), last[c].to_numpy(), m[c].to_numpy())
        else:
            m = last
        f = lambda s: s.to_numpy(dtype="float64")  # noqa: E731
        return cls(ts=last["ts"].to_numpy(dtype="int64"), open=f(last["open"]), high=f(last["high"]),
                   low=f(last["low"]), close=f(last["close"]), volume=f(last["volume"]),
                   mark_open=f(m["open"]), mark_high=f(m["high"]), mark_low=f(m["low"]),
                   mark_close=f(m["close"]))


def load_minutes(data_dir: Path, symbol: str, start: str | int | None = None,
                 end: str | int | None = None, with_mark: bool = True) -> MinuteData:
    last = pd.read_parquet(store.kline_path(data_dir, symbol, store.KIND_LAST))
    mark = None
    mpath = store.kline_path(data_dir, symbol, store.KIND_MARK)
    if with_mark and Path(mpath).exists():
        mark = pd.read_parquet(mpath)
    return MinuteData.from_frames(last, mark).slice(to_ms(start), to_ms(end))


def load_funding(data_dir: Path, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    f = pd.read_parquet(store.funding_path(data_dir, symbol)).sort_values("ts")
    return f["ts"].to_numpy(dtype="int64"), f["rate"].to_numpy(dtype="float64")


@dataclass
class Bars:
    """Свечи таймфрейма + индексы их строк в MinuteData: строки [m_start, m_end)."""
    tf: str
    ts: np.ndarray          # время открытия
    close_ts: np.ndarray    # время закрытия = ts + tf
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    m_start: np.ndarray
    m_end: np.ndarray

    def __len__(self) -> int:
        return len(self.ts)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({"ts": self.ts, "open": self.open, "high": self.high, "low": self.low,
                             "close": self.close, "volume": self.volume})


def aggregate(md: MinuteData, tf: str) -> Bars:
    """Собирает полные свечи tf из ряда с шагом md.base_ms. Неполные свечи отбрасываются."""
    step = TF_MINUTES[tf] * MINUTE_MS
    if step % md.base_ms:
        raise ValueError(f"Таймфрейм {tf} не кратен шагу данных {md.base_ms // MINUTE_MS} мин")
    n = step // md.base_ms
    if len(md) == 0:
        e = np.array([], dtype="int64")
        z = np.array([], dtype="float64")
        return Bars(tf, e, e, z, z, z, z, z, e, e)
    bucket = md.ts // step * step
    starts = np.flatnonzero(np.r_[True, bucket[1:] != bucket[:-1]])
    ends = np.r_[starts[1:], len(md)]
    # Агрегаты считаются по ВСЕМ группам, и только потом неполные отбрасываются,
    # иначе reduceat склеил бы соседние группы.
    high = np.maximum.reduceat(md.high, starts)
    low = np.minimum.reduceat(md.low, starts)
    volume = np.add.reduceat(md.volume, starts)
    full = (ends - starts) == n   # свеча полная, если у неё все n строк
    starts, ends = starts[full], ends[full]
    ts = bucket[starts]
    return Bars(
        tf=tf, ts=ts, close_ts=ts + step,
        open=md.open[starts], close=md.close[ends - 1],
        high=high[full], low=low[full], volume=volume[full],
        m_start=starts.astype("int64"), m_end=ends.astype("int64"),
    )
