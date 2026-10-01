"""Проверка качества загруженных данных: пропуски, дубли, аномалии.

Ничего не исправляет и не удаляет — только находит и описывает. Решение о том,
что делать с аномалиями, принимается явно на этапе исследования.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.data.downloader import MINUTE_MS, fmt_ts

# Пороги аномалий для минутных свечей.
RET_ABS_MIN = 0.03          # |лог-доходность| за минуту не меньше 3% ...
RET_ROBUST_Z = 12.0         # ... и в 12+ раз больше обычной (устойчивая оценка)
WICK_ABS_MIN = 0.05         # тень свечи ≥ 5% от цены закрытия
MARK_DIFF_WARN = 0.01       # расхождение last/mark ≥ 1%
FUNDING_ABS_WARN = 0.005    # |ставка| ≥ 0,5% за период


@dataclass
class Gap:
    start_ms: int   # первая отсутствующая свеча
    end_ms: int     # последняя отсутствующая свеча
    missing: int


@dataclass
class KlineReport:
    name: str
    rows: int
    first_ms: int | None
    last_ms: int | None
    duplicates: int
    non_monotonic: int
    misaligned: int
    nan_rows: int
    ohlc_violations: int
    non_positive: int
    zero_volume: int | None
    longest_zero_volume_run: int | None
    missing_total: int
    gaps: list[Gap] = field(default_factory=list)
    return_outliers: pd.DataFrame | None = None
    wick_outliers: pd.DataFrame | None = None

    @property
    def expected(self) -> int:
        if self.first_ms is None:
            return 0
        return (self.last_ms - self.first_ms) // MINUTE_MS + 1


def find_gaps(ts: np.ndarray, step: int = MINUTE_MS) -> list[Gap]:
    if len(ts) < 2:
        return []
    diffs = np.diff(ts)
    idx = np.nonzero(diffs > step)[0]
    return [Gap(int(ts[i] + step), int(ts[i + 1] - step), int(diffs[i] // step - 1)) for i in idx]


def _longest_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def check_klines(df: pd.DataFrame, name: str, step: int = MINUTE_MS) -> KlineReport:
    ts = df["ts"].to_numpy(dtype="int64")
    o, h, l, c = (df[k].to_numpy(dtype="float64") for k in ("open", "high", "low", "close"))
    has_volume = "volume" in df

    nan_rows = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    ohlc_bad = (h < np.maximum(o, c)) | (l > np.minimum(o, c)) | (h < l)
    non_pos = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)

    zero_vol = longest_zero = None
    if has_volume:
        vmask = df["volume"].to_numpy() == 0
        zero_vol = int(vmask.sum())
        longest_zero = _longest_run(vmask)

    uts = np.unique(ts)
    gaps = find_gaps(uts, step)

    # Выбросы доходности: сравниваем с устойчивым масштабом за скользящие сутки.
    ret_out = wick_out = None
    if len(df) > 10:
        logc = np.log(c)
        r = pd.Series(np.diff(logc, prepend=logc[0]), index=df.index)
        med_abs = r.abs().rolling(1440, min_periods=60).median().bfill()
        scale = (1.4826 * med_abs).replace(0, np.nan).fillna(r.abs().median() or 1e-9)
        z = r.abs() / scale
        mask = (r.abs() >= RET_ABS_MIN) & (z >= RET_ROBUST_Z)
        ret_out = pd.DataFrame({"time": [fmt_ts(t) for t in ts[mask.to_numpy()]],
                                "log_return": r[mask].round(4).to_numpy(),
                                "robust_z": z[mask].round(1).to_numpy()})
        upper = (h - np.maximum(o, c)) / c
        lower = (np.minimum(o, c) - l) / c
        wick = np.maximum(upper, lower)
        wmask = wick >= WICK_ABS_MIN
        wick_out = pd.DataFrame({"time": [fmt_ts(t) for t in ts[wmask]],
                                 "wick_pct": np.round(wick[wmask] * 100, 2),
                                 "close": c[wmask]})

    return KlineReport(
        name=name,
        rows=len(df),
        first_ms=int(ts.min()) if len(ts) else None,
        last_ms=int(ts.max()) if len(ts) else None,
        duplicates=int(len(ts) - len(uts)),
        non_monotonic=int((np.diff(ts) <= 0).sum()) if len(ts) > 1 else 0,
        misaligned=int((ts % step != 0).sum()),
        nan_rows=nan_rows,
        ohlc_violations=int(ohlc_bad.sum()),
        non_positive=int(non_pos.sum()),
        zero_volume=zero_vol,
        longest_zero_volume_run=longest_zero,
        missing_total=int(sum(g.missing for g in gaps)),
        gaps=gaps,
        return_outliers=ret_out,
        wick_outliers=wick_out,
    )


@dataclass
class MarkCompare:
    joined: int
    only_last: int
    only_mark: int
    abs_diff_q: dict
    over_warn: int
    worst: pd.DataFrame


def compare_last_mark(last: pd.DataFrame, mark: pd.DataFrame) -> MarkCompare:
    j = last[["ts", "close"]].merge(mark[["ts", "close"]], on="ts", how="inner", suffixes=("_last", "_mark"))
    d = (j["close_last"] / j["close_mark"] - 1).abs()
    q = {k: float(d.quantile(v)) if len(d) else float("nan")
         for k, v in (("p50", 0.5), ("p99", 0.99), ("p999", 0.999), ("max", 1.0))}
    worst = j.assign(diff_pct=(d * 100).round(3)).nlargest(10, "diff_pct")
    worst = worst.assign(time=[fmt_ts(t) for t in worst["ts"]])[["time", "close_last", "close_mark", "diff_pct"]]
    return MarkCompare(
        joined=len(j),
        only_last=int(len(set(last["ts"]) - set(mark["ts"]))),
        only_mark=int(len(set(mark["ts"]) - set(last["ts"]))),
        abs_diff_q=q,
        over_warn=int((d >= MARK_DIFF_WARN).sum()),
        worst=worst,
    )


@dataclass
class FundingReport:
    rows: int
    first_ms: int | None
    last_ms: int | None
    intervals_h: dict
    rate_q: dict
    extreme: pd.DataFrame
    out_of_bounds: int | None


def check_funding(df: pd.DataFrame, lower: float | None = None, upper: float | None = None) -> FundingReport:
    ts = df["ts"].to_numpy(dtype="int64")
    rates = df["rate"]
    diffs_h = pd.Series(np.diff(ts) / 3_600_000).round(2)
    intervals = {str(k): int(v) for k, v in diffs_h.value_counts().sort_index().items()}
    q = {k: float(rates.quantile(v)) if len(rates) else float("nan")
         for k, v in (("min", 0.0), ("p01", 0.01), ("p50", 0.5), ("p99", 0.99), ("max", 1.0))}
    ext = df[rates.abs() >= FUNDING_ABS_WARN]
    ext = pd.DataFrame({"time": [fmt_ts(t) for t in ext["ts"]], "rate_pct": (ext["rate"] * 100).round(4)})
    oob = None
    if lower is not None and upper is not None:
        oob = int(((rates < lower - 1e-12) | (rates > upper + 1e-12)).sum())
    return FundingReport(
        rows=len(df),
        first_ms=int(ts.min()) if len(ts) else None,
        last_ms=int(ts.max()) if len(ts) else None,
        intervals_h=intervals,
        rate_q=q,
        extreme=ext,
        out_of_bounds=oob,
    )


def early_listing_profile(df: pd.DataFrame, days: int = 14) -> pd.DataFrame:
    """Дневной размах и объём в первые дни после листинга против медианы всей истории."""
    d = df.assign(day=pd.to_datetime(df["ts"], unit="ms", utc=True).dt.floor("D"))
    daily = d.groupby("day").agg(high=("high", "max"), low=("low", "min"),
                                 close=("close", "last"), turnover=("turnover", "sum"),
                                 bars=("ts", "size"))
    daily["range_pct"] = (daily["high"] / daily["low"] - 1) * 100
    med_range = daily["range_pct"].median()
    med_turn = daily["turnover"].median()
    head = daily.head(days).copy()
    head["range_vs_median"] = (head["range_pct"] / med_range).round(2)
    head["turnover_vs_median"] = (head["turnover"] / med_turn).round(2)
    head.index = head.index.strftime("%Y-%m-%d")
    return head[["bars", "close", "range_pct", "range_vs_median", "turnover_vs_median"]].round(2)
