import numpy as np
import pandas as pd

from bot.data import quality
from bot.data.downloader import MINUTE_MS

T0 = 1_735_689_600_000  # 2025-01-01 00:00 UTC


def make(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.0005
    low = np.minimum(open_, close) * 0.9995
    return pd.DataFrame({"ts": T0 + np.arange(n, dtype="int64") * MINUTE_MS, "open": open_,
                         "high": high, "low": low, "close": close,
                         "volume": 10.0, "turnover": 200.0})


def test_clean_data_has_no_issues():
    rep = quality.check_klines(make(), "x")
    assert rep.missing_total == 0 and not rep.gaps
    assert rep.duplicates == rep.ohlc_violations == rep.non_positive == rep.nan_rows == 0
    assert rep.misaligned == 0 and rep.zero_volume == 0
    assert len(rep.return_outliers) == 0


def test_detects_gaps():
    df = make().drop(index=[100, 101, 102, 2000]).reset_index(drop=True)
    rep = quality.check_klines(df, "x")
    assert rep.missing_total == 4
    assert sorted(g.missing for g in rep.gaps) == [1, 3]
    assert rep.gaps[0].start_ms == T0 + 100 * MINUTE_MS


def test_detects_ohlc_violation_and_zero_volume_run():
    df = make()
    df.loc[10, "high"] = df.loc[10, "close"] * 0.9
    df.loc[500:504, "volume"] = 0
    rep = quality.check_klines(df, "x")
    assert rep.ohlc_violations == 1
    assert rep.zero_volume == 5 and rep.longest_zero_volume_run == 5


def test_detects_return_spike_and_wick():
    df = make()
    df.loc[1500:, ["open", "high", "low", "close"]] *= 1.08  # скачок +8% за минуту
    df.loc[1500, "open"] = df.loc[1499, "close"]
    df.loc[1500, "low"] = min(df.loc[1500, "low"], df.loc[1500, "open"])
    df.loc[2500, "low"] = df.loc[2500, "close"] * 0.85  # тень 15%
    rep = quality.check_klines(df, "x")
    assert len(rep.return_outliers) == 1
    assert len(rep.wick_outliers) == 1


def test_detects_duplicates_and_misalignment():
    df = make(100)
    df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    df.loc[50, "ts"] += 7
    rep = quality.check_klines(df, "x")
    assert rep.duplicates == 1
    assert rep.misaligned == 1


def test_compare_last_mark():
    last = make(200)
    mark = last[["ts", "open", "high", "low", "close"]].copy()
    mark.loc[50, "close"] *= 1.05
    mark = mark.drop(index=[10])
    mc = quality.compare_last_mark(last, mark)
    assert mc.only_last == 1 and mc.only_mark == 0
    assert mc.over_warn == 1


def test_check_funding():
    ts = T0 + np.arange(10, dtype="int64") * 8 * 3_600_000
    ts = np.r_[ts, ts[-1] + 4 * 3_600_000]
    df = pd.DataFrame({"ts": ts, "rate": [0.0001] * 10 + [0.03]})
    fr = quality.check_funding(df, -0.02, 0.02)
    assert fr.intervals_h == {"4.0": 1, "8.0": 9}
    assert fr.out_of_bounds == 1
    assert len(fr.extreme) == 1


def test_report_end_to_end(tmp_path):
    from bot.data import downloader
    from bot.data.report import build_report
    from tests.fake_bybit import FakeBybit

    launch = T0
    now = T0 + 3 * 86_400_000 + 1
    fake = FakeBybit(launch, now, missing={T0 + 60 * MINUTE_MS})
    c = downloader.Caller(pause_s=0, sleep=lambda s: None)
    downloader.download_symbol(fake, c, tmp_path, "HYPEUSDT", now_ms=now)
    text = build_report(tmp_path, ["HYPEUSDT"])
    assert "пропущено 1 " in text
    assert "minNotionalValue: 5" in text
    assert "Финансирование" in text and "mark-цена" in text
