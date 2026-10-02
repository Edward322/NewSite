import numpy as np
import pandas as pd
import pytest

from bot.data import downloader, store
from bot.data.downloader import MINUTE_MS, Caller
from tests.fake_bybit import FakeBybit

LAUNCH = 1_733_000_000_000 - (1_733_000_000_000 % MINUTE_MS) + 17 * MINUTE_MS
NOW = LAUNCH + 3_456 * MINUTE_MS + 25_000  # посреди формирующейся свечи


def call():
    return Caller(pause_s=0, sleep=lambda s: None)


def test_klines_cover_range_without_gaps_or_duplicates():
    fake = FakeBybit(LAUNCH, NOW)
    df = downloader.download_klines(fake, call(), "HYPEUSDT", store.KIND_LAST, LAUNCH, NOW)
    ts = df["ts"].to_numpy()
    assert ts[0] == LAUNCH
    assert np.all(np.diff(ts) == MINUTE_MS)
    assert len(ts) == 3_456  # ровно все закрытые свечи


def test_unclosed_candle_is_never_saved():
    fake = FakeBybit(LAUNCH, NOW)
    df = downloader.download_klines(fake, call(), "HYPEUSDT", store.KIND_LAST, LAUNCH, NOW)
    forming_open = (NOW // MINUTE_MS) * MINUTE_MS
    assert df["ts"].max() + MINUTE_MS <= NOW
    assert forming_open not in set(df["ts"])


def test_window_requests_respect_limit_and_do_not_overlap():
    fake = FakeBybit(LAUNCH, NOW)
    downloader.download_klines(fake, call(), "HYPEUSDT", store.KIND_LAST, LAUNCH, NOW)
    windows = [(p["start"], p["end"]) for name, p in fake.calls if name == "kline"]
    for (s1, e1), (s2, _) in zip(windows, windows[1:]):
        assert s2 == e1 + 1
    for s, e in windows:
        assert (e - s) // MINUTE_MS + 1 <= 1000


def test_exchange_side_gaps_are_kept_as_gaps():
    missing = {LAUNCH + 100 * MINUTE_MS, LAUNCH + 101 * MINUTE_MS}
    fake = FakeBybit(LAUNCH, NOW, missing=missing)
    df = downloader.download_klines(fake, call(), "HYPEUSDT", store.KIND_LAST, LAUNCH, NOW)
    assert not (missing & set(df["ts"]))
    assert len(df) == 3_456 - 2


def test_mark_klines_have_mark_columns():
    fake = FakeBybit(LAUNCH, NOW)
    df = downloader.download_klines(fake, call(), "HYPEUSDT", store.KIND_MARK, LAUNCH, NOW)
    assert list(df.columns) == store.MARK_COLUMNS


def test_funding_paginates_backwards_and_is_complete():
    launch = 1_700_006_400_000  # кратно 8 ч, как метки финансирования
    now = launch + 400 * 8 * 3_600_000 + 5
    fake = FakeBybit(launch, now)
    df = downloader.download_funding(fake, call(), "HYPEUSDT", launch, now)
    diffs = np.diff(df["ts"].to_numpy())
    assert len(df) == 401
    assert np.all(diffs == 8 * 3_600_000)
    assert sum(1 for n, _ in fake.calls if n == "funding") >= 3


def test_download_symbol_resumes_incrementally(tmp_path):
    fake = FakeBybit(LAUNCH, NOW)
    s1 = downloader.download_symbol(fake, call(), tmp_path, "HYPEUSDT", now_ms=NOW)
    assert s1["kline_last_rows"] == 3_456
    later = NOW + 50 * MINUTE_MS
    fake2 = FakeBybit(LAUNCH, later)
    s2 = downloader.download_symbol(fake2, call(), tmp_path, "HYPEUSDT", now_ms=later)
    assert s2["kline_last_new"] == 50
    assert s2["kline_last_rows"] == 3_506
    first_req = next(p for n, p in fake2.calls if n == "kline")
    assert first_req["start"] == LAUNCH + 3_456 * MINUTE_MS
    df = pd.read_parquet(store.kline_path(tmp_path, "HYPEUSDT", store.KIND_LAST))
    assert df["ts"].is_unique and df["ts"].is_monotonic_increasing


def test_risk_limit_follows_cursor(tmp_path):
    fake = FakeBybit(LAUNCH, NOW)
    tiers = downloader.fetch_risk_limits(fake, call(), "HYPEUSDT")
    assert [t["id"] for t in tiers] == [1, 2]


def test_history_days_limits_start(tmp_path):
    fake = FakeBybit(LAUNCH, NOW)
    s = downloader.download_symbol(fake, call(), tmp_path, "SOLUSDT", history_days=1,
                                   kinds=(store.KIND_LAST,), funding=False, now_ms=NOW)
    assert s["start_ms"] == NOW - 86_400_000
    assert s["kline_last_rows"] == 1440


def test_caller_retries_network_errors_then_succeeds():
    import requests

    attempts = {"n": 0}

    def flaky(**_):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise requests.exceptions.ConnectionError("boom")
        return {"retCode": 0, "result": {}}

    sleeps = []
    c = Caller(pause_s=0, sleep=sleeps.append)
    assert c(flaky)["retCode"] == 0
    assert attempts["n"] == 3
    assert sleeps == [1.0, 2.0]


def test_caller_does_not_retry_403():
    from pybit.exceptions import FailedRequestError

    attempts = {"n": 0}

    def forbidden(**_):
        attempts["n"] += 1
        raise FailedRequestError(request="GET", message="x", status_code=403, time="0", resp_headers={})

    with pytest.raises(downloader.DownloadError, match="403"):
        Caller(pause_s=0, sleep=lambda s: None)(forbidden)
    assert attempts["n"] == 1


def test_manifest(tmp_path):
    fake = FakeBybit(LAUNCH, NOW)
    downloader.download_symbol(fake, call(), tmp_path, "HYPEUSDT", now_ms=NOW)
    m = downloader.write_manifest(tmp_path, tmp_path / "manifest.json", ["HYPEUSDT"])
    entry = m["files"]["HYPEUSDT/kline_last_1m.parquet"]
    assert entry["rows"] == 3_456 and len(entry["sha256"]) == 64


def test_caller_reports_proxy_block_without_retries():
    import requests

    attempts = {"n": 0}

    def blocked(**_):
        attempts["n"] += 1
        raise requests.exceptions.ProxyError("Tunnel connection failed: 403 Forbidden")

    with pytest.raises(downloader.DownloadError, match="блокирует сеть"):
        Caller(pause_s=0, sleep=lambda s: None)(blocked)
    assert attempts["n"] == 1


def _invalid(code):
    from pybit.exceptions import InvalidRequestError
    return InvalidRequestError(request="GET /v5/market/kline", message="svc error: Get kline failed",
                               status_code=code, time="0", resp_headers={})


def test_caller_retries_bybit_server_error_10016():
    attempts = {"n": 0}

    def flaky(**_):
        attempts["n"] += 1
        if attempts["n"] < 4:
            raise _invalid(10016)
        return {"retCode": 0, "result": {}}

    sleeps = []
    assert Caller(pause_s=0, sleep=sleeps.append)(flaky)["retCode"] == 0
    assert attempts["n"] == 4 and sleeps == [1.0, 2.0, 4.0]


def test_caller_does_not_retry_parameter_errors():
    attempts = {"n": 0}

    def bad(**_):
        attempts["n"] += 1
        raise _invalid(10001)

    with pytest.raises(downloader.DownloadError, match="отклонил"):
        Caller(pause_s=0, sleep=lambda s: None)(bad)
    assert attempts["n"] == 1
