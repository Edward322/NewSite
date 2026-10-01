import zipfile

import pandas as pd
import pytest

from bot.data import downloader, store, transfer
from bot.data.downloader import MINUTE_MS, Caller
from tests.fake_bybit import FakeBybit

LAUNCH = 1_735_689_600_000
NOW = LAUNCH + 2_500 * MINUTE_MS + 1


def _download(tmp_path):
    fake = FakeBybit(LAUNCH, NOW)
    downloader.download_symbol(fake, Caller(pause_s=0, sleep=lambda s: None), tmp_path / "src",
                               "HYPEUSDT", now_ms=NOW)


def test_pack_unpack_roundtrip(tmp_path):
    _download(tmp_path)
    out = tmp_path / "upload" / "hype-data.zip"
    m = transfer.pack(tmp_path / "src", ["HYPEUSDT", "SOLUSDT"], out)
    assert "HYPEUSDT/kline_last_1m.parquet" in m["files"]
    m2 = transfer.unpack(out, tmp_path / "dst", tmp_path / "manifest.json")
    assert m2["files"] == m["files"]
    a = pd.read_parquet(tmp_path / "src/HYPEUSDT/kline_last_1m.parquet")
    b = pd.read_parquet(tmp_path / "dst/HYPEUSDT/kline_last_1m.parquet")
    pd.testing.assert_frame_equal(a, b)


def test_unpack_detects_tampering(tmp_path):
    _download(tmp_path)
    out = tmp_path / "hype-data.zip"
    transfer.pack(tmp_path / "src", ["HYPEUSDT"], out)
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(bad, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith("funding.parquet"):
                data = data[:-10] + b"x" * 10
            zout.writestr(item, data)
    with pytest.raises(transfer.TransferError, match="Контрольная сумма"):
        transfer.unpack(bad, tmp_path / "dst", tmp_path / "manifest.json")


def test_unpack_rejects_unexpected_paths(tmp_path):
    evil = tmp_path / "evil.zip"
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("manifest.json", '{"files": {}}')
        z.writestr("raw/../../etc/passwd.json", "{}")
    with pytest.raises(transfer.TransferError, match="Неожиданный файл"):
        transfer.unpack(evil, tmp_path / "dst", tmp_path / "manifest.json")


def test_interrupted_download_resumes_from_last_saved_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "CHUNK_REQUESTS", 1)  # сохранять после каждого запроса
    now = LAUNCH + 5_500 * MINUTE_MS + 1
    fake = FakeBybit(LAUNCH, now)
    calls = {"n": 0}
    real = fake.get_kline

    def flaky(**p):
        calls["n"] += 1
        if calls["n"] == 4:
            raise downloader.DownloadError("связь пропала")
        return real(**p)

    fake.get_kline = flaky
    c = Caller(pause_s=0, sleep=lambda s: None)
    with pytest.raises(downloader.DownloadError):
        downloader.download_symbol(fake, c, tmp_path, "HYPEUSDT", now_ms=now)
    saved = pd.read_parquet(store.kline_path(tmp_path, "HYPEUSDT", store.KIND_LAST))
    assert len(saved) == 3_000  # три полных окна по 1000 свечей сохранились

    fake2 = FakeBybit(LAUNCH, now)
    s = downloader.download_symbol(fake2, c, tmp_path, "HYPEUSDT", now_ms=now)
    first = next(p for n, p in fake2.calls if n == "kline")
    assert first["start"] == LAUNCH + 3_000 * MINUTE_MS
    assert s["kline_last_rows"] == 5_500
    df = pd.read_parquet(store.kline_path(tmp_path, "HYPEUSDT", store.KIND_LAST))
    assert df["ts"].is_unique and (df["ts"].diff().dropna() == MINUTE_MS).all()
