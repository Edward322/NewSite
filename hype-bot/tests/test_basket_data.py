import numpy as np
import pandas as pd
import pytest

from bot.data import basket, store, transfer
from bot.data.downloader import Caller
from tests.fake_bybit import H, MIN, FakeBybitBasket

LAUNCH = 1_735_689_600_000          # 2025-01-01
NOW = LAUNCH + 30 * 24 * H + 17 * MIN + 5


def inst(sym, base, launch=LAUNCH, min_qty="0.01", min_notional="5", **kw):
    d = {"symbol": sym, "baseCoin": base, "quoteCoin": "USDT", "settleCoin": "USDT",
         "contractType": "LinearPerpetual", "status": "Trading", "launchTime": str(launch),
         "lotSizeFilter": {"minOrderQty": min_qty, "minNotionalValue": min_notional, "qtyStep": min_qty},
         "symbolType": "", "underlyingTicker": "", "marketRegion": "", "isPreListing": False}
    d.update(kw)
    return d


RULE = {"listed_before": "2025-06-01", "min_turnover_24h": 1e7, "max_min_order_usdt": 5.5, "n_max": 3,
        "include": ["HYPEUSDT"]}


def test_universe_rule_filters_and_explains():
    ins = [inst("AAAUSDT", "AAA"), inst("BBBUSDT", "BBB"), inst("CCCUSDT", "CCC"), inst("DDDUSDT", "DDD"),
           inst("BTCUSDT", "BTC", min_qty="0.001"), inst("USDCUSDT", "USDC"),
           inst("TSLAUSDT", "TSLA", symbolType="stock"), inst("NEWUSDT", "NEW", launch=1_751_328_000_000),  # 2025-07-01, позже порога
           inst("LOWUSDT", "LOW"), inst("HYPEUSDT", "HYPE")]
    tk = [{"symbol": s, "lastPrice": p, "turnover24h": v} for s, p, v in (
        ("AAAUSDT", "1", "9e8"), ("BBBUSDT", "2", "8e8"), ("CCCUSDT", "3", "7e8"), ("DDDUSDT", "4", "6e8"),
        ("BTCUSDT", "100000", "1e10"), ("USDCUSDT", "1", "5e9"), ("TSLAUSDT", "300", "5e9"),
        ("NEWUSDT", "1", "5e9"), ("LOWUSDT", "1", "1e5"), ("HYPEUSDT", "84", "1e5"))]
    chosen, table = basket.select_universe(ins, tk, RULE)
    assert chosen == ["AAAUSDT", "BBBUSDT", "CCCUSDT", "HYPEUSDT"]       # n_max=3 по обороту + include
    why = dict(zip(table.symbol, table.reasons))
    assert "минимальный ордер 100.00" in why["BTCUSDT"]
    assert "стейблкоин" in why["USDCUSDT"] and "не криптовалюта" in why["TSLAUSDT"]
    assert "листинг позже" in why["NEWUSDT"] and "малый оборот" in why["LOWUSDT"]


def test_all_instruments_follow_cursor():
    fake = FakeBybitBasket(LAUNCH, NOW, instruments=[inst(f"X{i}USDT", f"X{i}") for i in range(5)])
    assert len(basket.all_linear_instruments(fake, Caller(pause_s=0, sleep=lambda s: None))) == 5


def test_download_symbol_basket_complete_and_resumable(tmp_path):
    fake = FakeBybitBasket(LAUNCH, NOW)
    c = Caller(pause_s=0, sleep=lambda s: None)
    s = basket.download_symbol_basket(fake, c, tmp_path, "HYPEUSDT", LAUNCH, NOW, "15")
    k = pd.read_parquet(store.kline_path(tmp_path, "HYPEUSDT", "last", "15"))
    assert (np.diff(k.ts) == 15 * MIN).all() and len(k) == 30 * 24 * 4 + 1   # последняя закрытая 15m свеча
    for name in ("oi_1h", "lsr_1h", "premium_1h"):
        d = pd.read_parquet(basket.series_path(tmp_path, "HYPEUSDT", name))
        assert d.ts.is_unique and (np.diff(d.ts) == H).all(), name
        assert d.ts.iloc[0] == LAUNCH, name
    assert s["oi_1h"] == 30 * 24 + 1          # почасовые точки включительно до текущего часа
    # дозагрузка через сутки продолжает с места остановки
    later = NOW + 24 * H
    fake2 = FakeBybitBasket(LAUNCH, later)
    s2 = basket.download_symbol_basket(fake2, c, tmp_path, "HYPEUSDT", LAUNCH, later, "15")
    assert s2["oi_1h"] == 31 * 24 + 1
    first_oi = next(p for n, p in fake2.calls if n == "oi")
    assert first_oi["startTime"] > NOW - 2 * H


def test_optional_series_failure_does_not_stop_download(tmp_path):
    fake = FakeBybitBasket(LAUNCH, NOW, fail={"lsr"})
    s = basket.download_symbol_basket(fake, Caller(pause_s=0, sleep=lambda s: None), tmp_path, "AAAUSDT",
                                      LAUNCH, NOW, "15")
    assert str(s["lsr_1h"]).startswith("ошибка") and s["oi_1h"] > 0 and s["klines"] > 0


def test_pack_with_root_files_roundtrip(tmp_path):
    src = tmp_path / "src"
    fake = FakeBybitBasket(LAUNCH, NOW)
    basket.download_symbol_basket(fake, Caller(pause_s=0, sleep=lambda s: None), src, "HYPEUSDT", LAUNCH, NOW, "15")
    store.write_json(src / "universe.json", {"tradable": ["HYPEUSDT"]})
    pd.DataFrame({"ts": [1, 2], "value": [10.0, 20.0]}).to_parquet(src / "fear_greed.parquet")
    out = tmp_path / "basket-data.zip"
    m = transfer.pack(src, ["HYPEUSDT"], out, extra_files=["universe.json", "fear_greed.parquet", "absent.csv"])
    assert "universe.json" in m["files"] and "fear_greed.parquet" in m["files"]
    m2 = transfer.unpack(out, tmp_path / "dst", tmp_path / "manifest.json")
    assert (tmp_path / "dst" / "fear_greed.parquet").exists() and m2["files"] == m["files"]


def test_cli_basket_end_to_end(tmp_path, monkeypatch):
    """Команда data basket целиком: отбор, загрузка в потоках, архив."""
    import bot.cli as cli
    from bot.data import downloader

    tk = [{"symbol": s, "lastPrice": "1", "turnover24h": "1e9"} for s in ("AAAUSDT", "BBBUSDT", "BTCUSDT", "ETHUSDT")]
    tk.append({"symbol": "HYPEUSDT", "lastPrice": "84", "turnover24h": "5e8"})
    old = 1_672_531_200_000   # 2023-01-01: раньше порога листинга из примера конфига
    ins = [inst("AAAUSDT", "AAA", launch=old), inst("BBBUSDT", "BBB", launch=old), inst("BTCUSDT", "BTC", min_qty="0.001"),
           inst("ETHUSDT", "ETH", min_qty="0.01"), inst("HYPEUSDT", "HYPE")]
    end = LAUNCH + 3 * 24 * H + 5
    monkeypatch.setattr(downloader, "make_public_session",
                        lambda *a, **k: FakeBybitBasket(LAUNCH, end, instruments=ins, tickers=tk))
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(basket, "download_fear_greed", lambda path: (_ for _ in ()).throw(OSError("нет сети")))
    rc = cli.main(["--config", str(__import__("bot.config").config.EXAMPLE_CONFIG), "data", "basket", "--threads", "3"])
    assert rc == 0
    out = tmp_path / "upload" / "basket-data.zip"
    m = transfer.unpack(out, tmp_path / "check", tmp_path / "m.json")
    files = set(m["files"])
    assert "universe.json" in files and "fear_greed.parquet" not in files
    for sym in ("AAAUSDT", "BBBUSDT", "HYPEUSDT", "BTCUSDT", "ETHUSDT"):   # BTC/ETH — только для сигналов
        assert f"{sym}/kline_last_15m.parquet" in files and f"{sym}/oi_1h.parquet" in files
