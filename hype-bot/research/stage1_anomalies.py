"""Этап 1: разбор аномалий, найденных отчётом о качестве данных.

Запуск: python -m research.stage1_anomalies > reports/stage1_anomalies.txt
Отвечает на вопрос, какие выбросы — реальные рыночные события, а какие — дефекты.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from bot.config import load_config

warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

CFG = load_config()
DATA = CFG.data_dir()
SYMBOLS = [CFG.symbol] + list(CFG.data.extra_symbols)


def load(sym: str, kind: str = "last") -> pd.DataFrame:
    df = pd.read_parquet(DATA / sym / f"kline_{kind}_1m.parquet")
    df["t"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("t")


def section(title: str) -> None:
    print(f"\n=== {title}")


def main() -> None:
    h, m = load(CFG.symbol), load(CFG.symbol, "mark")
    f = pd.read_parquet(DATA / CFG.symbol / "funding.parquet")
    f["t"] = pd.to_datetime(f["ts"], unit="ms", utc=True)

    section("Начало истории HYPEUSDT")
    print("Первые записи финансирования:")
    print(f.head(3)[["t", "rate"]].to_string(index=False))
    print("Первые свечи:")
    print(h.head(3)[["open", "high", "low", "close", "volume", "turnover"]])

    section("Свечи с нулевым объёмом HYPEUSDT по месяцам")
    z = h[h["volume"] == 0]
    print(z.groupby(z.index.strftime("%Y-%m")).size().to_string())

    section("Медианный дневной оборот HYPEUSDT по кварталам, млн USDT")
    d = h["turnover"].resample("D").sum()
    print((d.groupby(d.index.tz_localize(None).to_period("Q")).median() / 1e6).round(1).to_string())

    section("Минуты с расхождением last/mark ≥ 1% по месяцам")
    dv = (h["close"] / m["close"] - 1).abs()
    x = dv[dv >= 0.01]
    print(x.groupby(x.index.strftime("%Y-%m")).size().to_string())

    section("Обвал 2025-01-13 02:23 UTC: последняя цена против mark")
    w = slice("2025-01-13 02:21", "2025-01-13 02:25")
    print(pd.concat([h.loc[w, ["open", "high", "low", "close", "volume"]],
                     m.loc[w, ["low", "close"]].add_prefix("mark_")], axis=1))

    section("Обвал 2025-10-10: минимум 21:00–22:00 UTC относительно закрытия 21:00, все контракты")
    for s in SYMBOLS:
        k = load(s)
        pre = k.loc["2025-10-10 21:00", "close"]
        lo = k.loc["2025-10-10 21:00":"2025-10-10 22:00", "low"].min()
        lo_mark = ""
        if s == CFG.symbol:
            lo_mark = f"; mark-цена: {m.loc['2025-10-10 21:00':'2025-10-10 22:00', 'low'].min() / pre - 1:+.1%}"
        print(f"{s:9s} {lo / pre - 1:+.1%}{lo_mark}")

    section("Экстремальное финансирование (|ставка| ≥ 0,5%)")
    for s in SYMBOLS:
        ff = pd.read_parquet(DATA / s / "funding.parquet")
        e = ff[ff["rate"].abs() >= 0.005]
        rows = [(pd.to_datetime(t, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M"), r)
                for t, r in zip(e["ts"], e["rate"])]
        print(f"{s:9s} {rows if rows else 'нет'}")

    section("Смена шага цены HYPEUSDT")
    fine = np.zeros(len(h), bool)
    for c in ("open", "high", "low", "close"):
        fine |= ((h[c] * 1000).round().astype("int64") % 10 != 0).to_numpy()
    last_fine = h.index[fine].max()
    print(f"Последняя свеча с ценой точностью 0.001: {last_fine}")
    print(f"Медианная цена после смены: {h.loc[h.index > last_fine, 'close'].median():.2f}")

    section("Предлагаемые границы исследования")
    holdout_start = h.index.max().normalize() - pd.DateOffset(months=CFG.research.holdout_months)
    dev = h.loc[pd.Timestamp("2025-01-01", tz="UTC"):holdout_start - pd.Timedelta(minutes=1)]
    hold = h.loc[holdout_start:]
    print(f"Разработка (подбор, walk-forward): {dev.index.min()} — {dev.index.max()} "
          f"({len(dev):,} мин, {len(dev) / 1440:.0f} дн.)")
    print(f"Отложенные данные: {hold.index.min()} — {hold.index.max()} "
          f"({len(hold):,} мин, {len(hold) / 1440:.0f} дн.)")


if __name__ == "__main__":
    main()
