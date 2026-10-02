"""Качество данных корзины → reports/BASKET_DATA.md.

Проверяются все ряды каждой монеты: свечи 15m (пропуски, дубли, OHLC, выбросы),
финансирование (интервалы, крайние ставки), ряды 1h (премиальный индекс, открытый
интерес, лонг/шорт) и индекс страха и жадности. Ничего не исправляется.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.config import PROJECT_ROOT
from bot.data import quality, store
from bot.data.downloader import fmt_ts
from bot.data.panel import BASE_MS, BASKET_DIR, HOUR_MS

OUT = PROJECT_ROOT / "reports" / "BASKET_DATA.md"


def series_row(df: pd.DataFrame | None, step: int) -> dict:
    if df is None or not len(df):
        return {"строк": 0}
    ts = np.unique(df["ts"].to_numpy("int64"))
    gaps = quality.find_gaps(ts, step)
    return {"с": fmt_ts(int(ts[0]))[:10], "по": fmt_ts(int(ts[-1])), "строк": len(df),
            "пропущено": int(sum(g.missing for g in gaps)), "разрывов": len(gaps),
            "крупнейший, ч": round(max((g.missing for g in gaps), default=0) * step / HOUR_MS, 1),
            "дубли": int(len(df) - len(ts))}


def main() -> None:
    u = store.read_json(BASKET_DIR / "universe.json")
    symbols = u["tradable"] + u["signal_only"]
    out = ["# Качество данных корзины монет", "",
           f"Архив `data-v2`, загружен {u['generated_at'][:16].replace('T', ' ')} UTC. "
           "Все числа посчитаны скриптом `research/basket_quality.py` по распакованным файлам.", "",
           f"Торгуемые монеты ({len(u['tradable'])}): {', '.join(u['tradable'])}. "
           f"Только для сигналов: {', '.join(u['signal_only'])}.", "",
           "Правило отбора: " + ", ".join(f"{k}={v}" for k, v in u["rule"].items()) + ".", "",
           "**Смещение выживших.** Монеты отобраны по обороту на дату загрузки: те, что за 2023–2026 "
           "годы упали в забвение или были делистингованы, в корзину не попали. Поэтому buy & hold "
           "корзины и любые лонговые стратегии на истории выглядят лучше, чем было бы в реальном "
           "времени. Это ограничение учитывается при выводах.", ""]

    rows = []
    for s in symbols:
        k = pd.read_parquet(store.symbol_dir(BASKET_DIR, s) / "kline_last_15m.parquet")
        rep = quality.check_klines(k, s, BASE_MS)
        rows.append({"монета": s, "с": fmt_ts(rep.first_ms)[:10], "свечей": rep.rows,
                     "пропущено": rep.missing_total, "разрывов": len(rep.gaps),
                     "крупнейший, ч": round(max((g.missing for g in rep.gaps), default=0) * BASE_MS / HOUR_MS, 1),
                     "дубли": rep.duplicates, "OHLC": rep.ohlc_violations, "цена≤0": rep.non_positive,
                     "нулевой объём": rep.zero_volume,
                     "выбросы": 0 if rep.return_outliers is None else len(rep.return_outliers),
                     "тени ≥5%": 0 if rep.wick_outliers is None else len(rep.wick_outliers)})
    out += ["## Свечи 15m (последняя цена)", "",
            f"Выброс — |лог-доходность| ≥ {quality.RET_ABS_MIN:.0%} за 15 минут и ≥ {quality.RET_ROBUST_Z:g} "
            "устойчивых отклонений от обычного движения.", "",
            quality_table(pd.DataFrame(rows)), ""]

    frows = []
    for s in symbols:
        f = pd.read_parquet(store.funding_path(BASKET_DIR, s))
        inst = store.read_json(store.json_path(BASKET_DIR, s, "instrument"))["instrument"]
        fr = quality.check_funding(f)
        q = fr.rate_q
        frows.append({"монета": s, "записей": fr.rows, "интервалы, ч → раз": fr.intervals_h,
                      "сейчас, мин": inst.get("fundingInterval"),
                      "мин, %": round(q["min"] * 100, 4), "медиана, %": round(q["p50"] * 100, 4),
                      "макс, %": round(q["max"] * 100, 4), "|ставка| ≥ 0,5%": len(fr.extreme)})
    out += ["## Финансирование", "",
            "Ставка за один период. Интервал между выплатами у некоторых монет менялся — бэктестер "
            "берёт фактические метки выплат из ряда, а не постоянный интервал.", "",
            quality_table(pd.DataFrame(frows)), ""]

    for name, title in (("premium_1h", "Премиальный индекс (базис), 1h"), ("oi_1h", "Открытый интерес, 1h"),
                        ("lsr_1h", "Доля лонгов по счетам, 1h")):
        rr = []
        for s in symbols:
            p = store.symbol_dir(BASKET_DIR, s) / f"{name}.parquet"
            df = pd.read_parquet(p) if p.exists() else None
            row = {"монета": s, **series_row(df, HOUR_MS)}
            if df is not None and len(df):
                col = df.columns[-1] if name != "premium_1h" else "close"
                v = df[col]
                row.update({"NaN": int(v.isna().sum()), "мин": float(f"{v.min():.4g}"),
                            "медиана": float(f"{v.median():.4g}"), "макс": float(f"{v.max():.4g}")})
            rr.append(row)
        out += [f"## {title}", "", quality_table(pd.DataFrame(rr)), ""]

    fg = pd.read_parquet(BASKET_DIR / "fear_greed.parquet")
    fgr = series_row(fg, 86_400_000)
    out += ["## Индекс страха и жадности (alternative.me, дневной)", "",
            f"Период {fgr['с']} — {fgr['по']}, дней {fgr['строк']}, пропущено {fgr['пропущено']} "
            f"в {fgr['разрывов']} разрывах; значения {fg['value'].min():.0f}–{fg['value'].max():.0f}.", ""]

    out += ["## Когда значения считаются известными (защита от заглядывания в будущее)", "",
            "- свеча 15m — после её закрытия;",
            "- финансирование — в момент выплаты;",
            "- премиальный индекс, открытый интерес, лонг/шорт (метка — начало часа) — через 1 час после метки;",
            "- индекс страха и жадности за день — с начала следующего дня UTC.", "",
            "Mark-цены для корзины не загружались (экономия объёма): ликвидация и финансирование "
            "считаются по последней цене. При правиле «ликвидация ≥ 2× дальше стопа» это влияет "
            "только на редкие гэпы.", ""]
    OUT.write_text("\n".join(out), encoding="utf-8")
    print(OUT.read_text(encoding="utf-8"))


def quality_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


if __name__ == "__main__":
    main()
