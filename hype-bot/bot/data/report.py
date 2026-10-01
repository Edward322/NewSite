"""Отчёт о качестве данных в Markdown (все цифры считаются из файлов данных)."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bot.data import quality, store
from bot.data.downloader import DAY_MS, fmt_ts

MAX_LIST = 15


def _table(df: pd.DataFrame | None, limit: int = MAX_LIST) -> str:
    if df is None or not len(df):
        return "_нет_\n"
    shown = df.head(limit)
    head = "| " + " | ".join(map(str, shown.columns)) + " |"
    sep = "|" + "---|" * len(shown.columns)
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in shown.itertuples(index=False)]
    more = f"\n_…и ещё {len(df) - limit}_" if len(df) > limit else ""
    return "\n".join([head, sep, *rows]) + more + "\n"


def _kline_section(rep: quality.KlineReport) -> str:
    if rep.rows == 0:
        return f"### {rep.name}\n\nФайл пуст.\n"
    span_days = (rep.last_ms - rep.first_ms) / DAY_MS
    lines = [
        f"### {rep.name}",
        "",
        f"- Период: {fmt_ts(rep.first_ms)} — {fmt_ts(rep.last_ms)} ({span_days:.1f} дн.)",
        f"- Свечей: {rep.rows:,} из ожидаемых {rep.expected:,}; пропущено {rep.missing_total:,} "
        f"({rep.missing_total / max(rep.expected, 1) * 100:.3f}%) в {len(rep.gaps)} разрывах",
        f"- Дубли: {rep.duplicates}; нарушение порядка: {rep.non_monotonic}; "
        f"не кратно минуте: {rep.misaligned}; строки с NaN: {rep.nan_rows}",
        f"- Нарушения OHLC (high/low вне open/close): {rep.ohlc_violations}; "
        f"цены ≤ 0: {rep.non_positive}",
    ]
    if rep.zero_volume is not None:
        lines.append(f"- Свечей с нулевым объёмом: {rep.zero_volume:,} "
                     f"(самая длинная серия {rep.longest_zero_volume_run} мин)")
    lines.append(f"- Выбросы доходности (|r| ≥ {quality.RET_ABS_MIN:.0%} и z ≥ {quality.RET_ROBUST_Z:g}): "
                 f"{0 if rep.return_outliers is None else len(rep.return_outliers)}")
    lines.append(f"- Длинные тени (≥ {quality.WICK_ABS_MIN:.0%}): "
                 f"{0 if rep.wick_outliers is None else len(rep.wick_outliers)}")
    lines.append("")
    if rep.gaps:
        g = sorted(rep.gaps, key=lambda x: -x.missing)
        gdf = pd.DataFrame({"с": [fmt_ts(x.start_ms) for x in g],
                            "по": [fmt_ts(x.end_ms) for x in g],
                            "минут": [x.missing for x in g]})
        lines += ["Крупнейшие разрывы:", "", _table(gdf)]
    if rep.return_outliers is not None and len(rep.return_outliers):
        lines += ["Выбросы доходности:", "", _table(rep.return_outliers)]
    if rep.wick_outliers is not None and len(rep.wick_outliers):
        lines += ["Длинные тени:", "", _table(rep.wick_outliers.sort_values("wick_pct", ascending=False))]
    return "\n".join(lines) + "\n"


def build_report(data_dir: Path, symbols: list[str]) -> str:
    out = [
        "# Отчёт о качестве данных",
        "",
        f"Сформирован: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}. "
        "Все числа посчитаны кодом `bot/data/quality.py` по локальным файлам.",
        "",
    ]
    for sym in symbols:
        inst = store.read_json(store.json_path(data_dir, sym, "instrument"))
        if inst is None:
            out += [f"## {sym}", "", "Данных нет.", ""]
            continue
        i = inst["instrument"]
        lot, pf, lev = i.get("lotSizeFilter", {}), i.get("priceFilter", {}), i.get("leverageFilter", {})
        out += [
            f"## {sym}",
            "",
            f"- Статус: {i.get('status')}; листинг (launchTime): {fmt_ts(int(i.get('launchTime', 0)))}",
            f"- tickSize: {pf.get('tickSize')}; qtyStep: {lot.get('qtyStep')}; "
            f"minOrderQty: {lot.get('minOrderQty')}; minNotionalValue: {lot.get('minNotionalValue')}",
            f"- maxLeverage: {lev.get('maxLeverage')}; интервал финансирования: "
            f"{i.get('fundingInterval')} мин; границы ставки: "
            f"[{i.get('lowerFundingRate')}, {i.get('upperFundingRate')}]",
            "",
        ]
        rl = store.read_json(store.json_path(data_dir, sym, "risk_limit"))
        if rl and rl.get("tiers"):
            t = pd.DataFrame(rl["tiers"])
            cols = [c for c in ("id", "riskLimitValue", "maintenanceMargin", "initialMargin",
                                "maxLeverage", "mmDeduction") if c in t]
            out += ["Уровни лимита риска (первые):", "", _table(t[cols], 5)]

        last = store.read_parquet(store.kline_path(data_dir, sym, store.KIND_LAST), store.KLINE_COLUMNS)
        mark = store.read_parquet(store.kline_path(data_dir, sym, store.KIND_MARK), store.MARK_COLUMNS)
        if len(last):
            out.append(_kline_section(quality.check_klines(last, "Свечи 1m, последняя цена")))
        if len(mark):
            out.append(_kline_section(quality.check_klines(mark, "Свечи 1m, mark-цена")))
        if len(last) and len(mark):
            mc = quality.compare_last_mark(last, mark)
            q = mc.abs_diff_q
            out += [
                "### Последняя цена против mark-цены (close)",
                "",
                f"- Совпавших минут: {mc.joined:,}; только в last: {mc.only_last:,}; только в mark: {mc.only_mark:,}",
                f"- |last/mark − 1|: медиана {q['p50']*100:.4f}%, p99 {q['p99']*100:.3f}%, "
                f"p99.9 {q['p999']*100:.3f}%, максимум {q['max']*100:.2f}%",
                f"- Минут с расхождением ≥ {quality.MARK_DIFF_WARN:.0%}: {mc.over_warn:,}",
                "",
                "Наибольшие расхождения:",
                "",
                _table(mc.worst, 10),
            ]
        fund = store.read_parquet(store.funding_path(data_dir, sym), store.FUNDING_COLUMNS)
        if len(fund):
            lo = float(i["lowerFundingRate"]) if i.get("lowerFundingRate") else None
            hi = float(i["upperFundingRate"]) if i.get("upperFundingRate") else None
            fr = quality.check_funding(fund, lo, hi)
            q = fr.rate_q
            out += [
                "### Финансирование",
                "",
                f"- Записей: {fr.rows:,}; период {fmt_ts(fr.first_ms)} — {fmt_ts(fr.last_ms)}",
                f"- Интервалы между выплатами (ч → кол-во): {fr.intervals_h}",
                f"- Ставка: мин {q['min']*100:.4f}%, p1 {q['p01']*100:.4f}%, медиана {q['p50']*100:.4f}%, "
                f"p99 {q['p99']*100:.4f}%, макс {q['max']*100:.4f}%",
                f"- Вне текущих границ инструмента: {fr.out_of_bounds} "
                "(границы могли меняться со временем)",
                f"- |ставка| ≥ {quality.FUNDING_ABS_WARN:.1%}: {len(fr.extreme)}",
                "",
            ]
            if len(fr.extreme):
                out += [_table(fr.extreme)]
        if len(last):
            out += ["### Первые дни после листинга", "",
                    "Размах дня и оборот относительно медианы всей истории:", "",
                    _table(quality.early_listing_profile(last), 14)]
    return "\n".join(out)
