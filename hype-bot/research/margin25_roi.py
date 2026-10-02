"""Правило пользователя в его формулировке: «DOGE, плечо 75, стоп на −40 % от позиции, ликвидация от −45 %».

    python -m research.margin25_roi → reports/margin25_roi.txt, reports/margin25/equity_roi.png

То есть: маржа 25 % баланса на сделку, плечо максимальное, до 4 позиций, стоп на 4 пункта ROI (доли маржи)
раньше ликвидации. Ликвидация — по формуле Bybit (сверена с биржей: тест test_bybit_liquidation_formula_matches_
exchange); DOGE 75x: ликвидация −44,1 % ROI, стоп −40,1 %. У монет с другим плечом и ставкой поддерживающей
маржи стоп ставится по тому же правилу «ликвидация минус 4 пункта». Если стоп стратегии ближе — остаётся он.

Варианты записаны до прогона, параметры по результату не подбираются:
  C1 — правило + ограничители счёта (дневной лимит 6 %, просадка 20 % → маржа 12,5 %, 40 % → остановка);
  C2 — правило без ограничителей счёта;
  зазор 3 и 3,5 пункта — чувствительность (без ограничителей);
  обе модели исполнения: базовая (стоп, исполненный за ценой ликвидации, — ликвидация) и оптимистичная
  (стоп всегда раньше ликвидации, если свеча не открылась за ней);
  старт с шести дат — без ограничителей, обе модели;
  для сравнения — текущий вариант B этапа 1 (риск 2 %).
Та же стратегия (пробой канала 4h), правило ликвидности, период, счёт 25 USDT. ДАННЫЕ НЕ ЧИСТЫЕ.
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bot.config import PROJECT_ROOT, load_config  # noqa: E402
from bot.data.bars import to_ms  # noqa: E402
from bot.data.panel import load_basket_holdout  # noqa: E402
from bot.data.split import HOLDOUT_CONFIRM  # noqa: E402
from bot.risk.sizing import LONG, liquidation_price_bybit  # noqa: E402
from research.basket_report import INK2, NEUTRAL, S, legend, style  # noqa: E402
from research.margin25 import DEPOSIT, OUT, describe, run_margin  # noqa: E402
from research.margin25_stop import STARTS, details  # noqa: E402
from research.stage1_risk import START, run  # noqa: E402

GAP = 0.04


def run_c(panel, base, start: int, limits: bool, optimistic: bool, gap: float = GAP):
    return run_margin(panel, base, start, limits, "roi", stop_beats_liq=optimistic, roi_gap=gap,
                      liq_formula="bybit")


def robustness(panel, base, optimistic: bool, L: list[str]) -> None:
    for d in STARTS:
        st = to_ms(d)
        r = run_c(panel, base, st, False, optimistic)
        e = r.equity[r.equity["ts"] > st]
        ts = pd.to_datetime(e["ts"], unit="ms", utc=True).reset_index(drop=True)
        eq = e["equity"].reset_index(drop=True)
        hit = lambda lvl: next((f"{(ts[i] - pd.Timestamp(d, tz='UTC')).days} дн." for i in range(len(eq))  # noqa: E731
                                if eq[i] <= DEPOSIT * lvl), "не было")
        tr = r.trades
        nl = int((tr["exit_reason"] == "liquidation").sum()) if len(tr) else 0
        L.append(f"   старт {d}: −40 % через {hit(0.6)}, −90 % через {hit(0.1)}; сделок {len(tr)}, ликвидаций {nl}, "
                 f"прибыльных {int((tr['net_pnl'] > 0).sum()) if len(tr) else 0}; пик {max(DEPOSIT, eq.max()):.2f}; "
                 f"итог {r.final_equity:.2f} USDT")


def main() -> None:
    base = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_basket_holdout(base, HOLDOUT_CONFIRM, "правило пользователя: маржа 25 %, стоп на 4 п. ROI до "
                                                     "ликвидации (данные уже не чистые)")
    start = to_ms(START)
    L = ["Правило пользователя: маржа 25 % баланса, плечо максимальное, до 4 позиций, стоп на 4 пункта ROI раньше "
         "ликвидации (DOGE 75x: стоп −40 %, ликвидация −44 %).",
         "Пробой канала 4h (параметры этапа 1), правило ликвидности, счёт 25 USDT, "
         f"{START} — {pd.Timestamp(int(panel.ts[-1]), unit='ms', tz='UTC'):%Y-%m-%d}. ДАННЫЕ НЕ ЧИСТЫЕ.",
         "Ликвидация — по формуле Bybit (сверена с биржей). Стоп — рыночный, проскальзывание 0,02 %, "
         "проникновение 25 % хода свечи за стоп.", ""]
    L.append("Монеты корзины при максимальном плече (первый уровень риска): ликвидация и стоп в % от маржи (ROI) и "
             "в % цены:")
    for sym in panel.symbols:
        inst = panel.sym[sym].inst
        t = inst.tier_for(0.0)
        lev = inst.max_leverage
        liq = 1 - liquidation_price_bybit(LONG, 1.0, lev, t.mmr)
        L.append(f"   {sym.removesuffix('USDT'):9s} {lev:>4g}x  ликвидация −{liq * lev * 100:.1f}% ROI "
                 f"({liq * 100:.3f}% цены), стоп −{(liq * lev - GAP) * 100:.1f}% ROI ({(liq - GAP / lev) * 100:.3f}% "
                 f"цены), между ними {GAP / lev * 100:.3f}% цены")
    L.append("")
    curves = {}
    for optimistic in (False, True):
        tag = "оптимистичная модель" if optimistic else "базовая модель"
        L.append(f"=== {tag} исполнения ===")
        for limits in (True, False):
            name = ("C1" if limits else "C2") + ("-опт" if optimistic else "")
            label = (f"{name} (с ограничителями: дневной лимит 6 %, просадка 20 % → маржа 12,5 %, 40 % → остановка)"
                     if limits else f"{name} (без ограничителей счёта)")
            res = run_c(panel, base, start, limits, optimistic)
            s = describe(label, res, start, L)
            details(res, L, start)
            tr = res.trades
            if len(tr):
                dg = tr[tr["symbol"] == "DOGEUSDT"]
                if len(dg):
                    L.append(f"   из них DOGE: {len(dg)} сделок, прибыльных {int((dg['net_pnl'] > 0).sum())}, выходы "
                             f"{dg['exit_reason'].value_counts().to_dict()}")
            if not limits:
                curves[f"C2: без ограничителей, {tag}"] = s
            L.append("")
        L.append(f"Чувствительность к зазору ({tag}, без ограничителей):")
        for g in (0.03, 0.035):
            r = run_c(panel, base, start, False, optimistic, g)
            tr = r.trades
            nl = int((tr["exit_reason"] == "liquidation").sum()) if len(tr) else 0
            L.append(f"   зазор {g * 100:g} п.: 25 → {r.final_equity:.2f} USDT, сделок {len(tr)}, ликвидаций {nl}, "
                     f"прибыльных {int((tr['net_pnl'] > 0).sum()) if len(tr) else 0}"
                     f"{', ОСТАНОВЛЕН (просадка 99 %)' if r.halted else ''}")
        L.append(f"Устойчивость к дате старта ({tag}, без ограничителей):")
        robustness(panel, base, optimistic, L)
        L.append("")
    b = run(panel, base, "B", 0.02, DEPOSIT, start)
    curves["Текущий: риск 2 %"] = describe("Текущий вариант B этапа 1 (риск 2 %, сумма ≤ 6 %, корреляция ≤ 4 %)",
                                           b.res, start, L)
    details(b.res, L, start)
    text = "\n".join(L) + "\n"
    (PROJECT_ROOT / "reports" / "margin25_roi.txt").write_text(text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax, "Счёт 25 USDT: стоп на 4 п. ROI до ликвидации против риска 2 % (данные не чистые)", "USDT")
    placed: list[float] = []
    for i, (name, s) in enumerate(curves.items()):
        s = s.clip(lower=0.01)
        ax.plot(s.index, s.values, color=S[i], linewidth=2, label=name)
        y = float(s.iloc[-1])
        dy = sum((9 if y > q else -9) for q in placed if abs(math.log10(y / q)) < 0.08)   # близкие подписи — врозь
        placed.append(y)
        ax.annotate(f"{y:.2f}", (s.index[-1], y), xytext=(4, dy), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.axhline(DEPOSIT, color=NEUTRAL, linewidth=1)
    ax.set_yscale("log")
    legend(ax, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "equity_roi.png", dpi=150)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
