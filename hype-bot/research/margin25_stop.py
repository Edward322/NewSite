"""Правило пользователя «маржа 25 %, плечо максимальное» + «стоп около ликвидации, за 3–4 % от неё».

    python -m research.margin25_stop → reports/margin25_stop.txt, reports/margin25/equity_stop.png

«За 3–4 % от ликвидации» читается двумя способами — проверяются оба, варианты записаны до прогона:
  A — плечо максимальное, стоп за 3,5 % РАССТОЯНИЯ до ликвидации до неё (≈ −96,5 % маржи; при 75x стоп
      примерно в 1,2–1,3 % цены от входа); если стоп стратегии ближе — остаётся он;
  B — стоп стратегии (2,5 ATR), плечо снижено так, чтобы ликвидация была на 3,5 % ЦЕНЫ дальше стопа;
      маржа по-прежнему 25 % баланса.
Каждый — с ограничителями счёта (дневной лимит 6 %, просадка 20 % → маржа 12,5 %, 40 % → остановка) и без
них; чувствительность 3 % / 4 % и устойчивость к дате старта — без ограничителей (что делает само правило).
Для сравнения — текущий вариант B этапа 1 (риск 2 %). Параметры не подбираются по результату.
Та же стратегия, правило ликвидности, период, счёт 25 USDT. ДАННЫЕ НЕ ЧИСТЫЕ.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from bot.config import PROJECT_ROOT, load_config  # noqa: E402
from bot.data.bars import to_ms  # noqa: E402
from bot.data.panel import load_basket_holdout  # noqa: E402
from bot.data.split import HOLDOUT_CONFIRM  # noqa: E402
from research.basket_report import INK2, NEUTRAL, S, legend, style  # noqa: E402
from research.margin25 import DEPOSIT, OUT, describe, run_margin  # noqa: E402
from research.stage1_risk import START, run  # noqa: E402

GAP = 0.035
READINGS = {
    "near_liq": "A: плечо макс., стоп за 3,5 % расстояния до ликвидации",
    "gap": "B: стоп стратегии, ликвидация на 3,5 % цены дальше",
}
STARTS = ("2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01")


def details(res, L: list[str]) -> None:
    tr = res.trades
    if not len(tr):
        return
    dist = (tr["entry_price"] - tr["stop_initial"]).abs() / tr["entry_price"]
    eq_before = tr["equity_after"] - tr["net_pnl"]
    lost = tr[tr["net_pnl"] < 0]
    L.append(f"   стоп от входа (медиана) {dist.median() * 100:.2f}% цены; маржа (медиана) "
             f"{(tr['margin'] / eq_before).median() * 100:.1f}% капитала; позиция (медиана) "
             f"{(tr['margin'] * tr['leverage'] / eq_before).median():.1f}× капитала")
    if len(lost):
        L.append(f"   убыточная сделка (медиана) {(-lost['net_pnl'] / eq_before[lost.index]).median() * 100:.1f}% "
                 f"капитала; комиссии и проскальзывание за сделку (медиана) "
                 f"{(tr['margin'] * tr['leverage'] * 2 * 0.00075 / eq_before).median() * 100:.2f}% капитала")


def robustness(panel, base, rule: str, L: list[str]) -> None:
    for d in STARTS:
        st = to_ms(d)
        r = run_margin(panel, base, st, False, rule, GAP)
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
    panel = load_basket_holdout(base, HOLDOUT_CONFIRM, "правило пользователя: маржа 25 %, стоп у ликвидации "
                                                     "(данные уже не чистые)")
    start = to_ms(START)
    L = ["Правило пользователя: маржа 25 % баланса на сделку, до 4 позиций, стоп «около ликвидации, за 3–4 % от неё».",
         "Пробой канала 4h (параметры этапа 1), правило ликвидности, счёт 25 USDT, "
         f"{START} — {pd.Timestamp(int(panel.ts[-1]), unit='ms', tz='UTC'):%Y-%m-%d}. ДАННЫЕ НЕ ЧИСТЫЕ.",
         "Не действуют (несовместимы с правилом): риск 2 % на сделку, лимит суммы риска 6 %, "
         "«ликвидация вдвое дальше стопа».",
         "Ликвидация в модели — по последней цене (на бирже — по маркировочной), стоп — рыночный с проскальзыванием "
         "0,02 % и проникновением 25 % хода свечи за стоп; если цена исполнения стопа за ликвидацией — ликвидация.", ""]
    curves = {}
    for rule, title in READINGS.items():
        L.append(f"=== {title} ===")
        for limits in (True, False):
            name = ("A" if rule == "near_liq" else "B") + ("1" if limits else "2")
            label = (f"{name} (с ограничителями: дневной лимит 6 %, просадка 20 % → маржа 12,5 %, 40 % → остановка)"
                     if limits else f"{name} (без ограничителей счёта)")
            res = run_margin(panel, base, start, limits, rule, GAP)
            s = describe(label, res, start, L)
            details(res, L)
            if not limits:
                curves[f"{title.split(':')[0]}: без ограничителей"] = s
            L.append("")
        L.append("Чувствительность к зазору, без ограничителей:")
        for g in (0.03, 0.04):
            r = run_margin(panel, base, start, False, rule, g)
            tr = r.trades
            nl = int((tr["exit_reason"] == "liquidation").sum()) if len(tr) else 0
            L.append(f"   зазор {g * 100:.0f}%: 25 → {r.final_equity:.2f} USDT, сделок {len(tr)}, ликвидаций {nl}, "
                     f"прибыльных {int((tr['net_pnl'] > 0).sum()) if len(tr) else 0}"
                     f"{', ОСТАНОВЛЕН (просадка 99 %)' if r.halted else ''}")
        L.append("Устойчивость к дате старта, без ограничителей, старт 25 USDT с разных дат:")
        robustness(panel, base, rule, L)
        L.append("")
    b = run(panel, base, "B", 0.02, DEPOSIT, start)
    curves["Текущий: риск 2 %"] = describe("Текущий вариант B этапа 1 (риск 2 %, сумма ≤ 6 %, корреляция ≤ 4 %)",
                                           b.res, start, L)
    details(b.res, L)
    text = "\n".join(L) + "\n"
    (PROJECT_ROOT / "reports" / "margin25_stop.txt").write_text(text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax, "Счёт 25 USDT: маржа 25 % со стопом у ликвидации против риска 2 % (данные не чистые)", "USDT")
    for i, (name, s) in enumerate(curves.items()):
        s = s.clip(lower=0.01)
        ax.plot(s.index, s.values, color=S[i], linewidth=2, label=name)
        ax.annotate(f"{s.iloc[-1]:.2f}", (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.axhline(DEPOSIT, color=NEUTRAL, linewidth=1)
    ax.set_yscale("log")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "equity_stop.png", dpi=150)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
