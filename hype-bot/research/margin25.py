"""Проверка правила пользователя: маржа 25 % баланса на сделку, плечо максимальное, до 4 позиций.

    python -m research.margin25 → reports/margin25.txt, reports/margin25/equity.png

Та же стратегия (пробой канала 4h, параметры этапа 1), то же правило ликвидности, тот же период
2023-07-01 — конец данных, счёт 25 USDT. Правило несовместимо с риском 2 % на сделку, лимитом суммы
риска 6 % и правилом «ликвидация вдвое дальше стопа» — в прогоне они не действуют.
  N1 — новое правило + остальные ограничители (дневной лимит 6 %, просадка 20 % → маржа 12,5 %, 40 % → остановка);
  N2 — новое правило без ограничителей счёта (что делает само правило);
  B  — текущий вариант (этап 1) для сравнения.
ДАННЫЕ НЕ ЧИСТЫЕ: вся история уже использовалась в исследовании.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig  # noqa: E402
from bot.config import PROJECT_ROOT, RiskCfg, load_config  # noqa: E402
from bot.data.bars import to_ms  # noqa: E402
from bot.data.panel import load_basket_holdout  # noqa: E402
from bot.data.split import HOLDOUT_CONFIRM  # noqa: E402
from bot.risk.portfolio import LiquidityRule, PortfolioRules  # noqa: E402
from bot.runtime import make_strategy  # noqa: E402
from research.basket_report import INK2, NEUTRAL, S, legend, style  # noqa: E402
from research.stage1_risk import START, bot_config, run  # noqa: E402

DEPOSIT = 25.0
OUT = PROJECT_ROOT / "reports" / "margin25"


def run_margin(panel, base, start: int, limits: bool, margin_stop: str = "none", liq_gap: float = 0.035):
    cfg = bot_config(base, 0.02, "A")
    risk = RiskCfg(starting_equity_usdt=DEPOSIT, risk_per_trade=0.02,
                   daily_loss_limit=0.06 if limits else 0.98, max_drawdown=0.40 if limits else 0.99,
                   drawdown_steps=[(0.20, 0.5)] if limits else [], loss_streak_pause_trades=None)
    rules = PortfolioRules(max_positions=4, max_open_risk=10.0, downsize_to_fit=False, corr_cap=None,
                           liquidity=LiquidityRule(), sizing="margin", margin_fraction=0.25,
                           margin_stop=margin_stop, liq_gap=liq_gap)
    pc = PortfolioConfig(risk=risk, costs=cfg.costs, initial_equity=DEPOSIT, max_positions=4,
                         max_open_risk=10.0, trade_start_ms=start, rules=rules)
    return PortfolioBacktester(panel, make_strategy(cfg), pc).run()


def describe(name: str, res, start: int, L: list[str]) -> pd.Series:
    e = res.equity
    e = e[e["ts"] > start]
    eq = pd.concat([pd.Series([DEPOSIT]), e["equity"]], ignore_index=True)
    dd = float((1 - eq / eq.cummax()).max())
    tr = res.trades
    ts = pd.to_datetime(e["ts"], unit="ms", utc=True)
    L.append(f"{name}: 25 → {res.final_equity:.2f} USDT ({(res.final_equity / DEPOSIT - 1) * 100:+.0f}%), "
             f"пик {eq.max():.2f}, макс. просадка {dd * 100:.1f}%, сделок {len(tr)}"
             f"{', ОСТАНОВЛЕН' if res.halted else ''}")
    if len(tr):
        reasons = tr["exit_reason"].value_counts().to_dict()
        lev = tr["leverage"].median()
        L.append(f"   выходы: {reasons}; плечо (медиана) {lev:g}; прибыльных {(tr['net_pnl'] > 0).mean() * 100:.0f}%, "
                 f"средний результат сделки {(tr['net_pnl'] / (tr['equity_after'] - tr['net_pnl'])).mean() * 100:+.1f}% "
                 "капитала")
        liq = tr[tr["exit_reason"] == "liquidation"]
        if len(liq):
            loss = -(liq["net_pnl"] / (liq["equity_after"] - liq["net_pnl"]))
            L.append(f"   ликвидаций {len(liq)} ({len(liq) / len(tr) * 100:.0f}% сделок), потеря на ликвидации — "
                     f"медиана {loss.median() * 100:.1f}% капитала")
    below = eq[eq <= DEPOSIT * 0.6]
    if len(below):
        first = ts.iloc[max(0, below.index[0] - 1)]
        L.append(f"   капитал впервые упал на 40 % и ниже: {first:%Y-%m-%d}")
    halts = [ev for ev in res.events if ev.kind in ("max_drawdown", "halt")]
    if halts:
        L.append(f"   остановка: {pd.Timestamp(halts[0].ts, unit='ms', tz='UTC'):%Y-%m-%d} — {halts[0].detail}")
    s = pd.Series(eq.values[1:], index=ts.values)
    return s.resample("1D").last().ffill()


def main() -> None:
    base = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_basket_holdout(base, HOLDOUT_CONFIRM, "правило пользователя: маржа 25 %, плечо максимальное "
                                                     "(данные уже не чистые)")
    start = to_ms(START)
    L = ["Правило пользователя: маржа 25 % баланса на сделку, плечо максимальное биржевое, до 4 позиций.",
         "Пробой канала 4h (параметры этапа 1), правило ликвидности, счёт 25 USDT, "
         f"{START} — {pd.Timestamp(int(panel.ts[-1]), unit='ms', tz='UTC'):%Y-%m-%d}. ДАННЫЕ НЕ ЧИСТЫЕ.",
         "Не действуют (несовместимы с правилом): риск 2 % на сделку, лимит суммы риска 6 %, "
         "«ликвидация вдвое дальше стопа».", ""]
    curves = {}
    curves["N1: маржа 25 % + ограничители"] = describe("N1 (маржа 25 %, плечо макс., дневной лимит 6 %, просадка "
                                                       "20 % → маржа 12,5 %, 40 % → остановка)",
                                                       run_margin(panel, base, start, True), start, L)
    L.append("")
    curves["N2: маржа 25 % без ограничителей"] = describe("N2 (маржа 25 %, плечо макс., без ограничителей счёта)",
                                                         run_margin(panel, base, start, False), start, L)
    L.append("")
    b = run(panel, base, "B", 0.02, DEPOSIT, start)
    curves["B: текущий (риск 2 %)"] = describe("B (текущий: риск 2 %, сумма ≤ 6 %, корреляция ≤ 4 %)", b.res, start, L)
    L.append("")
    L.append("Устойчивость к дате старта, N2 (без ограничителей счёта), старт 25 USDT с разных дат:")
    for d in ("2023-07-01", "2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01", "2026-01-01"):
        st = to_ms(d)
        r = run_margin(panel, base, st, False)
        e = r.equity[r.equity["ts"] > st]
        ts = pd.to_datetime(e["ts"], unit="ms", utc=True).reset_index(drop=True)
        eq = e["equity"].reset_index(drop=True)
        hit = lambda lvl: next((f"{(ts[i] - pd.Timestamp(d, tz='UTC')).days} дн." for i in range(len(eq))  # noqa: E731
                                if eq[i] <= DEPOSIT * lvl), "не было")
        tr = r.trades
        nl = int((tr["exit_reason"] == "liquidation").sum()) if len(tr) else 0
        L.append(f"   старт {d}: −40 % через {hit(0.6)}, −90 % через {hit(0.1)}; сделок {len(tr)}, ликвидаций {nl}, "
                 f"прибыльных {int((tr['net_pnl'] > 0).sum()) if len(tr) else 0}; итог {r.final_equity:.2f} USDT")
    text = "\n".join(L) + "\n"
    (PROJECT_ROOT / "reports" / "margin25.txt").write_text(text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax, "Счёт 25 USDT: маржа 25 % с максимальным плечом против текущего риска 2 % (данные не чистые)", "USDT")
    for i, (name, s) in enumerate(curves.items()):
        s = s.clip(lower=0.01)
        ax.plot(s.index, s.values, color=S[i], linewidth=2, label=name)
        ax.annotate(f"{s.iloc[-1]:.2f}", (s.index[-1], s.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.axhline(DEPOSIT, color=NEUTRAL, linewidth=1)
    ax.set_yscale("log")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=150)
    plt.close(fig)
    print(text)
    _ = np


if __name__ == "__main__":
    main()
