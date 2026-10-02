"""Разведка после протокола (НЕ отбор финалиста, отложенные данные не используются).

Протокол не дал финалиста. Чтобы предложить пользователю варианты, здесь измеряется
то, что протокол не варьировал:
  A. Риск на сделку 1 / 2 / 3 % вместо 5 % для трендовых семейств 4h — те же правила
     walk-forward, меняется только размер позиций (выбор параметров практически не меняется:
     SQN не зависит от размера). Плюс оценка Келли по проверочным сделкам.
  B. Дневные трендовые стратегии без ограничения ширины стопа 9,5 % — то есть при депозите,
     который позволяет широкие стопы (реальный прогон со 100 USDT).
Все прогоны — дополнительные попытки; они учитываются в N для дефлированного Шарпа.

Запуск: python -m research.basket_explore → reports/basket_explore.txt
"""
from __future__ import annotations

import pickle
import time

import numpy as np

import bot.strategy.basket as SB
from bot.config import PROJECT_ROOT, load_config
from bot.data.panel import load_basket_dev
from research import basket_wf as W

TREND_4H = [("breakout", "4h", 200), ("oi_breakout", "4h", 200), ("tsmom", "4h", 60)]
DAILY = [("tsmom", "1d", 40), ("breakout", "1d", 40)]


def kelly(r: np.ndarray) -> float:
    """Доля капитала f, максимизирующая средний log(1 + f·R) (сделки по очереди, без наложения)."""
    fs = np.linspace(0.001, 0.5, 500)
    with np.errstate(invalid="ignore", divide="ignore"):
        g = np.array([np.mean(np.log1p(f * r)) if (1 + f * r.min()) > 0 else -np.inf for f in fs])
    return float(fs[int(np.argmax(g))])


def run(prefix: str, combos, cfg, panel) -> list[str]:
    out = []
    pool = W.make_pool(panel, cfg)
    try:
        for fam, tf, mt in combos:
            t0 = time.time()
            wf = W.walk_forward(fam, tf, pool, min_trades=mt)
            with open(W.OUT_DIR / f"{prefix}_{fam}_{tf}.pkl", "wb") as fh:
                pickle.dump(wf, fh)
            st = W.trade_stats(wf.trades_ref)
            eq = wf.equity_real["equity"]
            peak_dd = float((1 - eq / eq.cummax()).max()) if len(eq) else 0.0
            out.append(f"{prefix:8s} {fam:12s} {tf}: сделок {st['trades']:4d}, ср. R {st['mean_r']:+.3f}, PF {st['pf']:.2f}; "
                       f"справочный 1000 → {wf.fold_rows[-1]['equity_ref_end']:.0f}; "
                       f"реальный {cfg.risk.starting_equity_usdt:g} → {wf.fold_rows[-1]['equity_real_end']:.2f} "
                       f"(пик {eq.max():.2f}, макс. просадка {peak_dd * 100:.0f}%"
                       f"{', ОСТАНОВЛЕН' if wf.halted_real else ''}) [{time.time() - t0:.0f} с]")
            print(out[-1], flush=True)
    finally:
        pool.close()
    return out


def main() -> None:
    base = load_config()
    panel = load_basket_dev(base)
    lines = ["Разведка после протокола (не отбор, без отложенных данных). Период — проверочные кварталы "
             "2024Q1–2026Q1, правила walk-forward те же.", ""]
    lines.append("A. Риск на сделку для трендовых семейств 4h (остальные ограничители без изменений)")
    for rpt in (0.01, 0.02, 0.03, 0.05):
        cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": rpt})})
        lines += run(f"risk{int(rpt * 100):02d}", TREND_4H, cfg, panel)
    lines.append("")
    lines.append("Келли по проверочным сделкам (справочный капитал, риск 5 %, сделки по очереди — "
                 "наложение позиций не учтено, поэтому оценка завышена):")
    for fam, tf, _ in TREND_4H:
        wf = W.load_pickle(W.OUT_DIR / f"risk05_{fam}_{tf}.pkl")
        r = wf.trades_ref["r_multiple"].to_numpy()
        if len(r):
            k = kelly(r)
            lines.append(f"  {fam:12s} {tf}: f* = {k * 100:.1f}% капитала на сделку, половина Келли {k * 50:.1f}%")
    lines.append("")
    lines.append("B. Дневные трендовые стратегии без ограничения ширины стопа (депозит 100 USDT, риск 5 %)")
    SB.MAX_STOP_FRAC = 1.0
    cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"starting_equity_usdt": 100.0})})
    lines += run("wide1d", DAILY, cfg, panel)
    out = PROJECT_ROOT / "reports" / "basket_explore.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
