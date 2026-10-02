"""Разведка, часть C: кандидат «пробой канала 4h» при риске 1–2 % (после протокола, без отложенных данных).

Кандидат появился ПОСЛЕ просмотра результатов (basket_explore.py), поэтому числа здесь
оптимистичны по построению. Чистой проверкой могут быть только отложенные данные.

  1. Какой депозит нужен: доля проверочных сделок, проходящих минимальный ордер 5 USDT,
     при депозите D и риске r (по фактическим стопам сделок).
  2. Реальный счёт 25 / 50 / 100 USDT при риске 1 % и 2 % (те же параметры walk-forward).
  3. Лонги против шортов (смещение выживших завышает лонги).
  4. Издержки × 2.
  5. Дефлированный Шарп с учётом ВСЕХ прогонов walk-forward (протокол, справочный раунд,
     разведка).
  6. Сравнение с buy & hold корзины.

Запуск: python -m research.basket_explore2 → reports/basket_explore2.txt
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from bot.backtest.metrics import path_metrics
from bot.config import PROJECT_ROOT, load_config
from bot.data.bars import to_ms
from bot.data.panel import load_basket_dev
from research import basket_wf as W
from research.basket_gates import basket_bh, bootstrap_p_mean_le0, daily_returns, dsr

FAM, TF = "breakout", "4h"
MIN_NOTIONAL, COST = 5.0, 0.00055 + 0.0002


def load(prefix: str):
    return W.load_pickle(W.OUT_DIR / f"{prefix}_{FAM}_{TF}.pkl")


def daily_sr(wf, start) -> tuple[float, float, float, int, pd.Series]:
    d = daily_returns(wf.equity_ref, start, W.REF_EQUITY)
    r = d.pct_change().dropna()
    sd = r.std(ddof=1)
    return (float(r.mean() / sd) if sd > 0 else 0.0, float(r.skew()), float(r.kurt() + 3), len(r), d)


def main() -> None:
    base = load_config()
    start, end = to_ms(W.QUARTERS[0]), to_ms(W.QUARTERS[-1])
    L = ["Разведка, часть C: пробой канала 4h при риске 1–2 % (кандидат выбран ПОСЛЕ просмотра результатов;",
         "отложенные данные не использовались)", ""]

    # 1. какой депозит нужен
    tr = load("risk01").trades_ref
    dist = (tr["entry_price"] - tr["stop_initial"]).abs() / tr["entry_price"]
    L.append(f"1. Стоп в проверочных сделках: медиана {dist.median() * 100:.1f}%, 90-й перцентиль "
             f"{dist.quantile(0.9) * 100:.1f}%, максимум {dist.max() * 100:.1f}%.")
    L.append("   Доля сделок, проходящих минимальный ордер 5 USDT, при депозите D и риске r:")
    L.append("   D, USDT  " + "  ".join(f"r={r * 100:.0f}%" for r in (0.01, 0.02, 0.05)))
    for D in (10, 25, 50, 100, 200):
        cells = [f"{((D * r) / (dist + 2 * COST) >= MIN_NOTIONAL).mean() * 100:5.0f}%" for r in (0.01, 0.02, 0.05)]
        L.append(f"   {D:7d}  " + "  ".join(cells))
    L.append("")

    # 2. реальный счёт разного размера
    panel = load_basket_dev(base)
    L.append("2. Реальный счёт (параметры и правила walk-forward те же):")
    rows = []
    for rpt in (0.01, 0.02):
        for D in (25.0, 50.0, 100.0):
            cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": rpt,
                                                                                "starting_equity_usdt": D})})
            pool = W.make_pool(panel, cfg)
            try:
                wf = W.walk_forward(FAM, TF, pool)
            finally:
                pool.close()
            e = wf.equity_real["equity"]
            dd = float((1 - e / e.cummax()).max())
            rows.append((rpt, D, len(wf.trades_real), wf.fold_rows[-1]["equity_real_end"], dd, wf.halted_real))
            L.append(f"   риск {rpt * 100:.0f}%, депозит {D:5.0f} USDT: сделок {len(wf.trades_real):4d}, итог "
                     f"{wf.fold_rows[-1]['equity_real_end']:8.2f} USDT (×{wf.fold_rows[-1]['equity_real_end'] / D:.2f}), "
                     f"макс. просадка {dd * 100:.0f}%{', ОСТАНОВЛЕН' if wf.halted_real else ''}")
    L.append("")

    # 3. лонги против шортов, кварталы
    for pre in ("risk01", "risk02"):
        wf = load(pre)
        t = wf.trades_ref
        L.append(f"3. {pre}: лонгов {int((t.side == 1).sum())}, ср. R {t.loc[t.side == 1, 'r_multiple'].mean():+.3f}; "
                 f"шортов {int((t.side == -1).sum())}, ср. R {t.loc[t.side == -1, 'r_multiple'].mean():+.3f}; "
                 f"прибыльных кварталов {(t.groupby('fold')['net_pnl'].sum() > 0).sum()} из 9; "
                 f"P(ср. R ≤ 0) по бутстрэпу {bootstrap_p_mean_le0(t['r_multiple'].to_numpy()):.4f}")
    L.append("")

    # 4. издержки × 2
    costs2 = base.costs.model_copy(update={"taker_fee": base.costs.taker_fee * 2, "maker_fee": base.costs.maker_fee * 2,
                                           "slippage": base.costs.slippage * 2})
    for rpt in (0.01, 0.02):
        cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": rpt})})
        pool = W.make_pool(panel, cfg)
        try:
            wf = W.walk_forward(FAM, TF, pool, costs=costs2)
        finally:
            pool.close()
        st = W.trade_stats(wf.trades_ref)
        L.append(f"4. Издержки ×2 (подбор на обычных, проверка на двойных), риск {rpt * 100:.0f}%: сделок {st['trades']}, "
                 f"PF {st['pf']:.2f}, ср. R {st['mean_r']:+.3f}, справочный 1000 → {wf.fold_rows[-1]['equity_ref_end']:.0f}")
    L.append("")

    # 5. дефлированный Шарп по всем прогонам walk-forward
    files = sorted(glob.glob(str(W.OUT_DIR / "*.pkl")))
    sr_all, mine = [], {}
    for f in files:
        wf = W.load_pickle(f)
        sr, sk, ku, T, d = daily_sr(wf, start)
        sr_all.append(sr)
        name = Path(f).stem
        if name in (f"risk01_{FAM}_{TF}", f"risk02_{FAM}_{TF}"):
            mine[name] = (sr, sk, ku, T, d)
    L.append(f"5. Всего прогонов walk-forward (попыток): {len(sr_all)}")
    bh = basket_bh(start, end, base.costs.taker_fee)
    bh = pd.concat([pd.Series([1.0], index=[bh.index[0] - pd.Timedelta(days=1)]), bh])
    bhm = path_metrics(bh)
    for name, (sr, sk, ku, T, d) in mine.items():
        p, sr0 = dsr(sr, sr_all, T, sk, ku)
        m = path_metrics(d)
        L.append(f"   {name}: Шарп (год) {m['sharpe']:.2f}, Кальмар {m['calmar']:.2f}, макс. просадка "
                 f"{m['max_drawdown'] * 100:.0f}%, доходность {m['total_return'] * 100:+.0f}%; дефлированный Шарп {p:.3f} "
                 f"(порог SR0 дн. {sr0:.4f})")
    L.append(f"6. Buy & hold корзины: Шарп {bhm['sharpe']:.2f}, Кальмар {bhm['calmar']:.2f}, "
             f"макс. просадка {bhm['max_drawdown'] * 100:.0f}%, доходность {bhm['total_return'] * 100:+.0f}%")
    out = PROJECT_ROOT / "reports" / "basket_explore2.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
