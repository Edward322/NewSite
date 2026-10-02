"""Как размер депозита влияет на кандидата «пробой канала 4h» → reports/basket_deposit.txt.

Берутся параметры по кварталам из walk-forward при риске 2 % (reports/basket/risk02_breakout_4h.pkl)
и та же цепочка кварталов прогоняется с депозитом 10 / 25 / 50 / 100 USDT. Смотрим:
сколько сигналов пропущено из-за минимального ордера, какой риск фактически
получился у сделок (объём округляется вниз до шага биржи), итог и просадку.
Только период разработки; отложенные данные не используются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.config import PROJECT_ROOT, load_config
from bot.data.panel import load_basket_dev
from research import basket_wf as W

FAM, TF = "breakout", "4h"


def chain(panel, cfg, fold_rows, equity: float):
    trades, skips, eqs, state = [], [], [], None
    for r in fold_rows:
        if r["params"] is None:
            continue
        res = W.run_window(FAM, TF, r["params"], r["test_start"], r["test_end"], equity, state=state,
                           panel=panel, cfg=cfg)
        equity, state = res.final_equity, res.risk_state
        if len(res.trades):
            trades.append(res.trades)
        skips += [s for s in res.skips if r["test_start"] <= s[0] < r["test_end"]]
        e = res.equity
        eqs.append(e[(e["ts"] > r["test_start"]) & (e["ts"] <= r["test_end"])])
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return tr, skips, pd.concat(eqs, ignore_index=True), equity, state


def main() -> None:
    base = load_config()
    panel = load_basket_dev(base)
    rows = W.load_pickle(W.OUT_DIR / f"risk02_{FAM}_{TF}.pkl").fold_rows
    L = ["Депозит и кандидат «пробой канала 4h» (параметры по кварталам из walk-forward при риске 2 %),",
         "проверочные кварталы 2024Q1–2026Q1, период разработки. Риск — запланированный убыток сделки",
         "в % от капитала перед сделкой (с издержками); объём округляется вниз до шага биржи.", ""]
    for rpt in (0.02, 0.01):
        cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": rpt})})
        for dep in (10.0, 25.0, 50.0, 100.0):
            tr, sk, eq, end, st = chain(panel, cfg, rows, dep)
            small = sum(1 for s in sk if "ниже минимума" in s[2])
            if not len(tr):
                L.append(f"риск {rpt * 100:.0f}%, депозит {dep:5.0f}: сделок нет — все {small} сигналов меньше "
                         "минимального ордера")
                continue
            before = tr["equity_after"] - tr["net_pnl"]
            eff = tr["planned_loss"] / before * 100
            e = eq["equity"]
            dd = float((1 - e / e.cummax()).max()) * 100
            first = tr.iloc[:30]
            eff_first = (first["planned_loss"] / (first["equity_after"] - first["net_pnl"]) * 100).median()
            reach = eq[eq["equity"] >= 2 * dep]
            t2 = (pd.Timestamp(int(reach["ts"].iloc[0]), unit="ms", tz="UTC").strftime("%Y-%m")
                  if len(reach) else "не достигнут")
            L.append(f"риск {rpt * 100:.0f}%, депозит {dep:5.0f}: сделок {len(tr):4d}, пропущено из-за мин. ордера {small:4d}; "
                     f"фактический риск: медиана {eff.median():.2f}% (первые 30 сделок {eff_first:.2f}%), "
                     f"10-й перцентиль {np.percentile(eff, 10):.2f}%; итог {end:8.2f} (×{end / dep:.2f}), "
                     f"макс. просадка {dd:.0f}%, удвоение: {t2}{', ОСТАНОВЛЕН' if st and st.halted else ''}")
        L.append("")
    # почему 25 USDT отстают от 50 USDT при риске 2 %: какие сделки маленький счёт не смог взять
    cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": 0.02})})
    t25, t50 = chain(panel, cfg, rows, 25.0)[0], chain(panel, cfg, rows, 50.0)[0]
    k = ["symbol", "decision_ts"]
    m = t50.merge(t25[k], on=k, how="left", indicator=True)
    miss = m[m["_merge"] == "left_only"].copy()
    x = t25.merge(t50[k], on=k, how="left", indicator=True)
    extra = x[x["_merge"] == "left_only"]
    miss["q"] = pd.to_datetime(miss["decision_ts"], unit="ms", utc=True).dt.strftime("%Y") + "Q" + \
        pd.to_datetime(miss["decision_ts"], unit="ms", utc=True).dt.quarter.astype(str)
    L.append(f"Риск 2 %: сделок счёта 50 USDT, которых нет у счёта 25 USDT — {len(miss)}, сумма {miss['r_multiple'].sum():+.1f} R "
             f"(всего у 50 USDT {t50['r_multiple'].sum():+.1f} R); вместо них у 25 USDT {len(extra)} других сделок, "
             f"сумма {extra['r_multiple'].sum():+.1f} R.")
    by_q = miss.groupby("q")["r_multiple"].agg(["count", "sum"])
    L.append("По кварталам: " + "; ".join(f"{q}: {int(r['count'])} сделок, {r['sum']:+.1f} R" for q, r in by_q.iterrows()))
    top = miss.sort_values("r_multiple").tail(3)
    L.append("Крупнейшие пропущенные: " + ", ".join(
        f"{t.symbol} {pd.Timestamp(int(t.decision_ts), unit='ms', tz='UTC'):%Y-%m-%d} {t.r_multiple:+.1f} R"
        for t in top.itertuples()))
    out = PROJECT_ROOT / "reports" / "basket_deposit.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
