"""Разбор сохранённых результатов финальной проверки (reports/basket/holdout.pkl).

Отложенные данные повторно не загружаются — читаются только уже посчитанные сделки.
Вывод: reports/basket_holdout_notes.txt
"""
from __future__ import annotations

import pickle

import pandas as pd

from bot.config import PROJECT_ROOT
from research import basket_wf as W


def main() -> None:
    with open(W.OUT_DIR / "holdout.pkl", "rb") as fh:
        d = pickle.load(fh)
    tr = d["trades"].copy()
    tr["t"] = pd.to_datetime(tr["entry_ts"], unit="ms", utc=True)
    total = tr["net_pnl"].sum()
    top = tr.nlargest(5, "net_pnl")
    L = ["Разбор финальной проверки (счёт 25 USDT, риск 2 %)", "",
         f"Итог всех {len(tr)} сделок: {total:+.2f} USDT; сумма R {tr['r_multiple'].sum():+.1f}", "",
         "Пять самых прибыльных сделок:"]
    for t in top.itertuples():
        L.append(f"  {t.symbol:13s} вход {t.t:%Y-%m-%d %H:%M}, {'лонг' if t.side == 1 else 'шорт'}, "
                 f"{t.net_pnl:+.2f} USDT ({t.r_multiple:+.1f} R), держали {t.bars_held} свечей 4h")
    for k, name in ((1, "лучшей сделки"), (2, "двух лучших сделок"), (3, "трёх лучших сделок")):
        L.append(f"Без {name} сумма результатов: {total - tr['net_pnl'].nlargest(k).sum():+.2f} USDT")
    e = d["equity"].copy()
    e["t"] = pd.to_datetime(e["ts"], unit="ms", utc=True)
    dd = 1 - e["equity"] / e["equity"].cummax()
    i = int(dd.idxmax())
    j = int(e["equity"].iloc[: i + 1].idxmax())
    L += ["", f"Максимальная просадка {dd.iloc[i] * 100:.1f}%: с {e['t'].iloc[j]:%Y-%m-%d} "
              f"({e['equity'].iloc[j]:.2f} USDT) до {e['t'].iloc[i]:%Y-%m-%d} ({e['equity'].iloc[i]:.2f} USDT)", "",
          "По месяцам (по дате входа):"]
    m = tr.groupby(tr["t"].dt.strftime("%Y-%m")).agg(n=("net_pnl", "size"), net=("net_pnl", "sum"),
                                                     r=("r_multiple", "sum"))
    for mon, r in m.iterrows():
        L.append(f"  {mon}: сделок {int(r.n):3d}, {r.net:+6.2f} USDT, {r.r:+6.1f} R")
    bh = d["bh"]
    L += ["", f"Buy & hold корзины: {(bh.iloc[-1] - 1) * 100:+.1f}%; на 25 USDT → {bh.iloc[-1] * 25:.2f} USDT"]
    out = PROJECT_ROOT / "reports" / "basket_holdout_notes.txt"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
