"""Критерии отбора финалиста (протокол, раздел 5) по результатам walk-forward.

Читает reports/basket/wf_*.pkl (или другой префикс), считает на склеенных проверочных
окнах (справочный капитал): число сделок, PF, бутстрэп среднего R, прибыльные кварталы,
дефлированный Шарп (Bailey & López de Prado, 2014) по всем комбинациям, сравнение с
buy & hold корзины, итог прогона с 10 USDT.

Запуск: python -m research.basket_gates [префикс]   (по умолчанию wf)
"""
from __future__ import annotations

import math
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

from bot.backtest.metrics import _daily, path_metrics
from bot.config import PROJECT_ROOT, load_config
from bot.data.panel import aggregate_panel, load_basket_dev
from research.basket_wf import COMBOS, OUT_DIR, QUARTERS, REF_EQUITY, WF, trade_stats  # noqa: F401
from bot.data.bars import to_ms

N01 = NormalDist()
EULER = 0.5772156649
GATE_TRADES, GATE_PF, GATE_P, GATE_FOLDS, GATE_DSR = 400, 1.3, 0.05, 5, 0.95


def bootstrap_p_mean_le0(r: np.ndarray, n_boot: int = 20000, seed: int = 0) -> float:
    if len(r) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = r[rng.integers(0, len(r), (n_boot, len(r)))].mean(axis=1)
    return float((means <= 0).mean())


def daily_returns(eq: pd.DataFrame, start_ms: int, start_value: float) -> pd.Series:
    d = _daily(eq["ts"].to_numpy(), eq["equity"].to_numpy(), start_value, start_ms)
    return d


def dsr(sr: float, sr_all: list[float], t: int, skew: float, kurt: float) -> tuple[float, float]:
    """Дефлированный Шарп: P(истинный SR > SR0), SR0 — ожидаемый максимум из N попыток."""
    n = len(sr_all)
    v = float(np.var(sr_all, ddof=1)) if n > 1 else 0.0
    sr0 = math.sqrt(v) * ((1 - EULER) * N01.inv_cdf(1 - 1 / n) + EULER * N01.inv_cdf(1 - 1 / (n * math.e))) \
        if n > 1 else 0.0
    den = 1 - skew * sr + (kurt - 1) / 4 * sr * sr
    if den <= 0 or t < 2:
        return float("nan"), sr0
    return N01.cdf((sr - sr0) * math.sqrt(t - 1) / math.sqrt(den)), sr0


def basket_bh(start_ms: int, end_ms: int, taker: float) -> pd.Series:
    """Buy & hold корзины: равные доли в доступных монетах, ежедневная ребалансировка, плечо 1,
    финансирование и комиссия на оборот ребалансировки. Возвращает дневной капитал (старт 1)."""
    cfg = load_config()
    panel = load_basket_dev(cfg)
    b = aggregate_panel(panel, "1d")
    n = b.n_trade
    sel = (b.ts >= start_ms) & (b.close_ts <= end_ms)
    c = b.close[:n][:, sel]
    o = b.open[:n][:, sel]
    days = b.ts[sel]
    fund = np.zeros_like(c)
    for r, s in enumerate(panel.symbols):
        sd = panel.sym[s]
        idx = np.searchsorted(days, sd.funding_ts, "right") - 1
        ok = (idx >= 0) & (sd.funding_ts < days[-1] + 86_400_000) & (sd.funding_ts >= days[0])
        np.add.at(fund[r], idx[ok], sd.funding_rate[ok])
    ret = c / o - 1 - fund                     # день: открытие → закрытие, минус финансирование лонга
    eq, vals, w_prev = 1.0, [], None
    for d in range(c.shape[1]):
        avail = ~np.isnan(ret[:, d])
        w = np.where(avail, 1.0 / avail.sum(), 0.0)
        turnover = np.abs(w - (w_prev if w_prev is not None else 0)).sum()
        eq *= 1 - taker * turnover
        r = np.nansum(w * np.nan_to_num(ret[:, d]))
        eq *= 1 + r
        grown = w * (1 + np.nan_to_num(ret[:, d]))
        w_prev = grown / grown.sum() if grown.sum() > 0 else w
        vals.append(eq)
    return pd.Series(vals, index=pd.to_datetime(days + 86_400_000 - 1, unit="ms", utc=True).floor("D"))


def load_wf(prefix: str, family: str, tf: str) -> WF | None:
    p = OUT_DIR / f"{prefix}_{family}_{tf}.pkl"
    if not p.exists():
        return None
    from research.basket_wf import load_pickle
    return load_pickle(p)


def evaluate(prefix: str = "wf") -> tuple[pd.DataFrame, dict]:
    cfg = load_config()
    start, end = to_ms(QUARTERS[0]), to_ms(QUARTERS[-1])
    rows, daily = [], {}
    for fam, tf in COMBOS:
        wf = load_wf(prefix, fam, tf)
        if wf is None:
            continue
        st = trade_stats(wf.trades_ref)
        d = daily_returns(wf.equity_ref, start, REF_EQUITY)
        daily[(fam, tf)] = d
        m = path_metrics(d)
        r = d.pct_change().dropna()
        fold_pnl = wf.trades_ref.groupby("fold")["net_pnl"].sum() if len(wf.trades_ref) else pd.Series(dtype=float)
        real_end = wf.fold_rows[-1]["equity_real_end"]
        rows.append({
            "семейство": fam, "тф": tf, "сделок": st["trades"], "PF": st["pf"], "ср. R": st["mean_r"],
            "P(ср.R≤0)": bootstrap_p_mean_le0(wf.trades_ref["r_multiple"].to_numpy()) if st["trades"] else np.nan,
            "приб. кварталов": int((fold_pnl > 0).sum()),
            "кварталов с торговлей": int(sum(1 for x in wf.fold_rows if x["params"] is not None)),
            "доходность": m["total_return"], "Шарп": m["sharpe"], "Кальмар": m["calmar"], "макс. просадка": m["max_drawdown"],
            "sr_d": float(r.mean() / r.std(ddof=1)) if len(r) > 2 and r.std(ddof=1) > 0 else 0.0,
            "skew": float(r.skew()) if len(r) > 2 else 0.0, "kurt": float(r.kurt() + 3) if len(r) > 3 else 3.0,
            "T": len(r), "10 USDT итог": real_end, "10 USDT остановлен": wf.halted_real,
        })
    df = pd.DataFrame(rows)
    sr_all = df["sr_d"].tolist()
    out = [dsr(r.sr_d, sr_all, r.T, r.skew, r.kurt) for r in df.itertuples()]
    df["DSR"] = [o[0] for o in out]
    df["SR0 (дн.)"] = [o[1] for o in out]
    bh = basket_bh(start, end, cfg.costs.taker_fee)
    bh_full = pd.concat([pd.Series([1.0], index=[bh.index[0] - pd.Timedelta(days=1)]), bh])
    bhm = path_metrics(bh_full)
    df["> B&H"] = (df["Шарп"] > bhm["sharpe"]) & (df["Кальмар"] > bhm["calmar"])
    g = pd.DataFrame({
        "1 сделок≥400": df["сделок"] >= GATE_TRADES,
        "2 PF≥1.3": df["PF"] >= GATE_PF,
        "3 бутстрэп": df["P(ср.R≤0)"] < GATE_P,
        "4 кварталы≥5": df["приб. кварталов"] >= GATE_FOLDS,
        "5 DSR≥0.95": df["DSR"] >= GATE_DSR,
        "6 >B&H": df["> B&H"],
        "7 10 USDT": (~df["10 USDT остановлен"]) & (df["10 USDT итог"] > cfg.risk.starting_equity_usdt),
    })
    df["прошла"] = g.all(axis=1)
    return pd.concat([df, g], axis=1), {"bh": bhm, "bh_series": bh_full, "daily": daily}


def fmt(df: pd.DataFrame) -> str:
    cols = ["семейство", "тф", "кварталов с торговлей", "сделок", "PF", "ср. R", "P(ср.R≤0)", "приб. кварталов",
            "доходность", "Шарп", "Кальмар", "макс. просадка", "DSR", "10 USDT итог", "10 USDT остановлен", "прошла"]
    d = df[cols].copy()
    for c in ("PF", "ср. R", "P(ср.R≤0)", "Шарп", "Кальмар", "DSR"):
        d[c] = d[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    for c in ("доходность", "макс. просадка"):
        d[c] = d[c].map(lambda x: f"{x * 100:+.1f}%" if c == "доходность" else f"{x * 100:.1f}%")
    d["10 USDT итог"] = d["10 USDT итог"].map(lambda x: f"{x:.2f}")
    return d.to_string(index=False)


def main(argv: list[str]) -> None:
    prefix = argv[0] if argv else "wf"
    df, extra = evaluate(prefix)
    bhm = extra["bh"]
    gates = ["1 сделок≥400", "2 PF≥1.3", "3 бутстрэп", "4 кварталы≥5", "5 DSR≥0.95", "6 >B&H", "7 10 USDT"]
    text = [f"Критерии отбора (протокол, раздел 5), результаты walk-forward «{prefix}», "
            f"проверочные кварталы {QUARTERS[0]} — {QUARTERS[-1]}", "",
            fmt(df), "",
            "Выполнение критериев (True — выполнен):", df[["семейство", "тф"] + gates].to_string(index=False), "",
            f"Buy & hold корзины за тот же период: доходность {bhm['total_return'] * 100:+.1f}%, "
            f"Шарп {bhm['sharpe']:.3f}, Кальмар {bhm['calmar']:.3f}, макс. просадка {bhm['max_drawdown'] * 100:.1f}%",
            f"Порог дефлированного Шарпа SR0 (дневной) = {df['SR0 (дн.)'].iloc[0]:.4f} при N = {len(df)} попытках", "",
            "Прошли все критерии: " + (", ".join(f"{r.семейство} {r.тф}" for r in df[df["прошла"]].itertuples()) or "нет")]
    out = PROJECT_ROOT / "reports" / f"basket_gates_{prefix}.txt"
    out.write_text("\n".join(text) + "\n", encoding="utf-8")
    df.to_csv(OUT_DIR / f"gates_{prefix}.csv", index=False)
    print("\n".join(text))


if __name__ == "__main__":
    main(sys.argv[1:])
