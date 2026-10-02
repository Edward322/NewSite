"""Финальная проверка кандидата на отложенных данных (протокол, раздел 8).

    python -m research.basket_holdout --dry-run   # пробный прогон на последних 6 мес. разработки
    python -m research.basket_holdout --final     # ОДИН раз: отложенные данные 2026-04-02 — 2026-10-02

Пробный прогон проверяет код на данных разработки (2025-10-02 — 2026-04-01), отложенные данные
не загружаются. Финальный прогон открывает их через load_basket_holdout (запись в
reports/holdout_access.log).

Кандидат: пробой канала 4h; подбор параметров раз в квартал на 12 месяцах перед кварталом
(SQN, плато, ≥ 200 сделок), риск 2 %, счёт 25 USDT; критерии — на счёте 25 USDT.
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.metrics import path_metrics  # noqa: E402
from bot.backtest.portfolio import PortfolioBacktester, PortfolioConfig  # noqa: E402
from bot.config import PROJECT_ROOT, load_config  # noqa: E402
from bot.data.bars import to_ms  # noqa: E402
from bot.data.panel import load_basket_dev, load_basket_holdout  # noqa: E402
from bot.data.split import HOLDOUT_CONFIRM  # noqa: E402
from bot.strategy.base import LONG, SHORT  # noqa: E402
from bot.strategy.basket import FAMILIES, Breakout, make  # noqa: E402
from research import basket_wf as W  # noqa: E402
from research.basket_gates import basket_bh, daily_returns  # noqa: E402
from research.basket_report import INK2, NEUTRAL, S, legend, style  # noqa: E402

FAM, TF = "breakout", "4h"
RISK, DEPOSIT = 0.02, 25.0
N_RANDOM, N_MC = 200, 10_000


def windows(final: bool, cfg) -> list[tuple[int, int, int]]:
    if final:
        tests = [("2026-04-02", "2026-07-02"), ("2026-07-02", cfg.research.holdout_end)]
    else:
        tests = [("2025-10-02", "2026-01-02"), ("2026-01-02", "2026-04-01")]
    out = []
    for a, b in tests:
        ts = pd.Timestamp(a, tz="UTC")
        out.append((to_ms(ts - pd.DateOffset(months=12)), to_ms(ts), to_ms(b)))
    return out


class RandomEntry(Breakout):
    """Те же выходы (канал n/2, трейлинг, стоп k_stop × ATR), но входы случайные:
    на каждой свече монета входит с вероятностью p_enter в случайную сторону."""
    p_enter, seed = 0.0, 0

    def prepare(self, bars) -> None:
        super().prepare(bars)
        rng = np.random.default_rng(self.seed)
        self.rand = rng.random((self.n, len(bars)))
        self.rside = np.where(rng.random((self.n, len(bars))) < 0.5, LONG, SHORT)

    def on_bar(self, k, positions):
        out = [d for d in super().on_bar(k, positions) if not hasattr(d[1], "side")]   # только выходы
        entries = []
        for r in range(self.n):
            if r in positions or self.rand[r, k] >= self.p_enter:
                continue
            side = self.rside[r, k] if self.p["sides"] == "both" else LONG
            e = self._enter(r, k, side, float(self.rand[r, k]))
            if e:
                entries.append(e)
        return self._ordered(out, entries)


def run_custom(panel, cfg, strategy, start, end, equity, costs=None, state=None):
    w0 = max(int(panel.ts[0]), start - W.WARMUP_DAYS * W.DAY_MS)
    pc = PortfolioConfig(risk=cfg.risk, costs=costs or cfg.costs, initial_equity=equity,
                         max_positions=W.MAX_POSITIONS, max_open_risk=W.MAX_OPEN_RISK,
                         trade_start_ms=start, trade_end_ms=end, initial_risk_state=state)
    return PortfolioBacktester(panel.slice(w0, end), strategy, pc).run()


def chain(panel, cfg, plan, equity, costs=None, strat_fn=None):
    """Цепочка проверочных кварталов с переносом капитала и состояния ограничителей."""
    trades, eqs, skips, decisions, state, halted = [], [], [], [], None, False
    for (a, b, c), params in plan:
        if params is None:
            eqs.append(pd.DataFrame({"ts": [c], "equity": [equity], "cash": [equity], "position": [0]}))
            continue
        strat = strat_fn(params) if strat_fn else make(FAM, TF, params)
        res = run_custom(panel, cfg, strat, b, c, equity, costs, state)
        equity, state, halted = res.final_equity, res.risk_state, res.halted
        if len(res.trades):
            trades.append(res.trades)
        e = res.equity
        eqs.append(e[(e["ts"] > b) & (e["ts"] <= c)])
        skips += [s for s in res.skips if b <= s[0] < c]
        decisions += [d for d in res.decisions if b <= d[0] < c]
    tr = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return tr, pd.concat(eqs, ignore_index=True), skips, decisions, equity, halted


def pf(tr) -> float:
    if not len(tr):
        return float("nan")
    p = tr["net_pnl"]
    loss = -p[p < 0].sum()
    return float(p[p > 0].sum() / loss) if loss > 0 else float("inf")


def neighbours(params: dict) -> list[dict]:
    grid = FAMILIES[FAM].grid
    out = []
    for k, vals in grid.items():
        i = vals.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                out.append({**params, k: vals[j]})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    final = args.final
    tag = "holdout" if final else "holdout_dryrun"
    base = load_config()
    cfg = base.model_copy(update={"risk": base.risk.model_copy(update={"risk_per_trade": RISK,
                                                                        "starting_equity_usdt": DEPOSIT})})
    if final:
        panel = load_basket_holdout(cfg, HOLDOUT_CONFIRM,
                                    "basket: пробой канала 4h, 25 USDT, риск 2 %, протокол раздел 8")
    else:
        panel = load_basket_dev(cfg)
    wins = windows(final, cfg)
    start, end = wins[0][1], wins[-1][2]
    L = [f"{'ФИНАЛЬНАЯ ПРОВЕРКА НА ОТЛОЖЕННЫХ ДАННЫХ' if final else 'ПРОБНЫЙ ПРОГОН (данные разработки)'}: "
         f"пробой канала 4h, риск {RISK * 100:.0f}%, счёт {DEPOSIT:g} USDT",
         f"Период: {pd.Timestamp(start, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} — "
         f"{pd.Timestamp(end, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC; последняя 15m-свеча панели "
         f"{pd.Timestamp(int(panel.ts[-1]), unit='ms', tz='UTC'):%Y-%m-%d %H:%M}", ""]

    # 1. подбор параметров по кварталам
    pool = W.make_pool(panel, cfg)
    plan = []
    try:
        for a, b, c in wins:
            gdf = W.evaluate_grid(FAM, TF, a, b, pool)
            params, score = W.select(gdf, FAMILIES[FAM].grid)
            plan.append(((a, b, c), params))
            L.append(f"Квартал с {pd.Timestamp(b, unit='ms', tz='UTC'):%Y-%m-%d}: подбор на "
                     f"{pd.Timestamp(a, unit='ms', tz='UTC'):%Y-%m-%d} — {pd.Timestamp(b, unit='ms', tz='UTC'):%Y-%m-%d} → "
                     f"{W.fmt_params(params)} (плато SQN {score:.2f}; допустимых конфигураций "
                     f"{int((gdf['trades'] >= W.MIN_TRADES).sum())} из {len(gdf)})")
    finally:
        pool.close()
    L.append("")

    # 2. основной прогон: счёт 25 USDT и справочный 1000 USDT
    tr, eq, skips, decs, end_eq, halted = chain(panel, cfg, plan, DEPOSIT)
    tr_ref, eq_ref, _, _, end_ref, halted_ref = chain(panel, cfg, plan, W.REF_EQUITY)
    d_real = daily_returns(eq, start, DEPOSIT) if len(eq) else pd.Series(dtype=float)
    m = path_metrics(d_real)
    e = eq["equity"]
    max_dd = float((1 - e / e.cummax()).max()) if len(e) else 0.0
    max_dd = max(max_dd, m["max_drawdown"])
    bh = basket_bh(start, end, cfg.costs.taker_fee, panel)
    bh = pd.concat([pd.Series([1.0], index=[bh.index[0] - pd.Timedelta(days=1)]), bh])
    bhm = path_metrics(bh)
    pf_real = pf(tr)
    small = sum(1 for s in skips if "ниже минимума" in s[2])
    crit = {
        "1. PF > 1,3": (pf_real > 1.3, f"PF {pf_real:.3f}"),
        "2. ≥ 100 сделок": (len(tr) >= 100, f"сделок {len(tr)}"),
        "3. не остановлен, просадка < 80 %": (not halted and max_dd < 0.8,
                                             f"{'остановлен' if halted else 'работает'}, макс. просадка {max_dd * 100:.1f}%"),
        "4. Шарп и Кальмар > buy & hold": (m["sharpe"] > bhm["sharpe"] and m["calmar"] > bhm["calmar"],
                                          f"Шарп {m['sharpe']:.2f} против {bhm['sharpe']:.2f}, "
                                          f"Кальмар {m['calmar']:.2f} против {bhm['calmar']:.2f}"),
    }
    passed = all(v[0] for v in crit.values())
    L.append(f"Счёт {DEPOSIT:g} USDT → {end_eq:.2f} USDT ({(end_eq / DEPOSIT - 1) * 100:+.1f}%), "
             f"сделок {len(tr)}, сигналов пропущено из-за мин. ордера {small}")
    if len(tr):
        L.append(f"  средний R {tr['r_multiple'].mean():+.3f}, доля прибыльных {(tr['net_pnl'] > 0).mean() * 100:.0f}%, "
                 f"лонгов {int((tr.side == 1).sum())} (ср. R {tr.loc[tr.side == 1, 'r_multiple'].mean():+.3f}), "
                 f"шортов {int((tr.side == -1).sum())} (ср. R {tr.loc[tr.side == -1, 'r_multiple'].mean():+.3f})")
        L.append(f"  комиссии {(tr['entry_fee'] + tr['exit_fee']).sum():.2f} USDT, финансирование {tr['funding'].sum():+.2f} USDT; "
                 f"выходы: {tr['exit_reason'].value_counts().to_dict()}")
        for q, ((_, b, c), _) in enumerate(plan):
            t = tr[(tr["entry_ts"] >= b) & (tr["entry_ts"] < c)]
            L.append(f"  квартал {q + 1}: сделок {len(t)}, PF {pf(t):.2f}, итог {t['net_pnl'].sum():+.2f} USDT")
    L.append(f"Buy & hold корзины: доходность {bhm['total_return'] * 100:+.1f}%, Шарп {bhm['sharpe']:.2f}, "
             f"Кальмар {bhm['calmar']:.2f}, макс. просадка {bhm['max_drawdown'] * 100:.1f}%")
    L.append(f"Справочный капитал 1000 → {end_ref:.0f} ({(end_ref / 1000 - 1) * 100:+.1f}%), сделок {len(tr_ref)}, "
             f"PF {pf(tr_ref):.3f}{', остановлен' if halted_ref else ''}")
    L.append("")
    L.append("КРИТЕРИИ ПРИЁМКИ (счёт 25 USDT):")
    for k, (ok, txt) in crit.items():
        L.append(f"  {'ВЫПОЛНЕН ' if ok else 'НЕ ВЫПОЛНЕН'}  {k}: {txt}")
    L.append(f"ИТОГ: {'ПРОЙДЕНО' if passed else 'НЕ ПРОЙДЕНО'}")
    L.append("")

    # 3. для информации: устойчивость
    L.append("Для информации (не критерии):")
    costs2 = cfg.costs.model_copy(update={"taker_fee": cfg.costs.taker_fee * 2, "maker_fee": cfg.costs.maker_fee * 2,
                                          "slippage": cfg.costs.slippage * 2})
    tr2, _, _, _, end2, _ = chain(panel, cfg, plan, DEPOSIT, costs=costs2)
    L.append(f"  издержки × 2: счёт 25 → {end2:.2f} USDT, PF {pf(tr2):.3f}, сделок {len(tr2)}")
    nb_rows = []
    for (a, b, c), params in plan:
        if params is None:
            continue
        for nb in neighbours(params):
            r = run_custom(panel, cfg, make(FAM, TF, nb), b, c, W.REF_EQUITY)
            nb_rows.append({"q": b, **nb, "trades": len(r.trades), "pf": pf(r.trades),
                            "mean_r": r.trades["r_multiple"].mean() if len(r.trades) else np.nan})
    nbd = pd.DataFrame(nb_rows)
    if len(nbd):
        L.append(f"  соседние параметры ({len(nbd)} прогонов по кварталам, справочный капитал): PF медиана "
                 f"{nbd['pf'].median():.2f}, доля с PF > 1 — {(nbd['pf'] > 1).mean() * 100:.0f}%, "
                 f"средний R медиана {nbd['mean_r'].median():+.3f}")
    if len(tr) > 1:
        rng = np.random.default_rng(11)
        R = tr["r_multiple"].to_numpy()
        fin, dds = np.empty(N_MC), np.empty(N_MC)
        for i in range(N_MC):
            path = np.cumprod(1 + RISK * rng.choice(R, size=len(R), replace=True))
            path = np.r_[1.0, path]
            fin[i], dds[i] = path[-1], (1 - path / np.maximum.accumulate(path)).max()
        L.append(f"  бутстрэп сделок ({N_MC}): итог счёта 25 USDT медиана {np.median(fin) * DEPOSIT:.1f}, "
                 f"5-й перцентиль {np.percentile(fin, 5) * DEPOSIT:.1f}, P(убыток) {(fin < 1).mean() * 100:.0f}%, "
                 f"просадка медиана {np.median(dds) * 100:.0f}%, 95-й перцентиль {np.percentile(dds, 95) * 100:.0f}%")
    # случайные входы с теми же выходами
    cells = sum(int((~np.isnan(panel.close[:len(panel.symbols), int(np.searchsorted(panel.ts, b)):
                                          int(np.searchsorted(panel.ts, c))])).sum()) for (_, b, c), _ in plan) / 16
    n_sig = sum(1 for d in decs if d[2].startswith("Enter"))
    p_enter = n_sig / cells if cells else 0.0
    rnd = []
    for seed in range(N_RANDOM):
        def fn(params, seed=seed):
            s = RandomEntry(timeframe=TF, **params)
            s.p_enter, s.seed = p_enter, seed
            return s
        t_r, _, _, _, e_r, _ = chain(panel, cfg, plan, W.REF_EQUITY, strat_fn=fn)
        rnd.append((t_r["r_multiple"].mean() if len(t_r) else np.nan, pf(t_r), len(t_r), e_r))
    rd = pd.DataFrame(rnd, columns=["mean_r", "pf", "n", "end"])
    if len(tr_ref):
        pct = float((rd["mean_r"] < tr_ref["r_multiple"].mean()).mean() * 100)
        L.append(f"  случайные входы с теми же выходами ({N_RANDOM} прогонов, вероятность входа {p_enter:.4f} на свечу): "
                 f"средний R медиана {rd['mean_r'].median():+.3f}, PF медиана {rd['pf'].median():.2f}, сделок медиана "
                 f"{rd['n'].median():.0f}; стратегия лучше {pct:.0f}% случайных")
    text = "\n".join(L) + "\n"
    (PROJECT_ROOT / "reports" / f"basket_{tag}.txt").write_text(text, encoding="utf-8")
    with open(W.OUT_DIR / f"{tag}.pkl", "wb") as fh:
        pickle.dump({"plan": plan, "trades": tr, "equity": eq, "trades_ref": tr_ref, "equity_ref": eq_ref,
                     "bh": bh, "neighbours": nbd, "random": rd, "criteria": crit, "passed": passed}, fh)
    # график: счёт 25 USDT против buy & hold корзины на те же 25 USDT
    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax, f"{'Отложенные данные' if final else 'Пробный прогон'}: счёт {DEPOSIT:g} USDT, пробой канала 4h, "
              f"риск {RISK * 100:.0f} %", "USDT")
    if len(d_real):
        ax.plot(d_real.index, d_real.values, color=S[0], linewidth=2, label="бот")
        ax.annotate(f"{d_real.iloc[-1]:.2f}", (d_real.index[-1], d_real.iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.plot(bh.index, bh.values * DEPOSIT, color=NEUTRAL, linewidth=2, linestyle="--", label="buy & hold корзины")
    ax.annotate(f"{bh.iloc[-1] * DEPOSIT:.2f}", (bh.index[-1], bh.iloc[-1] * DEPOSIT), xytext=(4, 0),
                textcoords="offset points", color=INK2, fontsize=8, va="center")
    ax.axhline(DEPOSIT, color=NEUTRAL, linewidth=1)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))
    legend(ax)
    fig.tight_layout()
    fig.savefig(W.OUT_DIR / f"{tag}_equity.png", dpi=150)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
