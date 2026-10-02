"""Этап 3: сводка, проверка устойчивости лучшего кандидата и графики.

Запуск: python -m research.stage3_report > reports/stage3_report.txt
Использует результаты research.stage3_walkforward (r1, r2, slow). Отложенные
данные не загружаются: ни один кандидат не прошёл отбор на периоде разработки.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.metrics import _daily, path_metrics  # noqa: E402
from bot.config import CostsCfg, PROJECT_ROOT, load_config  # noqa: E402
from bot.data.bars import load_funding  # noqa: E402
from bot.data.split import dev_range, load_dev  # noqa: E402
from bot.market import Instrument  # noqa: E402
from research import wf  # noqa: E402

OUT = PROJECT_ROOT / "reports" / "stage3"
LABEL = {"r1": "раунд 1", "r2": "раунд 2", "slow": "медленные, без требования частоты"}
# Палитра (dataviz reference, светлая тема): поверхность, текст, сетка, серии 1–3.
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, NEUTRAL = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"


def load_all() -> pd.DataFrame:
    rows = []
    for tag in ("r1", "r2", "slow"):
        s = pd.read_csv(OUT / f"{tag}_summary.csv")
        rows.append(s.assign(round=tag))
    return pd.concat(rows, ignore_index=True)


def oos_chain(family, tf, fold_params, equity, costs, ctx):
    """Повтор OOS-цепочки walk-forward с заданными издержками и теми же параметрами по окнам."""
    trades, state = [], None
    for (_, b, c), params in fold_params:
        if params is None:
            continue
        r = wf.run_window(family, tf, params, b, c, equity, costs=costs, state=state, ctx=ctx)
        equity, state = r.final_equity, r.risk_state
        if len(r.trades):
            trades.append(r.trades)
    t = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()
    return t, equity


def monte_carlo(r: np.ndarray, risk: float, n: int = 10_000, seed: int = 7) -> dict:
    """Перестановка порядка сделок: капитал ×(1 + risk × R) по каждой сделке."""
    rng = np.random.default_rng(seed)
    dd, final = np.empty(n), np.empty(n)
    for k in range(n):
        path = np.cumprod(1 + risk * rng.permutation(r))
        path = np.r_[1.0, path]
        dd[k] = 1 - (path / np.maximum.accumulate(path)).min()
        final[k] = path[-1]
    boot = np.array([np.prod(1 + risk * rng.choice(r, len(r))) for _ in range(n)])
    return {"perm_dd_median": float(np.median(dd)), "perm_dd_p95": float(np.quantile(dd, 0.95)),
            "perm_p_dd_ge_80": float((dd >= 0.8).mean()), "perm_final": float(final[0]),
            "boot_final_p05": float(np.quantile(boot, 0.05)), "boot_final_median": float(np.median(boot)),
            "boot_p_loss": float((boot < 1).mean())}


def ru(v: float, nd: int = 2) -> str:
    return f"{v:.{nd}f}".replace(".", ",")


def style(ax, x_dates: bool = False):
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter
    if x_dates:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    else:
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: ru(v, 1)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ru(v, 1)))
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart_pf(df: pd.DataFrame, path):
    d = df[df["ref_trades"] > 0].copy()
    d["name"] = d["family"] + " " + d["tf"] + np.where(d["round"] == "slow", " (медл.)", "")
    d = d.sort_values("ref_pf")
    freq = d["ref_trades_per_6m"] >= 100
    fig, ax = plt.subplots(figsize=(8, 7.5), facecolor=SURF)
    style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(d))
    ax.barh(y[freq.to_numpy()], d.loc[freq, "ref_pf"], color=S1, height=0.7, label="≥ 100 сделок за 6 мес")
    ax.barh(y[~freq.to_numpy()], d.loc[~freq, "ref_pf"], color=S2, height=0.7, label="< 100 сделок за 6 мес")
    ax.axvline(1.0, color=INK2, linewidth=1, linestyle=":")
    ax.axvline(1.3, color=INK, linewidth=1.2, linestyle="--")
    ax.text(1.3, len(d) - 0.3, " порог\n приёмки 1,3", color=INK, fontsize=9, va="bottom")
    ax.text(1.0, len(d) - 0.3, "1,0 ", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.set_yticks(y, d["name"], fontsize=9, color=INK)
    for yi, v in zip(y, d["ref_pf"]):
        ax.text(v + 0.01, yi, ru(v), va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 1.6)
    ax.set_xlabel("Profit factor вне выборки (walk-forward, 276 дней)", color=INK2)
    ax.set_title("Ни один кандидат не дошёл до порога 1,3", color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, loc="lower right", fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def chart_equity(df, highlight, bh_path, which, start_value, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor=SURF)
    style(ax, x_dates=True)
    for _, row in df.iterrows():
        f = OUT / f"{row['round']}_equity_{which}_{row['family']}_{row['tf']}.parquet"
        if not f.exists():
            continue
        e = pd.read_parquet(f)
        t = pd.to_datetime(e["ts"], unit="ms", utc=True)
        ax.plot(t, e["equity"] / start_value, color=GRID, linewidth=0.9, zorder=1)
    ends = []
    for (rnd, fam, tf, lab), col in zip(highlight, (S1, S2, S3)):
        e = pd.read_parquet(OUT / f"{rnd}_equity_{which}_{fam}_{tf}.parquet")
        t = pd.to_datetime(e["ts"], unit="ms", utc=True)
        y = e["equity"] / start_value
        ax.plot(t, y, color=col, linewidth=2, label=lab, zorder=3)
        ends.append([float(y.iloc[-1]), t.iloc[-1], col])
    # подписи значений на концах линий без наложений
    ends.sort(key=lambda x: x[0])
    pos = []
    for v, t_end, col in ends:
        yy = max(v, pos[-1] + 0.05) if pos else v
        pos.append(yy)
        ax.annotate(f"{ru(v)}", (t_end, v), xytext=(6, (yy - v) * 220), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    bt, bv = bh_path
    ax.plot(bt, bv, color=NEUTRAL, linewidth=1.5, linestyle="--", label="Buy & hold HYPE (цена)", zorder=2)
    ax.axhline(1.0, color=INK2, linewidth=0.8, linestyle=":")
    ax.set_ylabel(ylabel, color=INK2)
    ax.set_title(title, color=INK, loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURF)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    d0, d1 = dev_range(cfg)
    md = load_dev(cfg)
    f_ts, f_r = load_funding(cfg.data_dir(), cfg.symbol)
    ctx = wf.Ctx(md, f_ts, f_r, Instrument.from_files(cfg.data_dir(), cfg.symbol), cfg, d0)
    df = load_all()
    meta = json.loads((OUT / "r1_meta.json").read_text())
    oos_start, oos_end = meta["oos_start"], meta["oos_end"]
    numbers: dict = {"buy_hold_oos": meta["buy_hold"]}

    cols = ["round", "family", "tf", "folds_traded", "ref_trades", "ref_trades_per_6m", "ref_pf", "ref_mean_r",
            "ref_sqn", "ref_win_rate", "real_total_return", "real_max_drawdown", "real_sharpe", "real_sortino",
            "real_trades"]
    pd.set_option("display.width", 220)
    print("=== Сводка walk-forward (вне выборки, 276 дней)")
    print(df[cols].round(3).to_string(index=False))
    passing = df[(df["ref_pf"] > 1.3) & (df["ref_trades_per_6m"] >= 100)]
    print(f"\nПрошли отбор (PF > 1,3 и ≥ 100 сделок за 6 мес вне выборки): {len(passing)}")
    numbers["passing"] = int(len(passing))
    numbers["configs_tried"] = {t: json.loads((OUT / f"{t}_meta.json").read_text())["n_configs"]
                                for t in ("r1", "r2", "slow", "r1dec", "r2dec")}

    # --- устойчивость лучшего «почти» кандидата: импульс на медленных таймфреймах
    folds = pd.read_csv(OUT / "slow_folds.csv")
    fl = wf.folds(d0, d1)
    c = cfg.costs
    double = CostsCfg(maker_fee=2 * c.maker_fee, taker_fee=2 * c.taker_fee, slippage=2 * c.slippage,
                      stop_penetration=min(1.0, 2 * c.stop_penetration))
    for tf in ("1h", "4h"):
        ff = folds[(folds["family"] == "momentum") & (folds["tf"] == tf)].sort_values("fold")
        fold_params = [(fl[int(r.fold)], json.loads(r.params) if isinstance(r.params, str) else None)
                       for r in ff.itertuples()]
        print(f"\n=== Импульс {tf} (справочный раунд без требования частоты)")
        for r in ff.itertuples():
            p = json.loads(r.params) if isinstance(r.params, str) else None
            print(f"окно {r.fold}: параметры {p}; сделок {getattr(r, 'oos_trades', float('nan'))}, "
                  f"PF {getattr(r, 'oos_pf', float('nan')):.2f}, ср.R {getattr(r, 'oos_mean_r', float('nan')):+.3f}")
        out = {}
        for lab, costs in (("обычные", cfg.costs), ("удвоенные", double)):
            t, e_end = oos_chain("momentum", tf, fold_params, wf.REF_EQUITY, costs, ctx)
            st = wf.trade_stats(t)
            out[lab] = {"trades": st["trades"], "pf": st["pf"], "mean_r": st["mean_r"],
                        "return": e_end / wf.REF_EQUITY - 1}
            print(f"издержки {lab}: сделок {st['trades']}, PF {st['pf']:.2f}, ср.R {st['mean_r']:+.3f}, "
                  f"доходность {e_end / wf.REF_EQUITY - 1:+.1%}")
            if lab == "обычные":
                mc = monte_carlo(t["r_multiple"].to_numpy(), cfg.risk.risk_per_trade)
                out["monte_carlo"] = mc
                print(f"Монте-Карло (10 000 перестановок, риск 5%): медианная просадка {mc['perm_dd_median']:.0%}, "
                      f"95-й перцентиль {mc['perm_dd_p95']:.0%}, P(просадка ≥ 80%) {mc['perm_p_dd_ge_80']:.1%}; "
                      f"бутстрэп итога: 5-й перцентиль ×{mc['boot_final_p05']:.2f}, медиана ×{mc['boot_final_median']:.2f}, "
                      f"P(убыток) {mc['boot_p_loss']:.0%}")
        numbers[f"momentum_{tf}_slow"] = out

    # --- графики
    sel = (md.ts >= oos_start) & (md.ts < oos_end)
    bh_t = pd.to_datetime(md.ts[sel][::60], unit="ms", utc=True)
    bh_v = md.close[sel][::60] / md.close[sel][0]
    best = df[df["ref_trades"] > 0].sort_values("ref_pf", ascending=False).head(3)
    hl = [(r["round"], r["family"], r["tf"],
           f"{r['family']} {r['tf']}{' (медл.)' if r['round'] == 'slow' else ''}: PF {ru(r['ref_pf'])}")
          for _, r in best.iterrows()]
    numbers["top3"] = [h[3] for h in hl]
    chart_pf(df, OUT / "pf_oos.png")
    chart_equity(df, hl, (bh_t, bh_v), "ref", wf.REF_EQUITY,
                 "Капитал вне выборки: 3 лучших кандидата против buy & hold",
                 "Капитал, доля от начального", OUT / "equity_oos_ref.png")
    chart_equity(df, hl, (bh_t, bh_v), "real", cfg.risk.starting_equity_usdt,
                 "То же на реальном депозите 10 USDT (минимальный ордер, лимиты риска)",
                 "Капитал, доля от 10 USDT", OUT / "equity_oos_real.png")
    (OUT / "report_numbers.json").write_text(json.dumps(numbers, indent=2, ensure_ascii=False, default=float))
    print("\nГрафики: reports/stage3/pf_oos.png, equity_oos_ref.png, equity_oos_real.png")


if __name__ == "__main__":
    main()
