"""Графики для отчёта по корзине → reports/basket/*.png.

Использует результаты basket_wf (wf, ref60), basket_diag и basket_explore.
Отложенные данные не загружаются.
"""
from __future__ import annotations


import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.data.bars import to_ms  # noqa: E402
from research import basket_wf as W  # noqa: E402
from research.basket_gates import basket_bh, daily_returns  # noqa: E402

OUT = W.OUT_DIR
# Палитра dataviz (светлая тема, проверена validate_palette.js): поверхность, текст, сетка, серии 1–4.
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#8a8984"
NAMES = {"tsmom": "импульс по монете", "breakout": "пробой канала", "xsmom": "межмонетный импульс",
         "carry": "против перегретых (фин./базис)", "oi_breakout": "пробой + рост OI", "crowd": "против толпы"}


def style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


def legend(ax, loc="upper left") -> None:
    ax.legend(frameon=True, facecolor=SURF, edgecolor="none", framealpha=0.9, fontsize=8, labelcolor=INK2, loc=loc)


def x_dates(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m.%y"))


def load(prefix, fam, tf):
    return W.load_pickle(OUT / f"{prefix}_{fam}_{tf}.pkl")


def chart_equity(prefix: str, combos: list[tuple[str, str]], fname: str, title: str) -> None:
    start, end = to_ms(W.QUARTERS[0]), to_ms(W.QUARTERS[-1])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style(ax, title, "капитал, справочный старт = 1 (лог. шкала)")
    for i, (fam, tf) in enumerate(combos):
        d = daily_returns(load(prefix, fam, tf).equity_ref, start, W.REF_EQUITY) / W.REF_EQUITY
        ax.plot(d.index, d.values, color=S[i], linewidth=2, label=f"{NAMES[fam]}, {tf}")
        ax.annotate(f"{d.iloc[-1]:.2f}", (d.index[-1], d.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    bh = basket_bh(start, end, 0.00055)
    ax.plot(bh.index, bh.values, color=NEUTRAL, linewidth=2, linestyle="--", label="buy & hold корзины")
    ax.annotate(f"{bh.iloc[-1]:.2f}", (bh.index[-1], bh.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                color=INK2, fontsize=8, va="center")
    log_y(ax)
    x_dates(ax)
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def chart_real(fname: str) -> None:
    """Путь реального счёта 10 USDT у трендовых 4h (протокол): рост и обвал до остановки."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    style(ax, "Счёт 10 USDT при риске 5 %: проверочные кварталы, параметры из walk-forward",
          "капитал, USDT (лог. шкала)")
    for i, (pre, fam) in enumerate((("wf", "breakout"), ("wf", "oi_breakout"), ("wf", "carry"))):
        e = load(pre, fam, "4h").equity_real
        t = pd.to_datetime(e["ts"], unit="ms", utc=True)
        ax.plot(t, e["equity"], color=S[i], linewidth=2, label=f"{NAMES[fam]}, 4h")
        j = int(e["equity"].idxmax())
        ax.annotate(f"пик {e['equity'].max():.0f}", (t[j], e["equity"].iloc[j]), xytext=(0, 6),
                    textcoords="offset points", color=INK2, fontsize=8, ha="center")
    ax.axhline(10, color=NEUTRAL, linewidth=1, linestyle="--")
    log_y(ax)
    x_dates(ax)
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def chart_diag(fname: str) -> None:
    df = pd.read_csv(OUT / "diag_grid.csv")
    df = df[df["trades"] > 0]
    order = [(f, tf) for f, tf in W.COMBOS]
    rows = []
    for f, tf in order:
        g = df[(df["family"] == f) & (df["tf"] == tf)]
        rows.append((f"{NAMES[f]}, {tf}", g["mean_r"].median(), (g["mean_r"] > 0).mean(), g["trades"].median()))
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 5))
    style(ax, "Средний R сделки по всей сетке, без подбора (2024Q1–2026Q1)", "")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    ax.barh(y, vals, color=[S[0] if v > 0 else S[1] for v in vals], height=0.6)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, [r[0] for r in rows], color=INK2, fontsize=9)
    for yi, r in zip(y, rows):
        ax.annotate(f"{r[1]:+.2f} R · {r[2] * 100:.0f}% конфигураций > 0 · ~{r[3]:.0f} сделок",
                    (max(r[1], 0), yi), xytext=(6, 0), textcoords="offset points", va="center",
                    color=INK2, fontsize=8)
    ax.set_xlabel("медиана среднего R по конфигурациям (после издержек)", color=INK2, fontsize=9)
    ax.set_xlim(min(vals) - 0.1, max(vals) + 2.2)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def chart_risk(fname: str) -> None:
    """Пробой канала 4h при разном риске на сделку (справочный капитал) против buy & hold."""
    start, end = to_ms(W.QUARTERS[0]), to_ms(W.QUARTERS[-1])
    fig, ax = plt.subplots(figsize=(9, 4.8))
    style(ax, "Пробой канала 4h: риск на сделку 1 %, 2 % и 5 % (разведка, после протокола)",
          "капитал, старт = 1 (лог. шкала)")
    for i, rk in enumerate((1, 2, 5)):
        d = daily_returns(load(f"risk{rk:02d}", "breakout", "4h").equity_ref, start, W.REF_EQUITY) / W.REF_EQUITY
        ax.plot(d.index, d.values, color=S[i], linewidth=2, label=f"риск {rk} %")
        ax.annotate(f"{d.iloc[-1]:.2f}", (d.index[-1], d.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    bh = basket_bh(start, end, 0.00055)
    ax.plot(bh.index, bh.values, color=NEUTRAL, linewidth=2, linestyle="--", label="buy & hold корзины")
    ax.annotate(f"{bh.iloc[-1]:.2f}", (bh.index[-1], bh.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                color=INK2, fontsize=8, va="center")
    log_y(ax)
    x_dates(ax)
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)


def main() -> None:
    chart_equity("wf", [("breakout", "4h"), ("oi_breakout", "4h"), ("carry", "4h"), ("crowd", "4h")],
                 "equity_protocol.png", "Протокол: склеенные проверочные кварталы (справочный капитал, риск 5 %)")
    chart_real("real_10usdt.png")
    chart_diag("grid_diag.png")
    chart_risk("risk_levels.png")
    print("Графики записаны в", OUT)


if __name__ == "__main__":
    main()
