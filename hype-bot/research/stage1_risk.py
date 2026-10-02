"""Этап 1: портфельный риск для пробоя канала 4h (правила — docs/RISK_PROTOCOL.md, записаны до прогона).

    python -m research.stage1_risk → reports/stage1_risk.txt, reports/stage1/*

1. Выбор риска на сделку: доля сигналов, пропущенных из-за минимального ордера при 25 USDT.
2. Варианты A / B / C на всей истории 2023-07-01 — конец данных (счёт 25 USDT и справочный 1000).
3. Выбор варианта по правилу протокола (наименьшая просадка; сложнее — только если лучше на ≥ 2 п.п.);
   если выбранный вариант доходит до остановки — риск снижается шагами по 0,25 п.п.
4. База для проверки на демо: распределение R сделок и сделок в неделю (reports/stage1/baseline.json).

ДАННЫЕ НЕ ЧИСТЫЕ: вся история, включая бывшие отложенные данные, уже использовалась в исследовании.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bot.backtest.metrics import path_metrics  # noqa: E402
from bot.backtest.portfolio import PortfolioBacktester  # noqa: E402
from bot.config import PROJECT_ROOT, Config, PortfolioCfg, RiskCfg, load_config  # noqa: E402
from bot.data.bars import to_ms  # noqa: E402
from bot.data.panel import aggregate_panel, load_basket_holdout  # noqa: E402
from bot.data.split import HOLDOUT_CONFIRM  # noqa: E402
from bot.risk.portfolio import PortfolioRisk  # noqa: E402
from bot.risk.sizing import SizingParams, Skip, size_position  # noqa: E402
from bot.runtime import make_strategy, portfolio_config  # noqa: E402
from bot.strategy.base import Enter  # noqa: E402
from research.basket_gates import basket_bh, daily_returns  # noqa: E402
from research.basket_report import INK2, NEUTRAL, S, legend, style  # noqa: E402

START = "2023-07-01"
DEPOSIT, REF = 25.0, 1000.0
RISKS = [0.0100, 0.0125, 0.0150, 0.0175, 0.0200]
MAX_SKIP = 0.20
OUT = PROJECT_ROOT / "reports" / "stage1"


def bot_config(base: Config, r: float, variant: str) -> Config:
    risk = RiskCfg(starting_equity_usdt=DEPOSIT, risk_per_trade=r, daily_loss_limit=0.06, max_drawdown=0.40,
                   drawdown_steps=[(0.20, 0.5)], loss_streak_pause_trades=None)
    pcfg = PortfolioCfg(max_positions=4, max_open_risk=0.06, downsize_to_fit=True,
                        corr_cap=0.04 if variant in ("B", "C") else None, vol_scaling=variant == "C")
    return base.model_copy(update={"risk": risk, "portfolio": pcfg})


class Recorder:
    """Обёртка стратегии: запоминает каждый вход, который стратегия выдала (монета без позиции)."""

    def __init__(self, s):
        self.s, self.signals = s, []
        self.name, self.timeframe = s.name, s.timeframe

    def params(self):
        return self.s.params()

    def prepare(self, bars):
        self.s.prepare(bars)
        self.warmup_bars = self.s.warmup_bars
        self.bars = bars

    def on_bar(self, k, positions):
        out = self.s.on_bar(k, positions)
        for r, d in out:
            if isinstance(d, Enter) and r not in positions:
                self.signals.append((k, r, d))
        return out


@dataclass
class Run:
    variant: str
    risk: float
    equity0: float
    res: object
    start: int

    @property
    def trades(self):
        return self.res.trades

    def eq(self):
        e = self.res.equity
        return e[e["ts"] > self.start]

    def max_dd(self) -> float:
        e = self.eq()["equity"]
        e = pd.concat([pd.Series([self.equity0]), e])
        return float((1 - e / e.cummax()).max())

    def pf(self) -> float:
        p = self.trades["net_pnl"] if len(self.trades) else pd.Series(dtype=float)
        loss = -p[p < 0].sum()
        return float(p[p > 0].sum() / loss) if loss > 0 else float("nan")

    def skip_counts(self) -> dict:
        sk = [s for s in self.res.skips if s[0] >= self.start]
        out = {}
        for _, _, why in sk:
            key = ("мин. ордер" if "ниже минимума биржи" in why else
                   "мин. ордер после уменьшения по лимиту" if "после уменьшения" in why else
                   "лимит суммарного/корреляционного риска" if "превышен лимит" in why else
                   "нет мест" if "мест" in why else
                   "ликвидность" if "ликвидности" in why else
                   "дневной лимит / остановка" if ("пауза" in why or "остановлен" in why) else why)
            out[key] = out.get(key, 0) + 1
        return out

    def n_signals(self) -> int:
        return sum(1 for d in self.res.decisions if d[0] >= self.start and d[2].startswith("Enter"))

    def half_risk_share(self) -> float:
        e = pd.concat([pd.Series([self.equity0]), self.eq()["equity"]])
        dd = 1 - e / e.cummax()
        return float((dd >= 0.20 - 1e-9).mean())


def run(panel, cfg: Config, variant: str, r: float, equity0: float, start: int) -> Run:
    c = bot_config(cfg, r, variant)
    res = PortfolioBacktester(panel, make_strategy(c), portfolio_config(c, equity0, start_ms=start)).run()
    return Run(variant, r, equity0, res, start)


def main() -> None:
    base = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    panel = load_basket_holdout(base, HOLDOUT_CONFIRM,
                                "этап 1: портфельный риск на всей истории (данные уже не чистые)")
    start = to_ms(START)
    end = int(panel.ts[-1]) + 15 * 60_000
    L = ["ЭТАП 1. Портфельный риск, пробой канала 4h с фиксированными параметрами (docs/RISK_PROTOCOL.md).",
         "ДАННЫЕ НЕ ЧИСТЫЕ: вся история уже использовалась в исследовании; это проверка правил риска, "
         "а не доказательство прибыльности.",
         f"Период торговли: {START} — {pd.Timestamp(end, unit='ms', tz='UTC'):%Y-%m-%d %H:%M} UTC "
         f"(данные с {pd.Timestamp(int(panel.ts[0]), unit='ms', tz='UTC'):%Y-%m-%d}; первые 6 месяцев — разогрев).",
         f"Параметры: {base.strategy.params}", ""]

    # ---------------------------------------------------------- 1. риск на сделку
    c0 = bot_config(base, 0.01, "A")
    rec = Recorder(make_strategy(c0))
    # без ограничений: каждая монета входит при каждом сигнале (сигнал = начало эпизода), без остановок
    unconstrained = c0.model_copy(update={
        "portfolio": c0.portfolio.model_copy(update={"max_positions": 20, "max_open_risk": 1.0, "liquidity": None}),
        "risk": RiskCfg(starting_equity_usdt=1e6, risk_per_trade=0.01, daily_loss_limit=0.98, max_drawdown=0.99)})
    PortfolioBacktester(panel, rec, portfolio_config(unconstrained, 1e6, start_ms=start)).run()
    bars = rec.bars
    prisk = PortfolioRisk(bars, c0.portfolio.rules())
    sig = [(k, r, d) for k, r, d in rec.signals if bars.close_ts[k] >= start and prisk.is_liquid(r, k)]
    n_all = sum(1 for k, r, d in rec.signals if bars.close_ts[k] >= start)
    inst = [panel.sym[s].inst for s in panel.symbols]
    dist = np.array([abs(bars.close[r, k] - d.stop) / bars.close[r, k] for k, r, d in sig])
    L.append(f"1. ВЫБОР РИСКА НА СДЕЛКУ. Сигналов стратегии за период: {n_all}, из них по монетам, прошедшим "
             f"правило ликвидности: {len(sig)}.")
    L.append(f"   Стоп в сигналах: медиана {np.median(dist) * 100:.1f}%, 90-й перцентиль "
             f"{np.percentile(dist, 90) * 100:.1f}%, максимум {dist.max() * 100:.1f}%.")
    L.append("   Объём считается size_position (тот же код, что в бэктесте) при капитале E, как будто других "
             "позиций нет.")
    L.append("   риск   E=25 USDT   E=20 USDT, половина риска   E=50 USDT")
    chosen_r, table = None, []
    for r in RISKS:
        sp = SizingParams(risk_per_trade=r, taker_fee=base.costs.taker_fee, slippage=base.costs.slippage,
                          liq_buffer=base.risk.liq_distance_vs_stop_min)
        shares = []
        for E, mult in ((DEPOSIT, 1.0), (20.0, 0.5), (50.0, 1.0)):
            skipped = 0
            for k, rr, d in sig:
                s = size_position(d.side, float(bars.close[rr, k]), d.stop, E, E, inst[rr], sp,
                                  risk_budget=E * r * mult)
                if isinstance(s, Skip) and "ниже минимума" in s.reason:
                    skipped += 1
            shares.append(skipped / len(sig))
        table.append((r, *shares))
        L.append(f"   {r * 100:5.2f}%   {shares[0] * 100:6.1f}%      {shares[1] * 100:6.1f}%                 "
                 f"{shares[2] * 100:6.1f}%")
        if chosen_r is None and shares[0] <= MAX_SKIP:
            chosen_r = r
    if chosen_r is None:
        chosen_r = RISKS[-1]
        L.append(f"   Ни один риск не проходит правило ≤ {MAX_SKIP:.0%}; берётся максимум диапазона 2 %.")
    L.append(f"   → Наименьший риск с пропуском ≤ {MAX_SKIP:.0%} сигналов при 25 USDT: {chosen_r * 100:.2f}%")
    L.append("")

    # ---------------------------------------------------------- 2. варианты A/B/C
    bh = basket_bh(start, end, base.costs.taker_fee, panel)
    bh = pd.concat([pd.Series([1.0], index=[bh.index[0] - pd.Timedelta(days=1)]), bh])
    bhm = path_metrics(bh)

    def describe(runs: dict[str, Run], r: float) -> None:
        L.append(f"   Риск {r * 100:.2f}% на сделку:")
        L.append("   вар.  счёт 25 USDT →  доходн.   макс.просадка  PF    сделок  Шарп  Кальмар  "
                 "полов. риск  остановка | справочный 1000 → (просадка, PF)")
        for v in ("A", "B", "C"):
            a, ref = runs[(v, DEPOSIT)], runs[(v, REF)]
            d = daily_returns(a.eq(), start, DEPOSIT)
            m = path_metrics(d)
            L.append(f"   {v}     {a.res.final_equity:8.2f}      {(a.res.final_equity / DEPOSIT - 1) * 100:+7.0f}%   "
                     f"{a.max_dd() * 100:6.1f}%        {a.pf():4.2f}  {len(a.trades):5d}   {m['sharpe']:4.2f}  "
                     f"{m['calmar']:5.2f}    {a.half_risk_share() * 100:5.1f}%     {'ДА' if a.res.halted else 'нет'}"
                     f"       | {ref.res.final_equity:8.0f} ({ref.max_dd() * 100:.0f}%, PF {ref.pf():.2f})")
        L.append("   Пропуски сигналов, счёт 25 USDT (доля от всех сигналов-решений стратегии; один и тот же "
                 "пробой может давать сигнал несколько свечей подряд):")
        for v in ("A", "B", "C"):
            a = runs[(v, DEPOSIT)]
            n = a.n_signals()
            sk = a.skip_counts()
            parts = ", ".join(f"{k} {c / n * 100:.1f}%" for k, c in sorted(sk.items(), key=lambda x: -x[1]))
            L.append(f"   {v}: сигналов {n}, открыто сделок {len(a.trades)}; пропуски: {parts}")

    r = chosen_r
    final = None
    while True:
        runs = {(v, E): run(panel, base, v, r, E, start) for v in ("A", "B", "C") for E in (DEPOSIT, REF)}
        L.append("2. ВАРИАНТЫ НА ВСЕЙ ИСТОРИИ (данные не чистые).")
        describe(runs, r)
        dd = {v: runs[(v, DEPOSIT)].max_dd() for v in ("A", "B", "C")}
        pick = "A"
        if dd["B"] <= dd["A"] - 0.02:
            pick = "B"
        if dd["C"] <= dd[pick] - 0.02:
            pick = "C"
        L.append(f"   Правило выбора: наименьшая просадка, сложнее — только при выигрыше ≥ 2 п.п. "
                 f"(A {dd['A'] * 100:.1f}%, B {dd['B'] * 100:.1f}%, C {dd['C'] * 100:.1f}%) → вариант {pick}")
        if runs[(pick, DEPOSIT)].res.halted and r > 0.0025 + 1e-12:
            L.append(f"   Вариант {pick} доходит до остановки при просадке 40 % → риск снижается на 0,25 п.п.")
            L.append("")
            r = round(r - 0.0025, 4)
            continue
        final = (pick, r, runs)
        break
    pick, r, runs = final
    L.append("")
    L.append(f"ИТОГ ЭТАПА 1: вариант {pick}, риск {r * 100:.2f}% на сделку, депозит {DEPOSIT:g} USDT.")
    L.append(f"Buy & hold корзины за тот же период: доходность {bhm['total_return'] * 100:+.0f}%, макс. просадка "
             f"{bhm['max_drawdown'] * 100:.0f}%, Шарп {bhm['sharpe']:.2f}.")

    # ---------------------------------------------------------- 3. база для демо
    best = runs[(pick, DEPOSIT)]
    ref = runs[(pick, REF)]
    tr = ref.trades
    weeks = (end - start) / (7 * 86_400_000)
    yearly = []
    for y, t in tr.groupby(pd.to_datetime(tr["entry_ts"], unit="ms", utc=True).dt.year):
        yearly.append(f"{y}: сделок {len(t)}, ср. R {t['r_multiple'].mean():+.3f}")
    L.append(f"Время в режиме половинного риска (просадка ≥ 20 %): счёт 25 USDT {best.half_risk_share() * 100:.0f}%, "
             f"справочный капитал {ref.half_risk_share() * 100:.0f}%. При 25 USDT половинный риск почти всегда "
             "меньше минимального ордера (таблица в п. 1), поэтому в этом режиме бот почти не торгует.")
    liq = prisk.liquid[:, bars.close_ts >= start].mean(axis=1)
    L.append("Доля времени, когда монета проходит правило ликвидности: " +
             ", ".join(f"{s.replace('USDT', '')} {v * 100:.0f}%" for s, v in zip(panel.symbols, liq)))
    info = run(panel, base, pick, r, 50.0, start)
    L.append(f"Для информации (не выбор): вариант {pick}, депозит 50 USDT → {info.res.final_equity:.2f} "
             f"({(info.res.final_equity / 50 - 1) * 100:+.0f}%), макс. просадка {info.max_dd() * 100:.1f}%, "
             f"сделок {len(info.trades)}, половинный риск {info.half_risk_share() * 100:.0f}% времени"
             f"{', ОСТАНОВЛЕН' if info.res.halted else ''}.")
    L.append(f"Справочный капитал, вариант {pick}: сделок {len(tr)} ({len(tr) / weeks:.1f} в неделю), "
             f"средний R {tr['r_multiple'].mean():+.3f}; " + "; ".join(yearly))
    base_json = {
        "variant": pick, "risk_per_trade": r, "deposit": DEPOSIT, "start": START,
        "end": pd.Timestamp(end, unit="ms", tz="UTC").isoformat(),
        "params": base.strategy.params, "trades_per_week": len(tr) / weeks,
        "r_multiples": [round(float(x), 5) for x in tr["r_multiple"]],
        "real_trades_per_week": len(best.trades) / weeks,
        "real_pnl_pct": [round(float(p / (e - p)), 6) for p, e in zip(best.trades["net_pnl"], best.trades["equity_after"])],
        "costs": base.costs.model_dump(),
        "note": "данные не чистые; база только для проверки «нет явного провала» на демо",
    }
    (OUT / "baseline.json").write_text(json.dumps(base_json, ensure_ascii=False, indent=1), encoding="utf-8")
    best.trades.to_csv(OUT / f"trades_{pick}_25usdt.csv", index=False)

    # график
    fig, ax = plt.subplots(figsize=(9, 4.6))
    style(ax, f"Счёт 25 USDT: пробой 4h, риск {r * 100:.2f} %, варианты A/B/C (данные не чистые)", "USDT")
    for i, v in enumerate(("A", "B", "C")):
        d = daily_returns(runs[(v, DEPOSIT)].eq(), start, DEPOSIT)
        ax.plot(d.index, d.values, color=S[i], linewidth=2 if v == pick else 1.2, label=f"вариант {v}")
        ax.annotate(f"{d.iloc[-1]:.1f}", (d.index[-1], d.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    color=INK2, fontsize=8, va="center")
    ax.plot(bh.index, bh.values * DEPOSIT, color=NEUTRAL, linewidth=1.5, linestyle="--", label="buy & hold корзины")
    ax.axhline(DEPOSIT, color=NEUTRAL, linewidth=1)
    ax.set_yscale("log")
    legend(ax)
    fig.tight_layout()
    fig.savefig(OUT / "equity_variants.png", dpi=150)
    plt.close(fig)

    text = "\n".join(L) + "\n"
    (PROJECT_ROOT / "reports" / "stage1_risk.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
