"""Этап 2: проверка бэктестера на реальных данных HYPEUSDT (только период разработки).

Запуск: python -m research.stage2_validation > reports/stage2_validation.txt

Стратегия здесь — тестовое пересечение скользящих средних (tests/strategies.py).
Это НЕ кандидат исследования: проверяется движок (учёт, причинность, скорость,
поведение стопов на реальных данных), а не прибыльность.
"""
from __future__ import annotations

import time

import numpy as np

from bot.backtest.audit import audit_trades
from bot.backtest.causality import perturbation_check, truncation_check
from bot.backtest.engine import Backtester, EngineConfig
from bot.config import load_config
from bot.data.bars import load_funding
from bot.data.split import dev_range, load_dev
from bot.market import Instrument
from tests.strategies import SmaCross

CFG = load_config()


def main() -> None:
    md = load_dev(CFG)
    f_ts, f_r = load_funding(CFG.data_dir(), CFG.symbol)
    inst = Instrument.from_files(CFG.data_dir(), CFG.symbol)
    d0, d1 = dev_range(CFG)
    print(f"Период разработки: {len(md):,} минут, {md.ts[0]} — {md.ts[-1]} (мс UTC); "
          f"отложенные данные не загружались")
    print(f"Издержки: {CFG.costs.model_dump()}")
    print(f"Риск: {CFG.risk.model_dump()}")

    for tf, fast, slow in (("15m", 20, 80), ("1h", 12, 48)):
        def run(m, tf=tf, fast=fast, slow=slow):
            ec = EngineConfig(risk=CFG.risk, costs=CFG.costs, initial_equity=CFG.risk.starting_equity_usdt)
            return Backtester(m, f_ts, f_r, inst, SmaCross(fast=fast, slow=slow, timeframe=tf), ec).run()

        t0 = time.perf_counter()
        res = run(md)
        dt = time.perf_counter() - t0
        tr = res.trades
        print(f"\n=== Таймфрейм {tf}: SmaCross({fast},{slow}) — тест движка")
        print(f"Время одного прогона: {dt:.2f} с; свечей: {len(res.bars):,}; сделок: {len(tr)}")
        print(f"Остановка по просадке: {res.halted}")
        if len(tr):
            print("Причины выхода:", tr["exit_reason"].value_counts().to_dict())
            print(f"Плечо: мин {tr.leverage.min():.2f}, медиана {tr.leverage.median():.2f}, макс {tr.leverage.max():.2f}")
            stops = tr[tr.exit_reason == "stop"]
            if len(stops):
                r = stops["net_pnl"] / stops["planned_loss"]
                print(f"Стоп-выходы: факт. убыток / план: среднее {-r.mean():.3f}, "
                      f"95-й перцентиль {-r.quantile(0.05):.3f}, худший {-r.min():.3f}")
            a = audit_trades(res, md, f_ts, f_r, CFG.costs, CFG.risk.starting_equity_usdt)
            print(f"Сверка учёта: макс |расхождение| вход {np.abs(a.d_entry).max():.2e}, "
                  f"финансирование {np.abs(a.d_funding).max():.2e}, итог сделки {np.abs(a.d_net).max():.2e}; "
                  f"капитал движок {a.attrs['final_equity_engine']:.6f} / пересчёт {a.attrs['final_equity_recomputed']:.6f}")
        skips = {}
        for _, why in res.skips:
            skips[why] = skips.get(why, 0) + 1
        print("Пропущенные входы:", skips or "нет")
        print("События риска:", [e.kind for e in res.events if e.kind != "day_start"][:20])
        cuts = [int(md.ts[0] + f * (md.ts[-1] - md.ts[0])) for f in (0.25, 0.5, 0.75)]
        t0 = time.perf_counter()
        problems = truncation_check(run, md, cuts) + perturbation_check(run, md, cuts)
        print(f"Проверка причинности (3 обрезки + 3 подмены будущего, {time.perf_counter() - t0:.1f} с): "
              f"{'нарушений нет' if not problems else problems}")


if __name__ == "__main__":
    main()
