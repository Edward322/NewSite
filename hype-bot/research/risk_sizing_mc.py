"""Как размер риска на сделку влияет на результат (Монте-Карло, бутстрэп сделок).

Запуск: python -m research.risk_sizing_mc > reports/risk_sizing_mc.txt
Сделки — вне выборки из walk-forward этапа 3 (импульс 1h/4h, справочный прогон).
«Без перевеса» — те же сделки, сдвинутые так, что средний R равен средней
стоимости сделки без перевеса (−0,06R, см. stage3_random_baseline.txt).
Капитал после сделки: ×(1 + риск × R); 100 сделок; 10 000 путей.
Иллюстрация влияния размера риска, а не прогноз доходности.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bot.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "reports" / "stage3"
RISKS = (0.05, 0.10, 0.15)
N_TRADES, N_PATHS = 100, 10_000


def simulate(r: np.ndarray, risk: float, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    sample = rng.choice(r, size=(N_PATHS, N_TRADES), replace=True)
    growth = np.clip(1 + risk * sample, 0, None)        # изолированная маржа: не ниже нуля
    paths = np.cumprod(growth, axis=1)
    paths = np.concatenate([np.ones((N_PATHS, 1)), paths], axis=1)
    dd = 1 - (paths / np.maximum.accumulate(paths, axis=1)).min(axis=1)
    final = paths[:, -1]
    # остановка по просадке −80 % от пика: после неё торговля прекращается
    halted = dd >= 0.8
    return {"медиана итога": np.median(final), "P(убыток)": (final < 1).mean(),
            "P(остановка −80%)": halted.mean(), "медианная просадка": np.median(dd)}


def kelly(r: np.ndarray) -> float:
    fs = np.linspace(0.001, 0.6, 600)
    g = [np.mean(np.log(np.clip(1 + f * r, 1e-9, None))) for f in fs]
    return float(fs[int(np.argmax(g))])


def main():
    for tf in ("1h", "4h"):
        r = pd.read_parquet(OUT / f"slow_trades_ref_momentum_{tf}.parquet")["r_multiple"].to_numpy()
        zero = r - r.mean() - 0.06
        print(f"\n=== Импульс {tf}: {len(r)} сделок вне выборки, средний R {r.mean():+.3f}, "
              f"ст. откл. {r.std(ddof=1):.2f}, худшая {r.min():+.2f}R")
        print(f"Риск по Келли (максимум роста на этой выборке): {kelly(r):.1%}")
        for label, rr in (("как в выборке", r), ("без перевеса", zero)):
            for risk in RISKS:
                m = simulate(rr, risk)
                print(f"{label:14s} риск {risk:.0%}: " + "; ".join(
                    f"{k} {v:.2f}" if k == "медиана итога" else f"{k} {v:.0%}" for k, v in m.items()))
    print("\nСерия убытков подряд и капитал после неё:")
    for risk in RISKS:
        print(f"риск {risk:.0%}: 2 подряд {(1 - risk) ** 2 - 1:+.0%}, 4 подряд {(1 - risk) ** 4 - 1:+.0%}, "
              f"10 подряд {(1 - risk) ** 10 - 1:+.0%}")


if __name__ == "__main__":
    main()
