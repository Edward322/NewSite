"""Каждый кандидат этапа 3 проходит проверки причинности и даёт сделки."""
import numpy as np
import pytest

from bot.backtest.causality import perturbation_check, truncation_check
from bot.strategy.candidates import FAMILIES, grid_configs, make
from tests.helpers import MIN, T0, random_walk, run

MD = random_walk(25 * 1440, seed=21)
# Объём растёт с размером движения (как на реальном рынке) — нужен стратегии «shock».
MD.volume = 50 + 2e5 * np.abs(MD.close / MD.open - 1)
CUTS = [T0 + d * 1440 * MIN + o for d, o in ((8, 0), (13, 11 * MIN), (19, 47 * MIN))]


def _configs(family):
    g = grid_configs(family)
    return [g[0], g[len(g) // 2], g[-1]]


@pytest.mark.parametrize("family", list(FAMILIES))
def test_candidate_is_causal(family):
    for params in _configs(family):
        r = lambda md, p=params: run(md, make(family, "15m", p))  # noqa: E731
        assert truncation_check(r, MD, CUTS) == [], (family, params)
        assert perturbation_check(r, MD, CUTS) == [], (family, params)


@pytest.mark.parametrize("family", list(FAMILIES))
def test_candidate_trades(family):
    n = sum(len(run(MD, make(family, "15m", p), equity=1000.0).trades) for p in _configs(family))
    assert n > 0


def test_grid_sizes():
    sizes = {f: len(grid_configs(f)) for f in FAMILIES}
    assert sizes == {"donchian": 108, "momentum": 48, "meanrev": 81, "squeeze": 24,
                     "fade": 108, "pullback": 36, "shock": 36}


def test_unknown_param_rejected():
    with pytest.raises(ValueError):
        make("donchian", "1h", {"oops": 1})
